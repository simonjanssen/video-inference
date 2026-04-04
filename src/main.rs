use anyhow::{Error, anyhow};
use clap::Parser;
use fast_image_resize as fr;
use ndarray::Array3;
use ort::{
    ep::CoreML,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::{Tensor, TensorValueType, Value},
};
use std::{path::PathBuf, time};
use tracing::debug;
use tracing_subscriber::EnvFilter;
use video_rs::decode::Decoder;

#[derive(Parser)]
struct Args {
    /// Path to the source video file
    #[arg(short, long)]
    source: PathBuf,

    /// Path to the ONNX checkpoint file (falls back to ONNX_CHECKPOINT_PATH in .env)
    #[arg(short, long)]
    checkpoint: Option<PathBuf>,
}

fn resolve_checkpoint(args: &Args) -> Result<PathBuf, Error> {
    let path = match &args.checkpoint {
        Some(p) => p.clone(),
        None => {
            let _ = dotenvy::dotenv();
            let val = std::env::var("ONNX_CHECKPOINT_PATH").map_err(|_| {
                anyhow!("No --checkpoint arg and ONNX_CHECKPOINT_PATH not set in .env")
            })?;
            PathBuf::from(val)
        }
    };
    if !path.is_file() {
        return Err(anyhow!(
            "Checkpoint path does not exist or is not a file: {}",
            path.display()
        ));
    }
    Ok(path)
}

pub fn determine_input_shape(session: &Session, input_name: &str) -> Result<(u32, u32), Error> {
    let inputs = session.inputs();
    debug!("{:?}", inputs);
    for input in inputs {
        if input.name() == input_name
            && let Some(dims) = input.dtype().tensor_shape()
        {
            let d = dims.len();
            if d > 1 {
                let (w, h) = (dims[d - 2], dims[d - 1]);
                return Ok((w as u32, h as u32));
            }
        }
    }
    Err(anyhow!("Failed to determine input shape!"))
}

/// Resize a decoded video frame and produce a normalized NCHW f32 tensor for inference.
///
/// Optimizations:
/// - Uses `fast_image_resize` (SIMD-accelerated) instead of the `image` crate's pure-Rust
///   resizer, which is ~5-10x faster for bilinear interpolation.
/// - The `resizer` and `dst_image` are pre-allocated by the caller and reused across frames
///   to avoid per-frame heap allocations (~1.2MB each) and to let the resizer cache its
///   interpolation coefficient tables for the given src→dst size pair.
/// - Normalization (u8→f32, /255) and HWC→CHW transpose are fused into a single pass,
///   eliminating an intermediate f32 HWC buffer and the non-contiguous copy that
///   `ndarray::permuted_axes` + `Tensor::from_array` would otherwise require.
fn load_resized_tensor(
    img_arr: Array3<u8>,
    src_height: u32,
    src_width: u32,
    resizer: &mut fr::Resizer,
    dst_image: &mut fr::images::Image,
) -> Result<Value<TensorValueType<f32>>, Error> {
    let (raw, _) = img_arr
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset();
    let src_image =
        fr::images::Image::from_vec_u8(src_width, src_height, raw, fr::PixelType::U8x3)?;
    resizer.resize(&src_image, dst_image, None)?;

    // Single-pass: normalize u8→f32 and transpose HWC→CHW
    let src = dst_image.buffer();
    let pixels = 640 * 640;
    let mut chw = vec![0.0f32; 3 * pixels];
    for i in 0..pixels {
        chw[i] = src[i * 3] as f32 / 255.0;
        chw[pixels + i] = src[i * 3 + 1] as f32 / 255.0;
        chw[2 * pixels + i] = src[i * 3 + 2] as f32 / 255.0;
    }

    let arr = ndarray::Array::from_shape_vec((1, 3, 640, 640), chw)?;
    let tensor = Tensor::from_array(arr)?;
    Ok(tensor)
}

fn run_inference(
    session: &mut Session,
    img_arr: Array3<u8>,
    width: u32,
    height: u32,
    resizer: &mut fr::Resizer,
    dst_image: &mut fr::images::Image,
) -> Result<(), Error> {
    let image_tensor = load_resized_tensor(img_arr, height, width, resizer, dst_image)?;
    let session_inputs = inputs! {
        "images" => image_tensor
    };
    let t = time::Instant::now();
    let _session_outputs = session.run(session_inputs)?;
    debug!("inference: {:?}", t.elapsed());
    Ok(())
}

/// Decode a video file and run per-frame inference.
///
/// Session configuration optimizations:
/// - **CoreML EP with CPU + Neural Engine**: Offloads inference to Apple's ANE/GPU instead
///   of the default CPU-only ONNX Runtime provider. This alone reduced inference from ~80ms
///   to ~25ms per frame.
/// - **Fixed-size ONNX model**: Using a static-shape model instead of a
///   dynamic-shape variant lets CoreML fully compile and optimize the graph for ANE,
///   bringing inference down to ~11ms per frame.
/// - **GraphOptimizationLevel::Level3**: Enables all ORT graph optimizations (constant
///   folding, operator fusion, etc.).
/// - **Intra-op threads**: Parallelizes CPU-bound ORT operations across 4 threads.
///
/// Pre-allocates the `fast_image_resize` Resizer and destination image buffer outside the
/// frame loop so they are reused across frames (see [`load_resized_tensor`]).
fn run_video(path_video: &PathBuf, path_onnx: &PathBuf) -> Result<(), Error> {
    let video_name = path_video.file_name().unwrap().to_str().unwrap();
    //let t = time::Instant::now();
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .with_execution_providers([CoreML::default()
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .build()])
        .unwrap()
        .commit_from_file(path_onnx)?;

    let mut decoder = Decoder::new(path_video.to_owned())?;
    let (width, height) = decoder.size();
    let n_frames = decoder.frames()?;

    let mut resizer = fr::Resizer::new();
    let mut dst_image = fr::images::Image::new(640, 640, fr::PixelType::U8x3);

    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            let tf = time::Instant::now();
            run_inference(
                &mut session,
                frame,
                width,
                height,
                &mut resizer,
                &mut dst_image,
            )?;
            debug!("{}/{}: {:?}", f + 1, n_frames, tf.elapsed());
        } else {
            break;
        }
    }
    debug!("{}: {:?}", video_name, t.elapsed());
    Ok(())
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();

    video_rs::init().unwrap();

    let args = Args::parse();
    let checkpoint = resolve_checkpoint(&args)?;
    run_video(&args.source, &checkpoint)?;

    Ok(())
}
