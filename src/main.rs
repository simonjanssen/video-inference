use anyhow::{Error, anyhow};
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

fn run_video(path_video: &PathBuf) -> Result<(), Error> {
    let video_name = path_video.file_name().unwrap().to_str().unwrap();
    //let t = time::Instant::now();
    let path_onnx = PathBuf::from("/Users/simon/repos/portal/checkpoints/yolo11n.onnx");
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_intra_threads(4).unwrap()
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
            run_inference(&mut session, frame, width, height, &mut resizer, &mut dst_image)?;
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

    let path_video = PathBuf::from("data/source/1775170800.000-192.168.178.59-00c0.mp4");
    run_video(&path_video)?;

    Ok(())
}
