use anyhow::{Error, anyhow};
use clap::Parser;
use fast_image_resize as fr;
use ndarray::{Array3, ArrayView1, Axis, s};
use ort::{
    ep::CoreML,
    inputs,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::{Tensor, TensorValueType, Value},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{cmp::Ordering, path::PathBuf, time};
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

#[derive(Debug)]
struct Config {
    // confidence threshold for detections
    conf_thres: f32,
    // intersection-over-unition threshold for non-maxima supression
    iou_thres: f32,
    // maximum number of detections per image
    max_detect: usize,
    // onnx-model input width
    input_width: u32,
    // onnx-model input height
    input_height: u32,
    // image/video-frame width
    target_width: u32,
    // image/video-frame height
    target_height: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            conf_thres: 0.25,
            iou_thres: 0.4,
            max_detect: 300,
            input_width: 640,
            input_height: 640,
            target_width: 640,
            target_height: 640,
        }
    }
}

#[derive(Default, Clone, Debug, Copy)]
pub struct BoundingBox {
    pub x1: f32, // left
    pub y1: f32, // top
    pub x2: f32, // right
    pub y2: f32, // bottom
    pub score: f32,
    pub class_idx: i32,
}

impl BoundingBox {
    /// center-x, center-y, width, height
    pub fn xywh(&self) -> (u32, u32, u32, u32) {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        let x = (self.x2 + self.x1) / 2.0;
        let y = (self.y2 + self.y1) / 2.0;
        (x as u32, y as u32, w as u32, h as u32)
    }

    /// left, top, width, height
    pub fn x1y1wh(&self) -> (u32, u32, u32, u32) {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        (self.x1 as u32, self.y1 as u32, w as u32, h as u32)
    }

    pub fn area(&self) -> f32 {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        if w > 0.0 && h > 0.0 { w * h } else { 0.0 }
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1_inter = self.x1.max(other.x1);
        let y1_inter = self.y1.max(other.y1);
        let x2_inter = self.x2.min(other.x2);
        let y2_inter = self.y2.min(other.y2);

        let w_inter = x2_inter - x1_inter;
        let h_inter = y2_inter - y1_inter;

        let intersection = if w_inter > 0.0 && h_inter > 0.0 {
            w_inter * h_inter
        } else {
            0.0
        };

        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    pub fn from_array(array: ArrayView1<f32>) -> Self {
        let bbox_xywh = array.slice(s![..4]).to_vec();
        let confs = array.slice(s![4..]).to_vec();
        let (class_idx, conf) = confs
            .iter()
            .enumerate()
            .filter_map(
                |(idx, &num)| {
                    if num.is_nan() { None } else { Some((idx, num)) }
                },
            )
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .unwrap();
        let (x1, y1, x2, y2) =
            xywh_to_xyxy(&bbox_xywh[0], &bbox_xywh[1], &bbox_xywh[2], &bbox_xywh[3]);
        Self {
            x1,
            y1,
            x2,
            y2,
            score: conf,
            class_idx: class_idx as i32,
        }
    }

    pub fn scale(&mut self, scale_w: f32, scale_h: f32) {
        self.x1 *= scale_w;
        self.y1 *= scale_h;
        self.x2 *= scale_w;
        self.y2 *= scale_h;
    }
}

fn xywh_to_xyxy(x: &f32, y: &f32, w: &f32, h: &f32) -> (f32, f32, f32, f32) {
    let x1 = x - w / 2.0;
    let y1 = y - h / 2.0;
    let x2 = x + w / 2.0;
    let y2 = y + h / 2.0;
    (x1, y1, x2, y2)
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
    config: &Config,
    resizer: &mut fr::Resizer,
    dst_image: &mut fr::images::Image,
) -> Result<Value<TensorValueType<f32>>, Error> {
    let (raw, _) = img_arr
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset();
    let src_image = fr::images::Image::from_vec_u8(
        config.target_width,
        config.target_height,
        raw,
        fr::PixelType::U8x3,
    )?;
    resizer.resize(&src_image, dst_image, None)?;

    // Single-pass: normalize u8→f32 and transpose HWC→CHW
    let src = dst_image.buffer();
    let pixels = (config.input_width * config.input_height) as usize;
    let mut chw = vec![0.0f32; 3 * pixels];
    for i in 0..pixels {
        chw[i] = src[i * 3] as f32 / 255.0;
        chw[pixels + i] = src[i * 3 + 1] as f32 / 255.0;
        chw[2 * pixels + i] = src[i * 3 + 2] as f32 / 255.0;
    }

    let arr = ndarray::Array::from_shape_vec(
        (
            1,
            3,
            config.input_height as usize,
            config.input_width as usize,
        ),
        chw,
    )?;
    let tensor = Tensor::from_array(arr)?;
    Ok(tensor)
}

fn run_inference(
    session: &mut Session,
    img_arr: Array3<u8>,
    config: &Config,
    resizer: &mut fr::Resizer,
    dst_image: &mut fr::images::Image,
) -> Result<Vec<BoundingBox>, Error> {
    let t = time::Instant::now();
    let image_tensor = load_resized_tensor(img_arr, config, resizer, dst_image)?;
    let session_inputs = inputs! {
        "images" => image_tensor
    };
    let dt_preprocess = t.elapsed();
    let t = time::Instant::now();
    let session_outputs = session.run(session_inputs)?;
    let dt_inference = t.elapsed();
    let t = time::Instant::now();
    let bboxes = postprocess(session_outputs, config)?;
    let dt_postprocess = t.elapsed();

    let class_idxs: Vec<_> = bboxes.iter().map(|b| b.class_idx).collect();
    debug!(
        "pre={:.1?} inf={:.1?} post={:.1?} classes={:?}",
        dt_preprocess, dt_inference, dt_postprocess, class_idxs
    );

    Ok(bboxes)
}

/// Class-Sensitive Non Maxima Suppression for Overlapping Bounding Boxes
/// Iteratively removes lower scoring bboxes which have an IoU above iou_thresold.
/// Inspired by: https://pytorch.org/vision/master/_modules/torchvision/ops/boxes.html#nms
pub fn nms(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Compute the maximum coordinate value among all boxes
    let max_coordinate = boxes.iter().fold(0.0_f32, |max_coord, bbox| {
        max_coord.max(bbox.x2).max(bbox.y2)
    });
    let offset = max_coordinate + 1.0;

    // Create a vector of shifted boxes with their original indices
    let mut boxes_shifted: Vec<(BoundingBox, usize)> = boxes
        .iter()
        .enumerate()
        .map(|(i, bbox)| {
            let class_offset = offset * bbox.class_idx as f32;
            let shifted_bbox = BoundingBox {
                x1: bbox.x1 + class_offset,
                y1: bbox.y1 + class_offset,
                x2: bbox.x2 + class_offset,
                y2: bbox.y2 + class_offset,
                score: bbox.score,
                class_idx: bbox.class_idx, // Keep class_idx the same
            };
            (shifted_bbox, i) // Keep track of the original index
        })
        .collect();

    // Sort boxes in decreasing order based on scores
    boxes_shifted
        .sort_unstable_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap_or(Ordering::Equal));

    let mut keep_indices = Vec::new();

    while let Some((current_box, original_index)) = boxes_shifted.first().cloned() {
        keep_indices.push(original_index);
        boxes_shifted.remove(0);

        // Retain boxes that have an IoU less than or equal to the threshold with the current box
        boxes_shifted.retain(|(bbox, _)| current_box.iou(bbox) <= iou_threshold);
    }

    // Collect the kept boxes from the original input
    let mut kept_boxes: Vec<BoundingBox> = keep_indices.into_iter().map(|idx| boxes[idx]).collect();

    // Sort the kept boxes in decreasing order of their scores
    kept_boxes.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    kept_boxes
}

fn postprocess(
    session_outputs: SessionOutputs<'_>,
    config: &Config,
) -> Result<Vec<BoundingBox>, Error> {
    let output = session_outputs["output0"].try_extract_array::<f32>()?;
    let view_candidates = output.slice(s![0, 4.., ..]);
    let mask_candidates: Vec<bool> = view_candidates
        .axis_iter(Axis(1))
        .map(|col| col.iter().cloned().fold(f32::NEG_INFINITY, f32::max) > config.conf_thres)
        .collect();
    let idx_candidates: Vec<usize> = mask_candidates
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();
    let candidates_image = output.select(Axis(2), &idx_candidates);
    // Remove only the batch dim; squeeze() would collapse a single-candidate axis and panic
    let candidates_image = candidates_image.index_axis(Axis(0), 0);
    //let mut bboxes = Vec::new();
    let mut bboxes: Vec<BoundingBox> = Vec::with_capacity(candidates_image.len_of(Axis(1)));
    for candidate in candidates_image.axis_iter(Axis(1)) {
        let bbox = BoundingBox::from_array(candidate.to_shape(candidate.len()).unwrap().view());
        bboxes.push(bbox);
    }
    let mut bboxes = nms(&bboxes, config.iou_thres);
    bboxes.truncate(config.max_detect); // keep only max detections
    //println!("len bboxes nms: {:?}", bboxes.len());

    let (base_w, base_h) = (config.input_width as f32, config.input_height as f32);
    let (target_w, target_h) = (config.target_width as f32, config.target_height as f32);
    let scale_w = target_w / base_w;
    let scale_h = target_h / base_h;
    for bbox in &mut bboxes {
        bbox.scale(scale_w, scale_h);
    }

    Ok(bboxes)
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
    let (target_width, target_height) = decoder.size();
    let n_frames = decoder.frames()?;

    let mut resizer = fr::Resizer::new();
    let config = Config {
        target_width,
        target_height,
        ..Default::default()
    };
    debug!("{:?}", config);
    
    let mut dst_image =
        fr::images::Image::new(config.input_width, config.input_height, fr::PixelType::U8x3);

    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            debug!("{}/{}", f, n_frames);
            let _bboxes =
                run_inference(&mut session, frame, &config, &mut resizer, &mut dst_image)?;
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
    let path_source = args.source;
    let paths_mp4s: Vec<PathBuf> = if path_source.is_dir() {
        std::fs::read_dir(&path_source)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension()?.to_str()? == "mp4").then_some(path)
            })
            .collect()
    } else if path_source.extension().and_then(|e| e.to_str()) == Some("mp4") {
        vec![path_source]
    } else {
        return Err(anyhow!("Source must be a directory or an .mp4 file"));
    };
    let _results: Vec<_> = paths_mp4s
        .par_iter()
        .map(|path_mp4| run_video(path_mp4, &checkpoint))
        .collect();
    Ok(())
}
