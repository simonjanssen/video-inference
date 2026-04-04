use anyhow::Error;
use fast_image_resize as fr;
use std::path::Path;
use std::time;
use tracing::debug;
use video_rs::decode::Decoder;

mod detection;
mod inference;
mod loaders;

use inference::detect_frame;

use crate::inference::{detect_input_shape, load_session};

const ONNX_INPUT_TENSOR_NAME: &str = "images";

#[derive(Debug)]
pub struct DetectionConfig {
    pub conf_thres: f32,
    pub iou_thres: f32,
    pub max_detect: usize,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            conf_thres: 0.25,
            iou_thres: 0.4,
            max_detect: 300,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Config {
    pub conf_thres: f32,
    pub iou_thres: f32,
    pub max_detect: usize,
    pub input_width: u32,
    pub input_height: u32,
    pub target_width: u32,
    pub target_height: u32,
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
pub fn detect_video(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<(), Error> {
    println!("hi");
    let path_video = path_video.as_ref();
    let video_name = path_video.file_name().unwrap().to_str().unwrap();
    //let t = time::Instant::now();
    let mut session = load_session(path_onnx)?;
    let (input_width, input_height) = detect_input_shape(&session, ONNX_INPUT_TENSOR_NAME)?;

    let mut decoder = Decoder::new(path_video.to_owned())?;
    let (target_width, target_height) = decoder.size();
    let n_frames = decoder.frames()?;

    let resolved = Config {
        conf_thres: config.conf_thres,
        iou_thres: config.iou_thres,
        max_detect: config.max_detect,
        input_height,
        input_width,
        target_height,
        target_width,
    };
    debug!("{:?}", config);

    let mut resizer = fr::Resizer::new();

    debug!("{:?}", resolved);

    let mut dst_image = fr::images::Image::new(
        resolved.input_width,
        resolved.input_height,
        fr::PixelType::U8x3,
    );

    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            debug!("{}/{}", f, n_frames);
            let _bboxes =
                detect_frame(&mut session, frame, &resolved, &mut resizer, &mut dst_image)?;
        } else {
            break;
        }
    }
    debug!("{}: {:?}", video_name, t.elapsed());
    Ok(())
}
