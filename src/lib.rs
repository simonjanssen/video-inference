use anyhow::Error;
use fast_image_resize as fr;
use std::path::Path;
use std::sync::Once;
use std::time;
use tracing::debug;
use video_rs::decode::Decoder;

pub mod detection;
mod inference;
mod loaders;

use inference::detect_frame;

use crate::{
    detection::BoundingBox,
    inference::{detect_input_shape, load_session},
};

static INIT_VIDEO_RS: Once = Once::new();

fn init_video_rs() {
    INIT_VIDEO_RS.call_once(|| {
        video_rs::init().expect("failed to initialize video-rs (FFmpeg)");
    });
}

/// All configuration options for `detect_video` bundled in one struct.
///
/// # Examples
///
/// Go with the default settings:
/// ```
/// let config = DetectionConfig::default();
/// ```
///
/// Specify a confidence threshold other than default:
/// ```
/// let config = DetectionConfig { conf_thres: 0.5, ..Default::default() };
/// ```
#[derive(Debug)]
pub struct DetectionConfig {
    /// Confidence threshold for detections:
    /// - Choose between 0.0 and 1.0
    /// - Higher is more restrictive
    pub conf_thres: f32,

    /// Intersection-Over-Union: filter out overlapping detections
    pub iou_thres: f32,

    /// Maximum number of detections per frame
    pub max_detect: usize,

    /// Detection Interval
    /// - Run detection every ..th second
    /// - Defaults to None (run detection on all frames)
    pub interval: Option<f32>,

    /// ONNX-model image input tensor name
    pub input_tensor_name: String,

    /// ONNX-model results output tensor name
    pub output_tensor_name: String,
    //pub rate_sec: Option<f32>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            conf_thres: 0.25,
            iou_thres: 0.4,
            max_detect: 300,
            interval: None,
            input_tensor_name: "images".to_string(),
            output_tensor_name: "output0".to_string(),
            //rate_sec: None,
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
    pub output_tensor_name: String,
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
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    init_video_rs();
    let path_video = path_video.as_ref();
    let video_name = path_video.file_name().unwrap().to_str().unwrap();
    //let t = time::Instant::now();
    let mut session = load_session(path_onnx)?;
    let (input_width, input_height) = detect_input_shape(&session, &config.input_tensor_name)?;

    let mut decoder = Decoder::new(path_video.to_owned())?;
    let (target_width, target_height) = decoder.size();
    let n_frames = decoder.frames()?;

    let interval_frames = match config.interval {
        Some(interval_sec) => {
            let duration_sec = decoder.duration()?.as_secs();
            if duration_sec > 0.0 && n_frames > 0 {
                ((n_frames as f32 / duration_sec) * interval_sec)
                    .round()
                    .max(1.0) as usize
            } else {
                1
            }
        }
        None => 1,
    };
    debug!("running detection every {}th frame", interval_frames);

    let inner_config = Config {
        conf_thres: config.conf_thres,
        iou_thres: config.iou_thres,
        max_detect: config.max_detect,
        output_tensor_name: config.output_tensor_name.clone(),
        input_height,
        input_width,
        target_height,
        target_width,
    };
    debug!("{:?}", config);

    let mut resizer = fr::Resizer::new();

    //debug!("{:?}", resolved);

    let mut dst_image = fr::images::Image::new(
        inner_config.input_width,
        inner_config.input_height,
        fr::PixelType::U8x3,
    );

    let capacity = if interval_frames > 1 {
        n_frames as usize / interval_frames + 1
    } else {
        n_frames as usize
    };
    let mut bboxes = Vec::with_capacity(capacity);
    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            if f % interval_frames == 0 {
                debug!("{}/{}", f, n_frames);
                let bboxes_frame = detect_frame(
                    &mut session,
                    frame,
                    &inner_config,
                    &mut resizer,
                    &mut dst_image,
                )?;
                bboxes.push(bboxes_frame);
            }
        } else {
            break;
        }
    }
    debug!("{}: {:?}", video_name, t.elapsed());
    Ok(bboxes)
}
