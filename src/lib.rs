use anyhow::Error;
use std::path::{Path, PathBuf};
use std::time;
use tracing::debug;
use video_rs::decode::Decoder;

pub mod detection;
mod inference;
mod loaders;
mod runtime;
#[cfg(feature = "visualize")]
mod vizualize;

//use inference::detect_frame;

use crate::detection::BoundingBox;
use crate::runtime::RuntimeBuilder;

/// All configuration options for `detect_video` bundled in one struct.
///
/// # Examples
///
/// Go with the default settings:
/// ```
/// use video_inference::DetectionConfig;
/// let config = DetectionConfig::default();
/// ```
///
/// Specify a confidence threshold other than default:
/// ```
/// use video_inference::DetectionConfig;
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

    /// Output path for annotated video (requires `visualize` feature)
    pub path_output: Option<PathBuf>,

    /// ONNX-model image input tensor name
    pub input_tensor_name: String,

    /// ONNX-model results output tensor name
    pub output_tensor_name: String,
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
            path_output: None,
        }
    }
}

pub fn detect_video(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    let mut runtime = RuntimeBuilder::from(config)
        .model(path_onnx)
        .video(path_video)
        .build()?;
    let results = runtime.detect_video()?;
    Ok(results)
}

#[cfg(feature = "visualize")]
pub fn annotate_video(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    let mut runtime = RuntimeBuilder::from(config)
        .model(path_onnx)
        .video(path_video)
        .build()?;
    let results = runtime.annotate_video()?;
    Ok(results)
}

/// Debug fn to check decoding speed
pub fn iterate_video(path_video: impl AsRef<Path>) -> Result<(), Error> {
    let mut decoder = Decoder::new(path_video.as_ref())?;
    let t = time::Instant::now();
    for next_frame in decoder.decode_iter() {
        if let Ok((ts, frame)) = next_frame {
            debug!("{} | {:?}", ts, frame.shape());
        } else {
            break;
        }
    }
    debug!("iterate: {:?}", t.elapsed());
    Ok(())
}
