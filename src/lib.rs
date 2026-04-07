use anyhow::{Error, anyhow};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::{thread, time};
use tracing::debug;
use video_rs::decode::Decoder;

pub mod detection;
mod inference;
mod loaders;
mod parallel;
mod runtime;
#[cfg(feature = "visualize")]
mod vizualize;

//use inference::detect_frame;

use crate::detection::BoundingBox;
use crate::parallel::{DetectionTask, detection_handler};
use crate::runtime::{RuntimeBuilder, calc_interval_frames, init_video_rs};

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
#[derive(Debug, Clone)]
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

/// Run video-detection on mp4-video
///
/// # Example
/// ```
/// use video_inference::{DetectionConfig, detect_video};
/// let config = DetectionConfig {interval: Some(1.0), ..Default::default()};
/// let path_video = "./tests/assets/video.mp4";
/// let path_onnx = "./tests/assets/model.onnx";
/// let bboxes = detect_video(path_video, path_onnx, &config)?;
/// ```
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

/// Run video-detection and write back detections as video with bounding boxes
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

/// Run video-decoding frame by frame
///
/// This is for testing purposes only to measures the decoder-runtime on different targets
pub fn iterate_video(path_video: impl AsRef<Path>) -> Result<(), Error> {
    init_video_rs();
    let mut decoder = Decoder::new(path_video.as_ref())?;
    let t = time::Instant::now();
    //let mut tf = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((ts, _frame)) = next_frame {
            debug!("{} | {:?}", f, ts.as_secs());
        } else {
            break;
        }
        //tf = time::Instant::now();
    }
    debug!("iterate: {:?}", t.elapsed());
    Ok(())
}

/// Sample video frames at regular intervals using keyframe-based seeking.
///
/// Instead of decoding every frame sequentially, this function seeks directly to
/// target positions in the video, skipping intermediate frames entirely. Because
/// seeking lands on the nearest keyframe *at or before* the target, the actual
/// sampling granularity is limited by the video's keyframe interval (typically
/// every few seconds). Duplicate keyframes from consecutive seeks are
/// deduplicated so each unique frame is only processed once.
///
/// Hopefully, this is significantly faster than sequential decoding when the desired
/// interval is larger than the keyframe spacing, at the cost of frame-exact
/// positioning.
pub fn iterate_video_keyframes(path_video: impl AsRef<Path>) -> Result<(), Error> {
    init_video_rs();
    let mut decoder = Decoder::new(path_video.as_ref())?;
    let t = time::Instant::now();
    //let mut tf = time::Instant::now();
    //let (w, h) = decoder.size();
    let n_frames = decoder.frames()? as i64;
    let fps = decoder.frame_rate();
    let interval = 10;
    let mut last_ts: f32 = -1.0;
    for f in (0i64..n_frames).step_by(interval) {
        let target_ms = (f as f64 / fps as f64 * 1000.0) as i64;
        let Ok(()) = decoder.seek(target_ms) else {
            break;
        };
        let Ok((ts, _frame)) = decoder.decode() else {
            break;
        };
        // skip if seek landed on the same keyframe as last iteration
        if ts.as_secs() == last_ts {
            //debug!("skipping");
            continue;
        }
        last_ts = ts.as_secs();
        debug!("{} / {}", f, ts.as_secs());
        // let (raw, _) = frame.into_raw_vec_and_offset();
        // let rgb_img = image::RgbImage::from_raw(w as u32, h as u32, raw).unwrap();
        // let dyn_img = image::DynamicImage::ImageRgb8(rgb_img);
        // let path_img = format!("./tmp/{:03}.jpg", f);
        // dyn_img.save(path_img)?;
    }
    debug!("iterate: {:?}", t.elapsed());
    Ok(())
}

/// Run video-detection multi-threaded
///
/// This version of `detect_video` runs decoding and detection in two separate threads.
///
/// # Example
/// ```
/// use video_inference::{DetectionConfig, detect_video_multithread};
/// let config = DetectionConfig {interval: Some(1.0), ..Default::default()};
/// let path_video = "./tests/assets/video.mp4";
/// let path_onnx = "./tests/assets/model.onnx";
/// let bboxes = detect_video_multithread(path_video, path_onnx, &config)?;
/// ```
pub fn detect_video_multithread(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    init_video_rs();
    // sync_channel used here for backpressure on the decoder loop
    let (tx, rx) = sync_channel::<DetectionTask>(10);
    let mut decoder = Decoder::new(path_video.as_ref().to_path_buf())?;
    let target_size = decoder.size();
    let n_frames = decoder.frames()?;

    let duration = decoder.duration()?.as_secs();
    let interval_frames = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;

    let path_onnx = path_onnx.as_ref().to_path_buf();
    let config_inner = config.clone();
    let handle = thread::spawn(move || detection_handler(rx, path_onnx, target_size, config_inner));

    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            if f % interval_frames == 0 {
                debug!("{}/{}", f, n_frames);
                let task = DetectionTask::new(frame);
                tx.send(task)
                    .map_err(|_| anyhow!("Failed: Detection thread dropped!"))?;
            }
        } else {
            break;
        }
    }
    debug!("decode video: {:?}", t.elapsed());
    // drop the last tx clone so rx knows when all senders are gone
    drop(tx);
    let bboxes = handle
        .join()
        .map_err(|e| anyhow!("Detection thread paniced: {e:?}"))??;
    debug!("detect video: {:?}", t.elapsed());
    Ok(bboxes)
}
