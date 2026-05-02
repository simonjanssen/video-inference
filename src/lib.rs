use ort::session::Session;
use std::path::Path;
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};
use std::{thread, time};
use tracing::debug;

#[cfg(feature = "annotate")]
mod annotate;
pub mod detection;
mod error;
mod onnx;
mod threading;
mod video;

use crate::detection::{Detection, detect_image};
use crate::error::VideoInferenceError;
use crate::onnx::{detect_input_shape, load_session};
use crate::threading::detection_handler;

// public exports
pub use error::VideoInferenceError as Error;
pub use video::{DecodedFrame, FrameIterator, DecodingStrategy, test_available_devices};
pub type Result<T> = std::result::Result<T, VideoInferenceError>;

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
    pub interval: Option<Duration>,

    /// ONNX-model image input tensor name
    pub input_tensor_name: String,

    /// ONNX-model results output tensor name
    pub output_tensor_name: String,

    /// Video decoding strategy
    pub strategy: DecodingStrategy,
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
            strategy: DecodingStrategy::default(),
        }
    }
}

pub struct Model {
    session: Session,
    size: (u32, u32),
}

pub fn load_model(path_onnx: impl AsRef<Path>, config: &DetectionConfig) -> Result<Model> {
    let session = load_session(path_onnx)?;
    let size = detect_input_shape(&session, Some(config.input_tensor_name.as_ref()))?;
    Ok(Model { session, size })
}

/// Run video-detection on mp4-video. Defaults to the optimal approach.
///
/// # Example
/// ```
/// use video_inference::{DetectionConfig, detect_video};
/// let config = DetectionConfig {interval: Some(4.7), ..Default::default()};
/// let path_video = "./tests/assets/video.mp4";
/// let path_onnx = "./tests/assets/model.onnx";
/// let detections = detect_video(path_video, path_onnx, &config)?;
/// ```
pub fn detect_video(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    let mut model = load_model(path_onnx, config)?;
    detect_video_multi_thread(path_video, &mut model, config)
}

/// Run video-detection on mp4-video. Defaults to the optimal approach.
///
/// This variant allows the caller to externally initialize the model.
/// This might be beneficial if running on multiple videos with the same model.
/// ```
/// use video_inference::{DetectionConfig, load_model, detect_video};
/// let config = DetectionConfig {interval: Some(4.7), ..Default::default()};
/// let path_video = "./tests/assets/video.mp4";
/// let path_onnx = "./tests/assets/model.onnx";
/// let mut model = load_model(path_onnx, config)?;
/// let detections = detect_video_with_model(path_video, &mut model, &config)?;
/// ```
pub fn detect_video_with_model(
    path_video: impl AsRef<Path>,
    model: &mut Model,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    detect_video_multi_thread(path_video, model, config)
}

pub fn detect_video_single_thread(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    let mut model = load_model(path_onnx, config)?;
    let frames = FrameIterator::builder(path_video)
        .resize(model.size)
        .every(config.interval.unwrap_or_default())
        .strategy(config.strategy)
        .build()?;
    let size_video = frames.size();

    let mut detections = Vec::new();
    let t = time::Instant::now();
    for frame in frames {
        let frame = frame?;
        let detection = detect_image(
            &mut model.session,
            frame.array,
            config,
            size_video,
            model.size,
            frame.index as u32,
        )?;
        detections.push(detection);
    }
    let dt = t.elapsed().as_secs() as f32;
    debug!(
        "decode+detect video: {:?} ({} frames, {} frames/sec)",
        dt,
        detections.len(),
        (detections.len() as f32 / dt)
    );
    Ok(detections)
}

pub fn detect_video_multi_thread(
    path_video: impl AsRef<Path>,
    model: &mut Model,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    let frames = FrameIterator::builder(path_video)
        .resize(model.size)
        .every(config.interval.unwrap_or_default())
        .strategy(config.strategy)
        .build()?;
    let size_video = frames.size();

    // sync_channel used here for backpressure on the decoder loop
    let (tx, rx) = sync_channel::<DecodedFrame>(15);
    let config_inner = config.clone();

    // thread::scope (not thread::spawn) so we can borrow `&mut Model` into
    // the spawned thread. Scoped threads are guaranteed to join before the
    // scope exits, so the borrow checker accepts non-'static references.
    // Rc/Arc wouldn't help: Rc is !Send, and Arc<Mutex<Session>> would just
    // serialize inference behind a lock — no better than a plain &mut.
    let detections = thread::scope(|s| {
        let handle = s.spawn(|| detection_handler(rx, model, config_inner, size_video));
        let t = time::Instant::now();
        for frame in frames {
            let frame = frame?;
            debug!("ts={:.3?} idx={:}", frame.time, frame.index);
            tx.send(frame).map_err(|_| {
                Error::Thread("Failed to dispatch task to detection thread!".to_string())
            })?;
        }
        debug!("decode video: {:?}", t.elapsed());
        // drop the last tx clone so rx knows when all senders are gone
        drop(tx);
        let detections = handle.join().map_err(|_| {
            Error::Thread("Failed to retrieve results from detection thread!".to_string())
        })??;
        let dt = t.elapsed().as_secs_f32();
        debug!(
            "detect video: {:?} ({} frames, {} frames/sec)",
            dt,
            detections.len(),
            (detections.len() as f32 / dt)
        );
        Ok(detections)
    })?;
    Ok(detections)
}

/// Run video-decoding frame by frame
///
/// This is for testing purposes only to measures the decoder-runtime on different targets
pub fn decode_video_sequential(
    path_video: impl AsRef<Path>,
    interval: Option<Duration>,
) -> Result<()> {
    let t = Instant::now();
    let frames = FrameIterator::builder(path_video)
        .every(interval.unwrap_or_default())
        .sequential()
        .build()?;
    for frame in frames {
        let frame = frame?;
        debug!(
            "ts={:.3?} idx={:} shape={:?}",
            frame.time,
            frame.index,
            frame.array.shape()
        );
    }
    debug!("decode video: {:?}", t.elapsed());
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
pub fn decode_video_seeking(
    path_video: impl AsRef<Path>,
    interval: Option<Duration>,
) -> Result<()> {
    let t = Instant::now();
    let frames = FrameIterator::builder(path_video)
        .every(interval.unwrap_or_default())
        .seeking()
        .build()?;
    for frame in frames {
        let frame = frame?;
        debug!(
            "ts={:.3?} idx={:} shape={:?}",
            frame.time,
            frame.index,
            frame.array.shape()
        );
    }
    debug!("decode video: {:?}", t.elapsed());
    Ok(())
}

#[cfg(feature = "annotate")]
pub fn annotate_video(
    path_video: impl AsRef<Path>,
    detections: &[Detection],
    config: &DetectionConfig,
    path_output: impl AsRef<Path>,
) -> Result<()> {
    use crate::annotate::draw_bboxes_arr;
    use crate::video::get_encoder;
    use std::collections::HashMap;

    let frames = FrameIterator::builder(path_video)
        .every(config.interval.unwrap_or_default())
        .strategy(config.strategy)
        .build()?;
    let size_video = frames.size();

    let mut detections_map = HashMap::with_capacity(detections.len());
    for detection in detections {
        detections_map.insert(detection.frame_idx as usize, &detection.bboxes);
    }

    let mut encoder = get_encoder(path_output, size_video)?;

    let t = time::Instant::now();
    for frame in frames {
        let frame = frame?;
        // draw bboxes
        let bboxes = detections_map
            .get(&frame.index)
            .ok_or(VideoInferenceError::Io("Missing detection".to_string()))?;
        let annotated = draw_bboxes_arr(frame.array.clone(), bboxes)?;
        encoder
            .encode(&annotated, frame.time.into())
            .map_err(|e| VideoInferenceError::Video {
                detail: "Failed to encode annotated frame!".to_string(),
                source: e,
            })?;
    }
    encoder.finish().map_err(|e| VideoInferenceError::Video {
        detail: "Failed to finalize annotated video!".to_string(),
        source: e,
    })?;
    debug!("annotate video: {:?}", t.elapsed());

    Ok(())
}
