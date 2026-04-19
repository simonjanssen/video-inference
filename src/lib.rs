use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::{thread, time};
use ort::session::Session;
use tracing::debug;

pub mod detection;
mod error;
mod onnx;
mod threading;
mod video;
#[cfg(feature = "visualize")]
mod vizualize;

//use inference::detect_frame;

use crate::detection::{Detection, detect_image};
use crate::error::VideoInferenceError;
use crate::onnx::{detect_input_shape, load_session};
use crate::threading::{DetectionTask, detection_handler};
use crate::video::{calc_interval_frames, get_decoder, init_video_rs};

pub use video::test_available_devices;

pub use error::VideoInferenceError as Error;
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
    detect_video_multi_thread_keyframes(path_video, &mut model, config)
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
    detect_video_multi_thread_keyframes(path_video, model, config)
}

pub fn detect_video_single_thread(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    init_video_rs();

    let mut session = load_session(path_onnx)?;
    let size_onnx = detect_input_shape(&session, Some("images"))?;
    let mut decoder = get_decoder(path_video, size_onnx)?;
    let size_video = decoder.size();

    let n_frames = decoder.frames().map_err(|e| Error::Video {
        detail: "Failed to determine number of frames!".to_string(),
        source: e,
    })?;
    let duration = decoder
        .duration()
        .map_err(|e| Error::Video {
            detail: "Failed to determine video duration!".to_string(),
            source: e,
        })?
        .as_secs();
    let interval_frames = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;

    let mut detections = Vec::new();
    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            if f % interval_frames == 0 {
                debug!("{}/{}", f, n_frames);
                let detection =
                    detect_image(&mut session, frame, config, size_video, size_onnx, f as u32)?;
                detections.push(detection);
            }
        } else {
            break;
        }
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
/// let detections = detect_video_multithread(path_video, path_onnx, &config)?;
/// ```
pub fn detect_video_multi_thread(
    path_video: impl AsRef<Path>,
    model: &mut Model,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    init_video_rs();

    let mut decoder = get_decoder(path_video, model.size)?;
    let size_video = decoder.size();

    let n_frames = decoder.frames().map_err(|e| Error::Video {
        detail: "Failed to determine number of frames!".to_string(),
        source: e,
    })?;
    let duration = decoder
        .duration()
        .map_err(|e| Error::Video {
            detail: "Failed to video duration!".to_string(),
            source: e,
        })?
        .as_secs();
    let interval_frames = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;

    // sync_channel used here for backpressure on the decoder loop
    let (tx, rx) = sync_channel::<DetectionTask>(15);
    let config_inner = config.clone();

    // thread::scope (not thread::spawn) so we can borrow `&mut Model` into
    // the spawned thread. Scoped threads are guaranteed to join before the
    // scope exits, so the borrow checker accepts non-'static references.
    // Rc/Arc wouldn't help: Rc is !Send, and Arc<Mutex<Session>> would just
    // serialize inference behind a lock — no better than a plain &mut.
    let detections = thread::scope(|s| {
        let handle = s.spawn(|| detection_handler(rx, model, config_inner, size_video));

        let t = time::Instant::now();
        for (f, next_frame) in decoder.decode_iter().enumerate() {
            if let Ok((_ts, frame)) = next_frame {
                if f % interval_frames == 0 {
                    debug!("{}/{}", f, n_frames);
                    let task = DetectionTask::new(frame, f as u32);
                    tx.send(task).map_err(|_| {
                        Error::Thread("Failed to dispatch to detection thread!".to_string())
                    })?;
                }
            } else {
                break;
            }
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

pub fn detect_video_multi_thread_keyframes(
    path_video: impl AsRef<Path>,
    model: &mut Model,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    init_video_rs();

    //let session = load_session(path_onnx)?;
    //let size_onnx = detect_input_shape(&session, Some("images"))?;
    let mut decoder = get_decoder(path_video, model.size)?;
    let size_video = decoder.size();

    let n_frames = decoder.frames().map_err(|e| Error::Video {
        detail: "Failed to determine number of frames!".to_string(),
        source: e,
    })? as i64;

    let duration = decoder
        .duration()
        .map_err(|e| Error::Video {
            detail: "Failed to video duration!".to_string(),
            source: e,
        })?
        .as_secs();
    let interval = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;
    debug!("interval frames: {}", interval);

    let fps = decoder.frame_rate();

    // sync_channel used here for backpressure on the decoder loop
    let (tx, rx) = sync_channel::<DetectionTask>(15);
    let config_inner = config.clone();

    // thread::scope (not thread::spawn) so we can borrow `&mut Model` into
    // the spawned thread. Scoped threads are guaranteed to join before the
    // scope exits, so the borrow checker accepts non-'static references.
    // Rc/Arc wouldn't help: Rc is !Send, and Arc<Mutex<Session>> would just
    // serialize inference behind a lock — no better than a plain &mut.
    let detections = thread::scope(|s| {
        let handle = s.spawn(|| detection_handler(rx, model, config_inner, size_video));

        let t = time::Instant::now();
        let mut last_ts: f32 = -1.0;
        for f in (0i64..n_frames).step_by(interval) {
            let target_ms = (f as f64 / fps as f64 * 1000.0) as i64;
            let Ok(()) = decoder.seek(target_ms) else {
                break;
            };
            let Ok((ts, frame)) = decoder.decode() else {
                break;
            };
            // skip if seek landed on the same keyframe as last iteration
            if ts.as_secs() == last_ts {
                debug!("skipping previous keyframe");
                continue;
            }
            last_ts = ts.as_secs();
            debug!("{} / {}", f, ts.as_secs());
            let task = DetectionTask::new(frame, f as u32);
            tx.send(task)
                .map_err(|_| Error::Thread("Failed to dispatch to detection thread!".to_string()))?;
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
pub fn decode_video(path_video: impl AsRef<Path>) -> Result<()> {
    init_video_rs();
    let mut decoder = get_decoder(path_video, (640, 640))?;
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
pub fn decode_video_keyframes(path_video: impl AsRef<Path>, interval: Option<usize>) -> Result<()> {
    init_video_rs();
    let mut decoder = get_decoder(path_video, (640, 640))?;
    let t = time::Instant::now();
    let n_frames = decoder.frames().map_err(|e| Error::Video {
        detail: "Failed to determine number of frames!".to_string(),
        source: e,
    })? as i64;
    let fps = decoder.frame_rate();
    let interval = interval.unwrap_or(50);
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
            debug!("skipping previous keyframe");
            continue;
        }
        last_ts = ts.as_secs();
        debug!("{} / {}", f, ts.as_secs());
    }
    debug!("decode video: {:?}", t.elapsed());
    Ok(())
}
