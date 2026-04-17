use anyhow::{Error, anyhow};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::{thread, time};
use tracing::debug;

pub mod detection;
mod onnx;
mod threading;
mod video;
#[cfg(feature = "visualize")]
mod vizualize;

//use inference::detect_frame;

use crate::detection::{BoundingBox, detect_image};
use crate::onnx::{detect_input_shape, load_session};
use crate::threading::{DetectionTask, detection_handler};
use crate::video::{calc_interval_frames, get_decoder, init_video_rs};

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

/// Best Approach
pub fn detect_video(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    detect_video_multi_thread_keyframes(path_video, path_onnx, config)
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
pub fn detect_video_single_thread(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    init_video_rs();

    let mut session = load_session(path_onnx)?;
    let size_onnx = detect_input_shape(&session, Some("images"))?;
    let mut decoder = get_decoder(path_video, size_onnx)?;
    let size_video = decoder.size();

    let n_frames = decoder.frames()?;
    let duration = decoder.duration()?.as_secs();
    let interval_frames = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;

    let mut bboxes = Vec::new();
    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            if f % interval_frames == 0 {
                debug!("{}/{}", f, n_frames);
                let bboxes_frame =
                    detect_image(&mut session, frame, config, size_video, size_onnx)?;
                bboxes.push(bboxes_frame);
            }
        } else {
            break;
        }
    }
    let dt = t.elapsed().as_secs() as f32;
    debug!(
        "decode+detect video: {:?} ({} frames, {} frames/sec)",
        dt,
        bboxes.len(),
        (bboxes.len() as f32 / dt)
    );
    Ok(bboxes)
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
pub fn detect_video_multi_thread(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    init_video_rs();

    let session = load_session(path_onnx)?;
    let size_onnx = detect_input_shape(&session, Some("images"))?;
    let mut decoder = get_decoder(path_video, size_onnx)?;
    let size_video = decoder.size();

    let n_frames = decoder.frames()?;
    let duration = decoder.duration()?.as_secs();
    let interval_frames = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;

    // sync_channel used here for backpressure on the decoder loop
    let (tx, rx) = sync_channel::<DetectionTask>(15);
    let config_inner = config.clone();
    let handle =
        thread::spawn(move || detection_handler(rx, session, config_inner, size_video, size_onnx));

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
    let dt = t.elapsed().as_secs_f32();
    debug!(
        "detect video: {:?} ({} frames, {} frames/sec)",
        dt,
        bboxes.len(),
        (bboxes.len() as f32 / dt)
    );
    Ok(bboxes)
}

pub fn detect_video_multi_thread_keyframes(
    path_video: impl AsRef<Path>,
    path_onnx: impl AsRef<Path>,
    config: &DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    init_video_rs();

    let session = load_session(path_onnx)?;
    let size_onnx = detect_input_shape(&session, Some("images"))?;
    let mut decoder = get_decoder(path_video, size_onnx)?;
    let size_video = decoder.size();

    let n_frames = decoder.frames()? as i64;

    let duration = decoder.duration()?.as_secs();
    let interval = calc_interval_frames(duration, n_frames as u32, config.interval) as usize;
    debug!("interval frames: {}", interval);

    let fps = decoder.frame_rate();

    // sync_channel used here for backpressure on the decoder loop
    let (tx, rx) = sync_channel::<DetectionTask>(15);
    let config_inner = config.clone();
    let handle =
        thread::spawn(move || detection_handler(rx, session, config_inner, size_video, size_onnx));

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
        let task = DetectionTask::new(frame);
        tx.send(task)
            .map_err(|_| anyhow!("Failed: Detection thread dropped!"))?;
    }
    debug!("decode video: {:?}", t.elapsed());
    // drop the last tx clone so rx knows when all senders are gone
    drop(tx);
    let bboxes = handle
        .join()
        .map_err(|e| anyhow!("Detection thread paniced: {e:?}"))??;
    let dt = t.elapsed().as_secs_f32();
    debug!(
        "detect video: {:?} ({} frames, {} frames/sec)",
        dt,
        bboxes.len(),
        (bboxes.len() as f32 / dt)
    );
    Ok(bboxes)
}

/// Run video-decoding frame by frame
///
/// This is for testing purposes only to measures the decoder-runtime on different targets
pub fn decode_video(path_video: impl AsRef<Path>) -> Result<(), Error> {
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
pub fn decode_video_keyframes(
    path_video: impl AsRef<Path>,
    interval: Option<usize>,
) -> Result<(), Error> {
    init_video_rs();
    let mut decoder = get_decoder(path_video, (640, 640))?;
    let t = time::Instant::now();
    let n_frames = decoder.frames()? as i64;
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
