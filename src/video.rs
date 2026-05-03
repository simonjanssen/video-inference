use crate::Error;
use ndarray::Array3;
use std::iter::FusedIterator;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;
use tracing::{debug, error, warn};
#[cfg(feature = "annotate")]
use video_rs::Encoder;
use video_rs::{Decoder, DecoderBuilder, hwaccel::HardwareAccelerationDeviceType};

use crate::{Result, error::VideoInferenceError};

static INIT_VIDEO_RS: Once = Once::new();

pub(crate) fn init_video_rs() {
    INIT_VIDEO_RS.call_once(|| {
        video_rs::init().expect("Failed to initialize video-rs/ffmpeg backend!");
    });
}

pub(crate) fn calc_interval_frames(
    video_duration: Duration,
    video_frames: u64,
    interval_duration: Option<Duration>,
) -> u64 {
    match interval_duration {
        Some(interval) => {
            if interval > Duration::ZERO && video_frames > 0 {
                ((video_frames as f64 / video_duration.as_secs_f64()) * interval.as_secs_f64())
                    .round()
                    .max(1.0) as u64
            } else {
                1
            }
        }
        None => 1,
    }
}

/// Initialize video_rs::Decoder with fitting output frame size for ONNX-inference.
/// Also applies hardware acceleration if available.
pub(crate) fn get_decoder(path: impl AsRef<Path>, size: Option<(u32, u32)>) -> Result<Decoder> {
    init_video_rs();
    let devices = HardwareAccelerationDeviceType::list_available();
    debug!("available devices: {:?}", devices);
    let mut builder = DecoderBuilder::new(path.as_ref().to_path_buf());
    if let Some((w, h)) = size {
        builder = builder.with_resize(video_rs::Resize::Exact(w, h));
        debug!("resizing video to {} x {}", w, h);
    };

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        builder = builder.with_hardware_acceleration(HardwareAccelerationDeviceType::VideoToolbox);
        debug!("using `VideoToolbox` hardware acceleration for video decoding");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if devices.contains(&HardwareAccelerationDeviceType::Cuda) {
            builder = builder.with_hardware_acceleration(HardwareAccelerationDeviceType::Cuda);
            debug!("using `Cuda` hardware acceleration for video decoding");
        }
    }
    let decoder = builder.build().map_err(|e| VideoInferenceError::Video {
        detail: "Failed to initialize video decoder!".to_string(),
        source: e,
    })?;
    debug!("video has size {:?}", decoder.size());
    Ok(decoder)
}

/// Test available Hardware-Acceleration Device Types
///
/// Using an input video verify whether the claimed devices actually work.
pub fn test_available_devices(path: impl AsRef<Path>) {
    let devices = HardwareAccelerationDeviceType::list_available();
    if devices.is_empty() {
        warn!("no devices for video decoder found!");
        return;
    }
    debug!("available devices: {:?}", devices);
    for device in devices {
        match DecoderBuilder::new(path.as_ref().to_path_buf())
            .with_hardware_acceleration(device)
            .build()
        {
            Ok(decoder) => {
                let _ = decoder.size();
                debug!("video decoder sucessfully built with `{:?}` device", device);
            }
            Err(e) => {
                error!(
                    "failed to build video decoder with `{:?}` (error: {})",
                    device, e
                );
            }
        }
    }
}

// pub(crate) fn debug_decoder(decoder: &Decoder) -> Result<(), Error> {
//     let duration = decoder.duration()?;
//     let frame_rate = decoder.frame_rate();
//     let frames = decoder.frames()?;
//     let size = decoder.size();
//     let size_out = decoder.size_out();
//     let time_base = decoder.time_base();
//     debug!(
//         ?duration,
//         ?frame_rate,
//         frames,
//         ?size,
//         ?size_out,
//         ?time_base,
//         "Decoder properties"
//     );
//     Ok(())
// }

#[cfg(feature = "annotate")]
pub(crate) fn get_encoder(path: impl AsRef<Path>, size: (u32, u32)) -> Result<Encoder> {
    let (w, h) = size;
    let settings = {
        use video_rs::encode::Settings;
        Settings::preset_h264_yuv420p(w as usize, h as usize, false)
    };
    let encoder =
        video_rs::encode::Encoder::new(path.as_ref().to_path_buf(), settings).map_err(|e| {
            VideoInferenceError::Video {
                detail: "Failed to load encoder!".to_string(),
                source: e,
            }
        })?;
    Ok(encoder)
}

pub struct DecodedFrame {
    pub time: Duration,
    pub index: usize,
    pub array: Array3<u8>,
}

/// Strategy used by [`FrameIterator`] to advance through a video.
///
/// The choice is a trade-off between throughput and the precision with which
/// returned frames align to the requested sampling interval. Both strategies
/// yield items at roughly the same time positions, but reach them via very
/// different decoder operations.
///
/// # Variants
///
/// - [`Sequential`](Self::Sequential): decode every frame from the beginning
///   of the video and emit only every *n*-th one. No seeking is performed.
///   This is robust on any input (including streams without a usable seek
///   index) and produces frames at exact frame indices, but pays the full
///   decode cost for every frame in the video.
///
/// - [`Seeking`](Self::Seeking) *(default)*: for each sampling point, seek
///   the decoder to the corresponding timestamp and decode a single frame.
///   This is typically much faster for coarse intervals but has two
///   consequences caused by how video containers and codecs implement seek:
///
///   * The decoded frame is the next frame produced after a backward seek to
///     the nearest preceding keyframe; its timestamp may therefore be earlier
///     than the requested one. The reported `index` on [`DecodedFrame`] is
///     derived from the actual timestamp, not the requested position.
///   * If the requested interval is finer than the keyframe spacing of the
///     video, consecutive seeks can land on the same keyframe. The iterator
///     detects and skips such duplicates, which means the seeking strategy
///     can produce fewer frames than requested.
///
/// # Choosing a strategy
///
/// Pick `Seeking` for sparse sampling (e.g. one frame per second on a long
/// video) where decode cost dominates. Pick `Sequential` when you need every
/// sampled frame to land at an exact frame index, when the input may be a
/// non-seekable stream, or when the sampling interval is small enough that
/// sequential decoding is cheaper than one seek per item.
#[derive(Debug, Clone, Copy, Default)]
pub enum DecodingStrategy {
    /// Decode every frame and emit every *n*-th one.
    Sequential,
    /// Seek to each sampling point and decode a single frame.
    #[default]
    Seeking,
}

/// Iterator that yields decoded frames sampled at a fixed time interval.
///
/// `FrameIterator` decodes a video file and produces [`DecodedFrame`] items
/// at approximately fixed intervals, configured via
/// [`FrameIteratorBuilder::every`]. The way frames are reached is controlled
/// by [`DecodingStrategy`]; see its documentation for the trade-offs.
///
/// The iterator is constructed via [`FrameIterator::builder`].
///
/// # Item type
///
/// `Item = Result<DecodedFrame>`. EOF is signalled with `None`; transient
/// decoder errors are surfaced as `Some(Err(_))`. Once an error has been
/// returned, the iterator does not retry the failed step — `frame_idx` has
/// already been advanced — so iteration always makes forward progress and
/// terminates within a bounded number of steps. The iterator implements
/// [`FusedIterator`].
///
/// # Size hints
///
/// [`size_hint`](Iterator::size_hint) reports a tight upper bound based on
/// the remaining frames and configured interval. The lower bound is exact
/// for [`DecodingStrategy::Sequential`] and `0` for
/// [`DecodingStrategy::Seeking`], because seeks may produce fewer items than
/// requested when the video's keyframe spacing is coarser than the interval.
///
/// # Example
///
/// ```no_run
/// use std::time::Duration;
/// use video_inference::{FrameIterator, DecodingStrategy};
///
/// let frames = FrameIterator::builder("input.mp4")
///     .every(Duration::from_secs(1))
///     .seeking()
///     .build()?;
///
/// for frame in frames {
///     let frame = frame?;
///     println!("t={:?} idx={}", frame.time, frame.index);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FrameIterator {
    strategy: DecodingStrategy,
    decoder: Decoder,
    interval: usize,
    n_frames: usize,
    frame_idx: usize,
    last_ts: Option<Duration>,
    fps: f64,
}

impl FrameIterator {
    pub fn size(&self) -> (u32, u32) {
        self.decoder.size()
    }
}

impl Iterator for FrameIterator {
    type Item = Result<DecodedFrame>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_frames.saturating_sub(self.frame_idx);
        // Ceiling division: how many more items at most.
        let upper = remaining.div_ceil(self.interval);
        match self.strategy {
            // Sequential decode hits exactly every `interval`-th frame until EOF.
            DecodingStrategy::Sequential => (upper, Some(upper)),
            // Seeking can skip duplicates when seeks land on the same keyframe,
            // so the lower bound is 0 but the upper bound still holds.
            DecodingStrategy::Seeking => (0, Some(upper)),
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        match self.strategy {
            DecodingStrategy::Sequential => loop {
                if self.frame_idx >= self.n_frames {
                    // end of video reached
                    return None;
                }
                let frame_idx = self.frame_idx;
                self.frame_idx += 1;
                match self.decoder.decode() {
                    Ok((ts, frame)) => {
                        if frame_idx % self.interval == 0 {
                            return Some(Ok(DecodedFrame {
                                time: ts.into(),
                                index: frame_idx,
                                array: frame,
                            }));
                        } else {
                            continue;
                        }
                    }
                    Err(video_rs::Error::DecodeExhausted) => return None,
                    Err(e) => {
                        return Some(Err(VideoInferenceError::Video {
                            detail: "Failed to decode next video frame!".to_string(),
                            source: e,
                        }));
                    }
                }
            },
            DecodingStrategy::Seeking => {
                loop {
                    if (self.frame_idx + self.interval) > self.n_frames {
                        // end of video reached
                        return None;
                    }
                    let target_frame_idx = self.frame_idx;
                    self.frame_idx = target_frame_idx + self.interval;
                    let target_ms = (target_frame_idx as f64 / self.fps * 1000.0).round() as i64;
                    match self.decoder.seek(target_ms) {
                        Ok(_) => {}
                        Err(video_rs::Error::ReadExhausted) => return None,
                        Err(e) => {
                            return Some(Err(VideoInferenceError::Video {
                                detail: "Failed to seek next video position!".to_string(),
                                source: e,
                            }));
                        }
                    }
                    let (ts, frame) = match self.decoder.decode() {
                        Ok(result) => result,
                        Err(video_rs::Error::DecodeExhausted) => return None,
                        Err(e) => {
                            return Some(Err(VideoInferenceError::Video {
                                detail: "Failed to decode next video frame!".to_string(),
                                source: e,
                            }));
                        }
                    };

                    // skip if seek landed on the same keyframe as last iteration
                    let this_ts = ts.into();
                    if Some(this_ts) == self.last_ts {
                        warn!("Skipping previous keyframe - consider increasing interval");
                        continue;
                    }
                    self.last_ts = Some(this_ts);
                    let approx_frame_idx = (ts.as_secs_f64() * self.fps).round() as usize;
                    return Some(Ok(DecodedFrame {
                        time: this_ts,
                        index: approx_frame_idx,
                        array: frame,
                    }));
                }
            }
        }
    }
}

impl FusedIterator for FrameIterator {}

pub struct FrameIteratorBuilder {
    path: PathBuf,
    size: Option<(u32, u32)>,
    interval: Option<Duration>,
    strategy: Option<DecodingStrategy>,
}

impl FrameIteratorBuilder {
    pub fn resize(self, to: (u32, u32)) -> Self {
        Self {
            size: Some(to),
            ..self
        }
    }

    pub fn every(self, interval: Duration) -> Self {
        Self {
            interval: Some(interval),
            ..self
        }
    }

    pub fn sequential(self) -> Self {
        Self {
            strategy: Some(DecodingStrategy::Sequential),
            ..self
        }
    }

    pub fn seeking(self) -> Self {
        Self {
            strategy: Some(DecodingStrategy::Seeking),
            ..self
        }
    }

    pub fn strategy(self, strategy: DecodingStrategy) -> Self {
        Self {
            strategy: Some(strategy),
            ..self
        }
    }

    pub fn build(self) -> Result<FrameIterator> {
        let decoder = get_decoder(self.path, self.size)?;
        let duration = decoder.duration().map_err(|e| Error::Video {
            detail: "Failed to determine video duration!".to_string(),
            source: e,
        })?;
        let n_frames = decoder.frames().map_err(|e| Error::Video {
            detail: "Failed to determine number of frames!".to_string(),
            source: e,
        })?;
        let interval = calc_interval_frames(duration.into(), n_frames, self.interval) as usize;
        debug!(
            "interval_sec={:?} interval_frames={}",
            self.interval, interval
        );
        let fps = decoder.frame_rate() as f64;
        Ok(FrameIterator {
            decoder,
            interval,
            frame_idx: 0,
            fps,
            last_ts: None,
            n_frames: n_frames as usize,
            strategy: self.strategy.unwrap_or(DecodingStrategy::Seeking),
        })
    }
}

impl FrameIterator {
    pub fn builder(path: impl AsRef<Path>) -> FrameIteratorBuilder {
        FrameIteratorBuilder {
            path: path.as_ref().to_path_buf(),
            size: None,
            interval: None,
            strategy: None,
        }
    }
}
