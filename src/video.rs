use crate::Error;
use ndarray::{ArrayBase, Dim, OwnedRepr};
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

pub(crate) fn calc_interval_frames(duration: f32, frames: u32, interval: Option<f32>) -> u32 {
    match interval {
        Some(interval) => {
            if duration > 0.0 && frames > 0 {
                ((frames as f32 / duration) * interval).round().max(1.0) as u32
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

pub struct KeyFrameIterator {
    decoder: Decoder,
    interval: usize,
    n_frames: usize,
    frame_idx: usize,
    last_ts: Duration,
    fps: f32,
}

impl KeyFrameIterator {
    pub fn size(&self) -> (u32, u32) {
        self.decoder.size()
    }
}

impl Iterator for KeyFrameIterator {
    type Item = (
        Duration,
        usize,
        ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>, u8>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame_idx = self.frame_idx;
            let target_ms = (self.frame_idx as f64 / self.fps as f64 * 1000.0) as i64;
            match self.decoder.seek(target_ms) {
                Ok(_) => {}
                Err(_e) => {
                    // end of video reached
                    return None;
                }
            };
            let (ts, frame) = match self.decoder.decode() {
                Ok(result) => result,
                Err(e) => {
                    error!("{}", e);
                    return None;
                }
            };

            // skip if seek landed on the same keyframe as last iteration
            let this_ts = ts.into();
            self.frame_idx = frame_idx + self.interval;
            if self.frame_idx > self.n_frames {
                // end of video reached
                return None;
            }
            if this_ts == self.last_ts && this_ts != Duration::ZERO {
                warn!("Skipping previous keyframe - consider increasing interval");
                continue;
            } else {
                self.last_ts = this_ts;
                return Some((this_ts, frame_idx, frame));
            }
        }
    }
}

#[derive(Default)]
pub struct KeyFrameIteratorBuilder {
    path: PathBuf,
    size: Option<(u32, u32)>,
    interval: Option<f32>,
}

impl KeyFrameIteratorBuilder {
    pub fn resize(self, to: (u32, u32)) -> Self {
        Self {
            size: Some(to),
            ..self
        }
    }

    pub fn interval(self, every: f32) -> Self {
        Self {
            interval: Some(every),
            ..self
        }
    }

    pub fn build(self) -> Result<KeyFrameIterator> {
        let decoder = get_decoder(self.path, self.size)?;
        let duration = decoder
            .duration()
            .map_err(|e| Error::Video {
                detail: "Failed to determine video duration!".to_string(),
                source: e,
            })?
            .as_secs();
        let n_frames = decoder.frames().map_err(|e| Error::Video {
            detail: "Failed to determine number of frames!".to_string(),
            source: e,
        })? as i64;
        let interval = calc_interval_frames(duration, n_frames as u32, self.interval) as usize;
        let fps = decoder.frame_rate();
        Ok(KeyFrameIterator {
            decoder,
            interval,
            frame_idx: 0,
            fps,
            last_ts: Duration::ZERO,
            n_frames: n_frames as usize,
        })
    }
}

impl KeyFrameIterator {
    pub fn builder(path: impl AsRef<Path>) -> KeyFrameIteratorBuilder {
        KeyFrameIteratorBuilder {
            path: path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }
}

pub struct FrameIterator {
    decoder: Decoder,
    interval: usize,
    frame_idx: usize,
}

impl FrameIterator {
    pub fn size(&self) -> (u32, u32) {
        self.decoder.size()
    }
}

impl From<KeyFrameIterator> for FrameIterator {
    fn from(value: KeyFrameIterator) -> Self {
        Self {
            decoder: value.decoder,
            interval: value.interval,
            frame_idx: 0,
        }
    }
}

impl Iterator for FrameIterator {
    type Item = (
        Duration,
        usize,
        ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>, u8>,
    );
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame_idx = self.frame_idx;
            match self.decoder.decode() {
                Ok((frame_ts, frame)) => {
                    self.frame_idx += 1;
                    if frame_idx % self.interval == 0 {
                        return Some((frame_ts.into(), frame_idx, frame));
                    } else {
                        continue;
                    }
                }
                Err(_e) => return None,
            }
        }
    }
}
