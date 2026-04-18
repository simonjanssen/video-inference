use std::path::Path;
use std::sync::Once;
use tracing::{debug, error, warn};
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
pub(crate) fn get_decoder(path: impl AsRef<Path>, size: (u32, u32)) -> Result<Decoder> {
    let (w, h) = size;
    let devices = HardwareAccelerationDeviceType::list_available();
    debug!("available devices: {:?}", devices);
    let mut builder =
        DecoderBuilder::new(path.as_ref().to_path_buf()).with_resize(video_rs::Resize::Exact(w, h));
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        use video_rs::hwaccel::HardwareAccelerationDeviceType;
        builder = builder.with_hardware_acceleration(HardwareAccelerationDeviceType::VideoToolbox)
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
