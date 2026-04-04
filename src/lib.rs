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

use crate::inference::load_session;

#[derive(Debug)]
pub struct Config {
    // confidence threshold for detections
    conf_thres: f32,
    // intersection-over-unition threshold for non-maxima supression
    iou_thres: f32,
    // maximum number of detections per image
    max_detect: usize,
    // onnx-model input width
    input_width: u32,
    // onnx-model input height
    input_height: u32,
    // image/video-frame width
    target_width: u32,
    // image/video-frame height
    target_height: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            conf_thres: 0.25,
            iou_thres: 0.4,
            max_detect: 300,
            input_width: 640,
            input_height: 640,
            target_width: 640,
            target_height: 640,
        }
    }
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
) -> Result<(), Error> {
    println!("hi");
    let path_video = path_video.as_ref();
    let video_name = path_video.file_name().unwrap().to_str().unwrap();
    //let t = time::Instant::now();
    let mut session = load_session(path_onnx)?;

    let mut decoder = Decoder::new(path_video.to_owned())?;
    let (target_width, target_height) = decoder.size();
    let n_frames = decoder.frames()?;

    let mut resizer = fr::Resizer::new();
    let config = Config {
        target_width,
        target_height,
        ..Default::default()
    };
    debug!("{:?}", config);

    let mut dst_image =
        fr::images::Image::new(config.input_width, config.input_height, fr::PixelType::U8x3);

    let t = time::Instant::now();
    for (f, next_frame) in decoder.decode_iter().enumerate() {
        if let Ok((_ts, frame)) = next_frame {
            debug!("{}/{}", f, n_frames);
            let _bboxes = detect_frame(&mut session, frame, &config, &mut resizer, &mut dst_image)?;
        } else {
            break;
        }
    }
    debug!("{}: {:?}", video_name, t.elapsed());
    Ok(())
}
