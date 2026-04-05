use anyhow::{Error, anyhow};
use fast_image_resize::{self as fr, Resizer};
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time;
use tracing::debug;
use video_rs::decode::Decoder;

//use inference::detect_frame;

use crate::DetectionConfig;
use crate::detection::detect_frame;
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

pub(crate) struct RuntimeConfig {
    pub conf_thres: f32,
    pub iou_thres: f32,
    pub max_detect: usize,
    pub input_tensor_width: u32,
    pub input_tensor_height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub output_tensor_name: String,
}

pub(crate) struct Runtime {
    session: Session,
    decoder: Decoder,
    resizer: Resizer,

    interval_frames: usize,
    n_frames: usize,
    path_output: Option<PathBuf>,
    path_video: PathBuf,

    config: RuntimeConfig,
}

#[derive(Default)]
pub(crate) struct RuntimeBuilder {
    path_video: Option<PathBuf>,
    path_model: Option<PathBuf>,
    path_output: Option<PathBuf>,

    conf_thres: f32,
    iou_thres: f32,
    max_detect: usize,
    interval: Option<f32>,
    input_tensor_name: String,
    output_tensor_name: String,
}

impl From<&DetectionConfig> for RuntimeBuilder {
    fn from(value: &DetectionConfig) -> Self {
        Self {
            conf_thres: value.conf_thres,
            iou_thres: value.iou_thres,
            max_detect: value.max_detect,
            interval: value.interval,
            input_tensor_name: value.input_tensor_name.to_owned(),
            output_tensor_name: value.output_tensor_name.to_owned(),
            path_output: value.path_output.to_owned(),
            ..Default::default()
        }
    }
}

impl Runtime {
    pub fn capacity(&self) -> usize {
        if self.interval_frames > 1 {
            self.n_frames / self.interval_frames + 1
        } else {
            self.n_frames
        }
    }

    #[cfg(feature = "visualize")]
    pub fn path_ann(&self) -> PathBuf {
        match &self.path_output {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                let stem = self.path_video.file_stem().unwrap().to_str().unwrap();
                let ext = self.path_video.extension().unwrap().to_str().unwrap();
                self.path_video.with_file_name(format!("{stem}_ann.{ext}"))
            }
        }
    }

    #[cfg(feature = "visualize")]
    pub fn as_encoder(&self) -> Result<video_rs::Encoder, Error> {
        let settings = {
            use video_rs::encode::Settings;
            Settings::preset_h264_yuv420p(
                self.config.target_width as usize,
                self.config.target_height as usize,
                false,
            )
        };
        let encoder = video_rs::encode::Encoder::new(self.path_ann(), settings)?;
        Ok(encoder)
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
    pub fn detect_video(&mut self) -> Result<Vec<Vec<BoundingBox>>, Error> {
        let mut dst_image = fr::images::Image::new(
            self.config.input_tensor_width,
            self.config.input_tensor_height,
            fr::PixelType::U8x3,
        );
        let mut bboxes = Vec::with_capacity(self.capacity());
        let t = time::Instant::now();
        for (f, next_frame) in self.decoder.decode_iter().enumerate() {
            if let Ok((_ts, frame)) = next_frame {
                if f % self.interval_frames == 0 {
                    debug!("{}/{}", f, self.n_frames);
                    let bboxes_frame = detect_frame(
                        frame,
                        &self.config,
                        &mut self.session,
                        &mut self.resizer,
                        &mut dst_image,
                    )?;
                    bboxes.push(bboxes_frame);
                }
            } else {
                break;
            }
        }
        debug!("detect video: {:?}", t.elapsed());
        Ok(bboxes)
    }

    #[cfg(feature = "visualize")]
    pub fn annotate_video(&mut self) -> Result<Vec<Vec<BoundingBox>>, Error> {
        use crate::vizualize::draw_bboxes_arr;
        let mut dst_image = fr::images::Image::new(
            self.config.input_tensor_width,
            self.config.input_tensor_height,
            fr::PixelType::U8x3,
        );
        let mut bboxes = Vec::with_capacity(self.capacity());
        let mut encoder = self.as_encoder()?;
        let t = time::Instant::now();
        for (f, next_frame) in self.decoder.decode_iter().enumerate() {
            if let Ok((ts, frame)) = next_frame {
                if f % self.interval_frames == 0 {
                    debug!("{}/{}", f, self.n_frames);
                    let frame_copy = frame.clone();
                    let bboxes_frame = detect_frame(
                        frame,
                        &self.config,
                        &mut self.session,
                        &mut self.resizer,
                        &mut dst_image,
                    )?;
                    let annotated_arr3 = draw_bboxes_arr(frame_copy, &bboxes_frame)?;
                    encoder.encode(&annotated_arr3, ts)?;
                    bboxes.push(bboxes_frame);
                }
            } else {
                break;
            }
        }
        encoder.finish()?;
        debug!("detect video: {:?}", t.elapsed());
        Ok(bboxes)
    }
}

impl RuntimeBuilder {
    pub fn video(self, path: impl AsRef<Path>) -> Self {
        Self {
            path_video: Some(path.as_ref().to_path_buf()),
            ..self
        }
    }

    pub fn model(self, path: impl AsRef<Path>) -> Self {
        Self {
            path_model: Some(path.as_ref().to_path_buf()),
            ..self
        }
    }

    pub fn build(self) -> Result<Runtime, Error> {
        init_video_rs();
        let path_model = self.path_model.ok_or(anyhow!("Model path not set!"))?;
        let session = load_session(path_model)?;
        let (input_tensor_width, input_tensor_height) =
            detect_input_shape(&session, Some(self.input_tensor_name.as_str()))?;
        let path_video = self.path_video.ok_or(anyhow!("Video path not set!"))?;
        let decoder = Decoder::new(&*path_video)?;
        let (target_width, target_height) = decoder.size();
        let n_frames = decoder.frames()?;
        let interval_frames = match self.interval {
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
        let resizer = fr::Resizer::new();
        Ok(Runtime {
            session,
            decoder,
            resizer,
            interval_frames,
            n_frames: n_frames as usize,
            path_output: self.path_output,
            path_video,
            config: RuntimeConfig {
                conf_thres: self.conf_thres,
                iou_thres: self.iou_thres,
                max_detect: self.max_detect,
                input_tensor_height,
                input_tensor_width,
                target_height,
                target_width,
                output_tensor_name: self.output_tensor_name,
            },
        })
    }
}
