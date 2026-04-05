use std::path::PathBuf;

use anyhow::Error;
use tracing::info;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, annotate_video};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detections in 1s intervals
    let config = DetectionConfig {
        interval: Some(1.0),
        path_output: Some(PathBuf::from("./tests/assets/video_with_bboxes.mp4")),
        ..Default::default()
    };
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";
    let bboxes = annotate_video(path_video, path_onnx, &config)?;
    info!("done: {}", bboxes.len());
    Ok(())
}
