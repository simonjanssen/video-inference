use anyhow::Error;
use video_inference::{detect_video, DetectionConfig};
use tracing_subscriber::EnvFilter;
use tracing::info;

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    let config = DetectionConfig::default();
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";
    let bboxes = detect_video(path_video, path_onnx, &config)?;
    info!("done: {}", bboxes.len());
    Ok(())
}