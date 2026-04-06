use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, detect_video_multithread};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detections in 1s intervals
    let config = DetectionConfig {
        interval: Some(1.0),
        ..Default::default()
    };
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";
    let bboxes = detect_video_multithread(path_video, path_onnx, &config)?;
    let file = std::fs::File::create("./tests/assets/video_ann.json")?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &bboxes)?;
    Ok(())
}
