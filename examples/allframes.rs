use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, detect_video_multi_thread, load_model};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detection on all video frames
    let config = DetectionConfig::default();
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";
    let mut model = load_model(path_onnx, &config)?;
    let detections = detect_video_multi_thread(path_video, &mut model, &config)?;
    let file = std::fs::File::create("./tests/assets/video_all_ann.json")?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &detections)?;
    Ok(())
}
