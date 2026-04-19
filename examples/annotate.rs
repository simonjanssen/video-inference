use std::time;

use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, annotate_video, detect_video};

fn main() -> Result<(), Error> {
    let t = time::Instant::now();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detections in 4.7s intervals (optimum for our test videos)
    let config = DetectionConfig {
        interval: Some(4.7),
        ..Default::default()
    };
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";
    let path_output = "./video_ann.mp4";
    let detections = detect_video(path_video, path_onnx, &config)?;
    let detections = annotate_video(path_video, &detections, &config, path_output)?;
    let file = std::fs::File::create("./tests/assets/video_ann.json")?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &detections)?;
    println!("total time spent: {:?}", t.elapsed());
    Ok(())
}
