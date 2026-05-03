use std::time::{self, Duration};

use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, detect_video_with_model, load_model};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detections in 4.7s intervals (optimum for our test videos)
    let config = DetectionConfig {
        interval: Some(Duration::from_millis(4700)),
        ..Default::default()
    };
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";

    let t = time::Instant::now();
    // loading the model once
    let mut model = load_model(path_onnx, &config)?;
    let _detections = detect_video_with_model(path_video, &mut model, &config)?;
    println!("total time spent (1): {:?}", t.elapsed());

    let t = time::Instant::now();
    // re-using the loaded model for sequentially appearing, new videos
    let detections = detect_video_with_model(path_video, &mut model, &config)?;
    println!("total time spent (2): {:?}", t.elapsed());

    let file = std::fs::File::create("./tests/assets/video_ann.json")?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &detections)?;
    Ok(())
}
