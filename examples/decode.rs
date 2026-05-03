use std::time::Duration;

use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{decode_video_seeking, decode_video_sequential};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,video_inference=trace"))
        .init();
    let path_video = "./tests/assets/video.mp4";
    // iterate every frame
    decode_video_sequential(path_video, Some(Duration::from_millis(4700)))?;
    // vs. iterate by seeking
    decode_video_seeking(path_video, Some(Duration::from_millis(4700)))?;
    Ok(())
}
