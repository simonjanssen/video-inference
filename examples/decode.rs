use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{decode_video, decode_video_keyframes};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,video_inference=trace"))
        .init();
    let path_video = "./tests/assets/video.mp4";
    // iterate every frame
    decode_video(path_video, Some(4.7))?;
    // vs. iterate over keyframes - 47 is the optimal frame interval for our test videos
    decode_video_keyframes(path_video, Some(4.7))?;
    Ok(())
}
