use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{iterate_video, iterate_video_keyframes};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    let path_video = "./tests/assets/video.mp4";
    // iterate every frame
    iterate_video(path_video)?;
    // vs. iterate over keyframes
    iterate_video_keyframes(path_video, Some(50))?;
    Ok(())
}
