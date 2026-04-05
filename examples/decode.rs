use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::iterate_video;

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    let path_video = "./tests/assets/video.mp4";
    iterate_video(path_video)?;
    Ok(())
}
