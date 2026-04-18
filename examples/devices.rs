use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::test_available_devices;

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();

    // we need a test video
    let path_video = "./tests/assets/video.mp4";

    // test available hardware acceleration options
    test_available_devices(path_video);
    Ok(())
}
