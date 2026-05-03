use std::time::{Duration, Instant};

use anyhow::Error;
use tracing::info;
use tracing_subscriber::EnvFilter;
use video_inference::FrameIterator;

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,video_inference=trace"))
        .init();
    let path_video = "./tests/assets/video.mp4";
    let frames = FrameIterator::builder(path_video)
        .resize((640, 640))
        .every(Duration::from_millis(4700))
        .sequential()
        .build()?;

    let t = Instant::now();
    for frame in frames {
        let frame = frame?;
        info!(
            "ts={:.3?} idx={:} shape={:?}",
            frame.time,
            frame.index,
            frame.array.shape()
        );
    }
    println!("iterate video (sequential): {:?}", t.elapsed());

    let frames = FrameIterator::builder(path_video)
        .resize((640, 640))
        .every(Duration::from_millis(4700))
        .seeking()
        .build()?;

    let t = Instant::now();
    for frame in frames {
        let frame = frame?;
        info!(
            "ts={:.3?} idx={:} shape={:?}",
            frame.time,
            frame.index,
            frame.array.shape()
        );
    }
    println!("iterate video (seeking): {:?}", t.elapsed());

    Ok(())
}
