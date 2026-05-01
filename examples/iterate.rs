use std::time::Instant;

use anyhow::Error;
use tracing::info;
use tracing_subscriber::EnvFilter;
use video_inference::{FrameIterator, KeyFrameIterator};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,video_inference=trace"))
        .init();
    let path_video = "./tests/assets/video.mp4";
    let frames = KeyFrameIterator::builder(path_video)
        .resize((640, 640))
        .interval(4.7)
        .build()?;

    let t = Instant::now();
    for (frame_ts, frame_idx, frame) in frames {
        info!(
            "ts={:.3?} idx={:} shape={:?}",
            frame_ts,
            frame_idx,
            frame.shape()
        );
    }
    println!("decode video: {:?}", t.elapsed());

    let frames: FrameIterator = KeyFrameIterator::builder(path_video)
        .resize((640, 640))
        .interval(4.7)
        .build()?
        .into();

    let t = Instant::now();
    for (frame_ts, frame_idx, frame) in frames {
        info!(
            "ts={:.3?} idx={:} shape={:?}",
            frame_ts,
            frame_idx,
            frame.shape()
        );
    }
    println!("decode video: {:?}", t.elapsed());

    Ok(())
}
