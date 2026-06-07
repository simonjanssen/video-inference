// ! Testing https://github.com/nathanbabcock/ffmpeg-sidecar

use std::time::Instant;

use anyhow::Error;
use ffmpeg_sidecar::command::FfmpegCommand;
use tracing::info;
use tracing_subscriber::EnvFilter;

use video_inference::FrameIterator;

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,video_inference=trace"))
        .init();
    let path_video = "tests/assets/1776618850_905.mp4";

    let t0 = Instant::now();
    let sidecar = FfmpegCommand::new()
        //.hwaccel("auto")
        .input(path_video)
        .rawvideo()
        .spawn()?
        .iter()?;

    let mut count = 0;
    for frame in sidecar.filter_frames() {
        let _pixels: Vec<u8> = frame.data; // <- raw RGB pixels! 🎨
        count += 1
    }
    info!("{} frames in {:?}", count, t0.elapsed());

    let t0 = Instant::now();
    let frames = FrameIterator::builder(path_video).sequential().build()?;
    let mut count = 0;
    for frame in frames {
        let _pixels = frame?.array;
        count += 1
    }
    info!("{} frames in {:?}", count, t0.elapsed());

    Ok(())
}
