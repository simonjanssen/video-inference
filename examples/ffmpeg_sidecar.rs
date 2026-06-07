// ! Benchmarking https://github.com/nathanbabcock/ffmpeg-sidecar

use std::time::{Duration, Instant};

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
        let _pixels: Vec<u8> = frame.data;
        count += 1
    }
    info!("{} frames (sidecar all) in {:?}", count, t0.elapsed());

    // use ffmpeg_sidecar with seeking every xth interval
    // spawn a new ffmpeg per seek position with `-ss <time>` placed BEFORE `-i`
    // for fast input-side seeking, decoding only a single frame each time
    let interval = Duration::from_millis(1000);
    let t0 = Instant::now();
    let mut count = 0;
    let mut position = Duration::ZERO;
    loop {
        let secs = position.as_secs_f64();
        let mut child = FfmpegCommand::new()
            .seek(format!("{secs}"))
            .input(path_video)
            .frames(1)
            .rawvideo()
            .spawn()?;
        let got_frame = child.iter()?.filter_frames().next().is_some();
        child.wait()?;
        if !got_frame {
            break;
        }
        count += 1;
        position += interval;
    }
    info!("{} frames (sidecar seeking) in {:?}", count, t0.elapsed());

    let t0 = Instant::now();
    let frames = FrameIterator::builder(path_video).sequential().build()?;
    let mut count = 0;
    for frame in frames {
        let _pixels = frame?.array;
        count += 1
    }
    info!("{} frames (video-rs all) in {:?}", count, t0.elapsed());

    let t0 = Instant::now();
    let frames = FrameIterator::builder(path_video)
        .seeking()
        .every(Duration::from_millis(1000))
        .build()?;
    let mut count = 0;
    for frame in frames {
        let _pixels = frame?.array;
        count += 1
    }
    info!("{} frames (video-rs seeking) in {:?}", count, t0.elapsed());

    Ok(())
}
