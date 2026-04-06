# video-inference

> My grandma likes to watch birds. So we've built a multi-camera system around her cottage in Sweden to record 60s-snippets of the surroundings. The question is: what's the fastest way to run object detection on these 60s-mp4 video snippets? This is what this repository intends to find out.

## Setup
This crate relies on [`video-rs`](https://lib.rs/crates/video-rs) for mp4-video decoding which in turn requires [`rust-ffmpeg` dependencies](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies) to be installed at build time.

Add this crate as a dependency to your `Cargo.toml`:
```toml
video-inference = { git = "https://github.com/simonjanssen/video-inference" }
```

Note: matching targets are auto-discovered at build time, such that the right features for `ort` (inference) and `rust-ffmpeg` (decoding) are enabled. Currently tested on `macos/aarch64` (Apple Silicon) and `linux/aarch64` (Raspberry Pi).

## Quickstart
```rust
use anyhow::Error;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, detect_video};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detections in 1s intervals
    let config = DetectionConfig {
        interval: Some(1.0),
        ..Default::default()
    };
    let path_video = "./tests/assets/video.mp4";
    let path_onnx = "./tests/assets/model.onnx";
    let bboxes = detect_video(path_video, path_onnx, &config)?;
    Ok(())
}
```

See the [examples](./examples/) directory for more usage patterns.