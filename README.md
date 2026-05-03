# video-inference

> My grandma likes to watch birds. So we've built a multi-camera system around her cottage in Sweden to record 60s-snippets of the surroundings. The question is: what's the fastest way to run object detection on these 60s-mp4 video snippets? This is what this repository intends to find out.

## Setup
This crate relies on [`video-rs`](https://lib.rs/crates/video-rs) for mp4-video decoding which in turn requires [`rust-ffmpeg` dependencies](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies) to be installed at build time.

Specifically on Raspberry Pi, the following packages are required (or just follow the compiler hints when trying for the first time):
```bash
apt install libssl-dev \
    && libavutil-dev libavformat-dev libavfilter-dev libavdevice-dev \
    && libclang-dev
```

Add this crate as a dependency to your `Cargo.toml`:
```toml
video-inference = { git = "https://github.com/simonjanssen/video-inference" }
```

Note: matching targets are auto-discovered at build time, such that the right features for `ort` (inference) and `rust-ffmpeg` (decoding) are enabled. Currently tested on `macos/aarch64` (Apple Silicon) and `linux/aarch64` (Raspberry Pi).

## Quickstart
```rust,no_run
use std::time::Duration;
use tracing_subscriber::EnvFilter;
use video_inference::{DetectionConfig, detect_video, Result};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();
    // run detections in 1s intervals
    let config = DetectionConfig {
        interval: Some(Duration::from_secs(1)),
        ..Default::default()
    };
    let path_video = "video.mp4";
    let path_onnx = "model.onnx";
    let detections = detect_video(path_video, path_onnx, &config)?;
    Ok(())
}
```

See the [examples](./examples/) directory for more usage patterns.

## Profiling

Please read the [profiling setup guide](./.github/skills/profiling-setup/) for examples.

## Troubleshooting

Specifically for `linux/cuda` [execution providers](https://ort.pyke.io/perf/execution-providers) issues there's a [skill-formatted guide](./.github/skills/cuda-setup/) with instructions that turned out to be helpful for my own setup.