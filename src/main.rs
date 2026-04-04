use anyhow::{Error, anyhow};
use clap::Parser;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use video_inference::detect_video;

fn resolve_checkpoint(args: &Args) -> Result<PathBuf, Error> {
    let path = match &args.checkpoint {
        Some(p) => p.clone(),
        None => {
            let _ = dotenvy::dotenv();
            let val = std::env::var("ONNX_CHECKPOINT_PATH").map_err(|_| {
                anyhow!("No --checkpoint arg and ONNX_CHECKPOINT_PATH not set in .env")
            })?;
            PathBuf::from(val)
        }
    };
    if !path.is_file() {
        return Err(anyhow!(
            "Checkpoint path does not exist or is not a file: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[derive(Parser)]
struct Args {
    /// Path to the source video file
    #[arg(short, long)]
    source: PathBuf,

    /// Path to the ONNX checkpoint file (falls back to ONNX_CHECKPOINT_PATH in .env)
    #[arg(short, long)]
    checkpoint: Option<PathBuf>,
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("error,video_inference=trace"))
        .init();

    video_rs::init().unwrap();

    let args = Args::parse();
    let checkpoint = resolve_checkpoint(&args)?;
    let path_source = args.source;
    let paths_mp4s: Vec<PathBuf> = if path_source.is_dir() {
        std::fs::read_dir(&path_source)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension()?.to_str()? == "mp4").then_some(path)
            })
            .collect()
    } else if path_source.extension().and_then(|e| e.to_str()) == Some("mp4") {
        vec![path_source]
    } else {
        return Err(anyhow!("Source must be a directory or an .mp4 file"));
    };
    let _results: Vec<_> = paths_mp4s
        .par_iter()
        .map(|path_mp4| detect_video(path_mp4, &checkpoint))
        .collect();
    Ok(())
}
