use thiserror::Error;

#[derive(Debug, Error)]
pub enum VideoInferenceError {
    #[error("ONNX Error: {detail} ({source})")]
    Onnx {
        detail: String,
        #[source]
        source: ort::Error,
    },

    #[error("IO Error: {0}")]
    Io(String),

    #[error("Video Decoder Error: {detail} ({source}")]
    Video {
        detail: String,
        #[source]
        source: video_rs::Error,
    },

    #[error("Threading Error ({0})")]
    Thread(String),
}
