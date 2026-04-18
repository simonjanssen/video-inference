use ndarray::Array3;
use ort::session::Session;
use std::sync::mpsc::Receiver;

use crate::DetectionConfig;
use crate::Result;
use crate::detection::Detection;
use crate::detection::detect_image;

pub(crate) struct DetectionTask {
    frame: Array3<u8>,
    frame_idx: u32,
}

impl DetectionTask {
    pub fn new(frame: Array3<u8>, frame_idx: u32) -> Self {
        Self { frame, frame_idx }
    }
}

pub(crate) fn detection_handler(
    rx: Receiver<DetectionTask>,
    mut session: Session,
    config: DetectionConfig,
    size_video: (u32, u32),
    size_onnx: (u32, u32),
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();
    while let Ok(task) = rx.recv() {
        let detection = detect_image(
            &mut session,
            task.frame,
            &config,
            size_video,
            size_onnx,
            task.frame_idx,
        )?;
        detections.push(detection);
    }
    Ok(detections)
}
