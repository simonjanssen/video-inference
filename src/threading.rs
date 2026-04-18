use ndarray::Array3;
use ort::session::Session;
use std::sync::mpsc::Receiver;

use crate::DetectionConfig;
use crate::Result;
use crate::detection::{BoundingBox, detect_image};

pub(crate) struct DetectionTask {
    frame: Array3<u8>,
    frame_idx: Option<u32>,
}

impl DetectionTask {
    pub fn new(frame: Array3<u8>, frame_idx: Option<u32>) -> Self {
        Self { frame, frame_idx }
    }
}

pub(crate) fn detection_handler(
    rx: Receiver<DetectionTask>,
    mut session: Session,
    config: DetectionConfig,
    size_video: (u32, u32),
    size_onnx: (u32, u32),
) -> Result<Vec<Vec<BoundingBox>>> {
    let mut bboxes = Vec::new();
    while let Ok(task) = rx.recv() {
        let bboxes_frame = detect_image(&mut session, task.frame, &config, size_video, size_onnx, task.frame_idx)?;
        bboxes.push(bboxes_frame);
    }
    Ok(bboxes)
}
