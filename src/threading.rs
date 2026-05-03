use std::sync::mpsc::Receiver;

use crate::DetectionConfig;
use crate::Model;
use crate::Result;
use crate::detection::Detection;
use crate::detection::detect_image;
use crate::video::DecodedFrame;

pub(crate) fn detection_handler(
    rx: Receiver<DecodedFrame>,
    model: &mut Model,
    config: DetectionConfig,
    size_video: (u32, u32),
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();
    while let Ok(frame) = rx.recv() {
        let detection = detect_image(
            &mut model.session,
            frame.array,
            &config,
            size_video,
            model.size,
            frame.index as u32,
        )?;
        detections.push(detection);
    }
    Ok(detections)
}
