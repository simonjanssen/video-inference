use anyhow::Error;
use fast_image_resize as fr;
use ndarray::Array3;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use crate::DetectionConfig;
use crate::detection::{BoundingBox, detect_frame};
use crate::inference::{detect_input_shape, load_session};
use crate::runtime::RuntimeConfig;

pub(crate) struct DetectionTask {
    frame: Array3<u8>,
}

impl DetectionTask {
    pub fn new(frame: Array3<u8>) -> Self {
        Self { frame }
    }
}

pub(crate) fn detection_handler(
    rx: Receiver<DetectionTask>,
    path_onnx: PathBuf,
    target_size: (u32, u32),
    config: DetectionConfig,
) -> Result<Vec<Vec<BoundingBox>>, Error> {
    let mut session = load_session(path_onnx)?;
    let (input_tensor_width, input_tensor_height) = detect_input_shape(&session, Some("images"))?;
    let mut dst_image =
        fr::images::Image::new(input_tensor_width, input_tensor_height, fr::PixelType::U8x3);
    let mut resizer = fr::Resizer::new();
    let (target_width, target_height) = target_size;
    let runtime_config = RuntimeConfig {
        conf_thres: config.conf_thres,
        iou_thres: config.iou_thres,
        max_detect: config.max_detect,
        input_tensor_height,
        input_tensor_width,
        target_height,
        target_width,
        output_tensor_name: config.output_tensor_name.to_owned(),
    };
    let mut bboxes = Vec::new();
    while let Ok(task) = rx.recv() {
        let bboxes_frame = detect_frame(
            &task.frame,
            None,
            &runtime_config,
            &mut session,
            &mut resizer,
            &mut dst_image,
        )?;
        bboxes.push(bboxes_frame);
    }
    Ok(bboxes)
}
