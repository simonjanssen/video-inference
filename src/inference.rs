use anyhow::Error;
use fast_image_resize as fr;
use ndarray::Array3;
use ort::{
    ep::CoreML,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
};
use std::{path::Path, time};
use tracing::debug;

use crate::Config;
use crate::detection::{BoundingBox, extract_bboxes};
use crate::loaders::load_resized_tensor;

pub fn load_session(path_onnx: impl AsRef<Path>) -> Result<Session, Error> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .with_execution_providers([CoreML::default()
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .build()])
        .unwrap()
        .commit_from_file(path_onnx)?;
    //let _shape = determine_input_shape(&session, "inputs0")?;
    Ok(session)
}

// fn determine_input_shape(session: &Session, input_name: &str) -> Result<(u32, u32), Error> {
//     let inputs = session.inputs();
//     debug!("{:?}", inputs);
//     for input in inputs {
//         if input.name() == input_name
//             && let Some(dims) = input.dtype().tensor_shape()
//         {
//             let d = dims.len();
//             if d > 1 {
//                 let (w, h) = (dims[d - 2], dims[d - 1]);
//                 return Ok((w as u32, h as u32));
//             }
//         }
//     }
//     Err(anyhow!("Failed to determine input shape!"))
// }

pub(crate) fn detect_frame(
    session: &mut Session,
    img_arr: Array3<u8>,
    config: &Config,
    resizer: &mut fr::Resizer,
    dst_image: &mut fr::images::Image,
) -> Result<Vec<BoundingBox>, Error> {
    let t = time::Instant::now();
    let image_tensor = load_resized_tensor(img_arr, config, resizer, dst_image)?;
    let session_inputs = inputs! {
        "images" => image_tensor
    };
    let dt_preprocess = t.elapsed();
    let t = time::Instant::now();
    let session_outputs = session.run(session_inputs)?;
    let dt_inference = t.elapsed();
    let t = time::Instant::now();
    let bboxes = extract_bboxes(session_outputs, config)?;
    let dt_postprocess = t.elapsed();

    let class_idxs: Vec<_> = bboxes.iter().map(|b| b.class_idx).collect();
    debug!(
        "pre={:.1?} inf={:.1?} post={:.1?} classes={:?}",
        dt_preprocess, dt_inference, dt_postprocess, class_idxs
    );

    Ok(bboxes)
}
