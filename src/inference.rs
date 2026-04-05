use anyhow::{Error, bail};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use tracing::debug;

pub fn load_session(path_onnx: impl AsRef<Path>) -> Result<Session, Error> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        use ort::ep::CoreML;
        builder = builder
            .with_execution_providers([CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                .build()])
            .unwrap();
    }

    let session = builder.commit_from_file(path_onnx)?;
    Ok(session)
}

pub(crate) fn detect_input_shape(
    session: &Session,
    input_name: Option<&str>,
) -> Result<(u32, u32), Error> {
    let inputs = session.inputs();
    for input in inputs {
        debug!("{:?}", input);
        let name_matched = input_name.is_none_or(|name| input.name() == name);
        if name_matched && let Some(dims) = input.dtype().tensor_shape() {
            let d = dims.len();
            if d > 1 {
                let (w, h) = (dims[d - 2], dims[d - 1]);
                debug!("onnx model has image input shape {} x {}", w, h);
                return Ok((w as u32, h as u32));
            }
        }
    }
    bail!("Failed to determine input shape!")
}

// pub(crate) fn detect_frame(
//     session: &mut Session,
//     img_arr: Array3<u8>,
//     config: &Config,
//     resizer: &mut fr::Resizer,
//     dst_image: &mut fr::images::Image,
// ) -> Result<Vec<BoundingBox>, Error> {
//     let t = time::Instant::now();
//     let image_tensor = load_resized_tensor(img_arr, config, resizer, dst_image)?;
//     let session_inputs = inputs! {
//         "images" => image_tensor
//     };
//     let dt_preprocess = t.elapsed();
//     let t = time::Instant::now();
//     let session_outputs = session.run(session_inputs)?;
//     let dt_inference = t.elapsed();
//     let t = time::Instant::now();
//     let bboxes = extract_bboxes(session_outputs, config)?;
//     let dt_postprocess = t.elapsed();

//     let class_idxs: Vec<_> = bboxes.iter().map(|b| b.class_idx).collect();
//     debug!(
//         "pre={:.1?} inf={:.1?} post={:.1?} classes={:?}",
//         dt_preprocess, dt_inference, dt_postprocess, class_idxs
//     );

//     Ok(bboxes)
// }
