use ndarray::Array3;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{Tensor, TensorValueType, Value};
use std::path::Path;
use tracing::debug;

use crate::{Result, VideoInferenceError};

pub(crate) fn load_session(path_onnx: impl AsRef<Path>) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| VideoInferenceError::Onnx {
            detail: "Failed to initialize ONNX runtime session!".to_string(),
            source: e,
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VideoInferenceError::Onnx {
            detail: "Failed to apply graph optimization!".to_string(),
            source: e.into(),
        })?
        .with_intra_threads(4)
        .map_err(|e| VideoInferenceError::Onnx {
            detail: "Failed to finalize ONNX runtime session!".to_string(),
            source: e.into(),
        })?;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        use ort::ep::CoreML;
        builder = builder
            .with_execution_providers([CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                .build()])
            .map_err(|e| VideoInferenceError::Onnx {
                detail: "Failed to load `CoreML` execution provider!".to_string(),
                source: e.into(),
            })?;
        debug!("using `CoreML` execution provider for ONNX inference");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        use ort::ep::CUDA;
        builder = builder
            .with_execution_providers([CUDA::default().build()])
            .map_err(|e| VideoInferenceError::Onnx {
                detail: "Failed to register execution providers!".to_string(),
                source: e.into(),
            })?;
    }

    let session = builder
        .commit_from_file(path_onnx)
        .map_err(|e| VideoInferenceError::Onnx {
            detail: "Failed to load ONNX model from file!".to_string(),
            source: e,
        })?;
    Ok(session)
}

pub(crate) fn detect_input_shape(
    session: &Session,
    input_name: Option<&str>,
) -> Result<(u32, u32)> {
    let inputs = session.inputs();
    for input in inputs {
        //debug!("{:?}", input);
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
    Err(VideoInferenceError::Io(
        "Failed to detect model input dimensions!".to_string(),
    ))
}

/// Normalize a HWC u8 frame into a NCHW f32 tensor for inference (no resize).
///
/// Normalization (u8→f32, /255) and HWC→CHW transpose are fused into a single pass,
/// eliminating an intermediate f32 HWC buffer and the non-contiguous copy that
/// `ndarray::permuted_axes` + `Tensor::from_array` would otherwise require.
pub(crate) fn load_tensor(
    img_arr: &Array3<u8>,
    size_tensor: (u32, u32),
) -> Result<Value<TensorValueType<f32>>> {
    let std_layout = img_arr.as_standard_layout();
    let src = std_layout.as_slice().expect("contiguous HWC frame");
    hwc_to_nchw_tensor(src, size_tensor)
}

/// Single-pass: normalize u8→f32 and transpose HWC→CHW into a NCHW tensor.
fn hwc_to_nchw_tensor(src: &[u8], size_tensor: (u32, u32)) -> Result<Value<TensorValueType<f32>>> {
    let (w, h) = size_tensor;
    let pixels = (w * h) as usize;
    let mut chw = vec![0.0f32; 3 * pixels];
    for i in 0..pixels {
        chw[i] = src[i * 3] as f32 / 255.0;
        chw[pixels + i] = src[i * 3 + 1] as f32 / 255.0;
        chw[2 * pixels + i] = src[i * 3 + 2] as f32 / 255.0;
    }
    let arr = ndarray::Array::from_shape_vec((1, 3, h as usize, w as usize), chw)
        .map_err(|_| VideoInferenceError::Io("Failed to load image as nchw array!".to_string()))?;
    let tensor = Tensor::from_array(arr).map_err(|e| VideoInferenceError::Onnx {
        detail: "Failed to load input tensor from image array".to_string(),
        source: e,
    })?;
    Ok(tensor)
}
