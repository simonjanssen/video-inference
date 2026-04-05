use anyhow::Error;
use fast_image_resize as fr;
use ndarray::Array3;
use ort::value::{Tensor, TensorValueType, Value};

use crate::runtime::RuntimeConfig;

/// Resize a decoded video frame and produce a normalized NCHW f32 tensor for inference.
///
/// Optimizations:
/// - Uses `fast_image_resize` (SIMD-accelerated) instead of the `image` crate's pure-Rust
///   resizer, which is ~5-10x faster for bilinear interpolation.
/// - The `resizer` and `dst_image` are pre-allocated by the caller and reused across frames
///   to avoid per-frame heap allocations (~1.2MB each) and to let the resizer cache its
///   interpolation coefficient tables for the given src→dst size pair.
/// - Normalization (u8→f32, /255) and HWC→CHW transpose are fused into a single pass,
///   eliminating an intermediate f32 HWC buffer and the non-contiguous copy that
///   `ndarray::permuted_axes` + `Tensor::from_array` would otherwise require.
pub(crate) fn load_resized_tensor(
    img_arr: &Array3<u8>,
    config: &RuntimeConfig,
    resizer: &mut fr::Resizer,
    dst_image: &mut fr::images::Image,
) -> Result<Value<TensorValueType<f32>>, Error> {
    let (raw, _) = img_arr
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset();
    let src_image = fr::images::Image::from_vec_u8(
        config.target_width,
        config.target_height,
        raw,
        fr::PixelType::U8x3,
    )?;
    resizer.resize(&src_image, dst_image, None)?;

    // Single-pass: normalize u8→f32 and transpose HWC→CHW
    let src = dst_image.buffer();
    let pixels = (config.input_tensor_height * config.input_tensor_width) as usize;
    let mut chw = vec![0.0f32; 3 * pixels];
    for i in 0..pixels {
        chw[i] = src[i * 3] as f32 / 255.0;
        chw[pixels + i] = src[i * 3 + 1] as f32 / 255.0;
        chw[2 * pixels + i] = src[i * 3 + 2] as f32 / 255.0;
    }

    let arr = ndarray::Array::from_shape_vec(
        (
            1,
            3,
            config.input_tensor_height as usize,
            config.input_tensor_width as usize,
        ),
        chw,
    )?;
    let tensor = Tensor::from_array(arr)?;
    Ok(tensor)
}
