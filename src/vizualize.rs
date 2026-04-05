use crate::detection::BoundingBox;
use ab_glyph::FontArc;
use anyhow::{Error, Result};
use image::{DynamicImage, Rgba};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use ndarray::Array3;

// manually determined scale factors to print annotations / draw boxes
const SCALE_THICKNESS: f32 = 15. / 3726.;
const SCALE_FONT: f32 = 100. / 3726.;

const COLORS: [Rgba<u8>; 10] = [
    Rgba([204, 102, 204, 255]), // Darker Magenta
    Rgba([204, 102, 136, 255]), // Darker Pink
    Rgba([204, 163, 102, 255]), // Darker Peach
    Rgba([204, 204, 102, 255]), // Darker Yellow
    Rgba([102, 204, 142, 255]), // Darker Mint Green
    Rgba([102, 163, 204, 255]), // Darker Blue
    Rgba([163, 102, 204, 255]), // Darker Lavender
    Rgba([204, 102, 153, 255]), // Darker Rose
    Rgba([163, 204, 102, 255]), // Darker Lime
    Rgba([102, 204, 204, 255]), // Darker Cyan
];

/// # Draw Rectangles
/// Draws hollow rectangles onto input image using BoundingBox coordinates
/// Applies box thickness that is dynamically scaled by input image resolution
pub(crate) fn draw_bboxes(
    mut img: DynamicImage,
    bboxes: &[BoundingBox],
) -> Result<DynamicImage, Error> {
    let img_d = img.width().min(img.height()) as f32;
    let thickness = SCALE_THICKNESS * img_d; // scale thickness by smaller image edge
    let thickness = (thickness as u32).max(1);
    let font_data = include_bytes!("../assets/DejaVuSans.ttf");
    let font = FontArc::try_from_slice(font_data as &[u8]).unwrap();
    let font_scale = SCALE_FONT * img_d;
    let font_offset = (font_scale * 1.1) as u32;

    for bbox in bboxes.iter() {
        let box_color = COLORS[(bbox.class_idx as usize) % COLORS.len()];
        let (x1, y1, w, h) = bbox.x1y1wh();
        let label = format!("class {} ({:.2})", bbox.class_idx, bbox.score);
        println!("{}", &label);
        draw_text_mut(
            &mut img,
            box_color,
            x1 as i32,
            (y1 - font_offset) as i32,
            font_scale,
            &font,
            &label,
        );
        for t in 0..thickness {
            let x = if x1 > t { x1 - t } else { x1 };
            let y = if y1 > t { y1 - t } else { y1 };
            let w = w + 2 * t;
            let h = h + 2 * t;
            let rect = Rect::at(x as i32, y as i32).of_size(w, h);
            draw_hollow_rect_mut(&mut img, rect, box_color);
        }
    }
    Ok(img)
}

pub(crate) fn draw_bboxes_arr(
    img_arr: Array3<u8>,
    bboxes: &[BoundingBox],
) -> Result<Array3<u8>, Error> {
    let (h, w, _) = img_arr.dim();
    let (raw, _) = img_arr.into_raw_vec_and_offset();
    let rgb_img = image::RgbImage::from_raw(w as u32, h as u32, raw).unwrap();
    let dyn_img = image::DynamicImage::ImageRgb8(rgb_img);
    let annotated = draw_bboxes(dyn_img, bboxes)?;
    let annotated_arr3 = img_to_arr3(&annotated)?;
    Ok(annotated_arr3)
}

pub(crate) fn img_to_arr3(img: &DynamicImage) -> Result<Array3<u8>, Error> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let raw = rgb.into_raw(); // Vec<u8>, length = height * width * 3
    let arr3 = Array3::from_shape_vec((height as usize, width as usize, 3), raw)?;
    Ok(arr3)
}
