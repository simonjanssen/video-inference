use video_inference::detection::{BoundingBox, nms};

fn bbox(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_idx: i32) -> BoundingBox {
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        score,
        class_idx,
        frame_idx: None,
    }
}

#[test]
fn bounding_box_area() {
    let b = bbox(0.0, 0.0, 10.0, 20.0, 0.9, 0);
    assert!((b.area() - 200.0).abs() < f32::EPSILON);
}

#[test]
fn bounding_box_area_degenerate() {
    let b = bbox(5.0, 5.0, 5.0, 5.0, 0.5, 0);
    assert_eq!(b.area(), 0.0);
}

#[test]
fn iou_identical_boxes() {
    let a = bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    let b = bbox(0.0, 0.0, 10.0, 10.0, 0.8, 0);
    assert!((a.iou(&b) - 1.0).abs() < 1e-6);
}

#[test]
fn iou_no_overlap() {
    let a = bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    let b = bbox(20.0, 20.0, 30.0, 30.0, 0.8, 0);
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn iou_partial_overlap() {
    let a = bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    let b = bbox(5.0, 5.0, 15.0, 15.0, 0.8, 0);
    // intersection = 5*5 = 25, union = 100+100-25 = 175
    let expected = 25.0 / 175.0;
    assert!((a.iou(&b) - expected).abs() < 1e-6);
}

#[test]
fn scale_bounding_box() {
    let mut b = bbox(10.0, 20.0, 30.0, 40.0, 0.9, 0);
    b.scale(2.0, 0.5);
    assert_eq!(b.x1, 20.0);
    assert_eq!(b.y1, 10.0);
    assert_eq!(b.x2, 60.0);
    assert_eq!(b.y2, 20.0);
}

#[test]
fn nms_empty_input() {
    let result = nms(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn nms_single_box() {
    let boxes = vec![bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0)];
    let result = nms(&boxes, 0.5);
    assert_eq!(result.len(), 1);
}

#[test]
fn nms_suppresses_overlapping_same_class() {
    let boxes = vec![
        bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        bbox(1.0, 1.0, 11.0, 11.0, 0.8, 0), // high overlap, lower score
    ];
    let result = nms(&boxes, 0.5);
    assert_eq!(result.len(), 1);
    assert!((result[0].score - 0.9).abs() < f32::EPSILON);
}

#[test]
fn nms_keeps_non_overlapping() {
    let boxes = vec![
        bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        bbox(50.0, 50.0, 60.0, 60.0, 0.8, 0),
    ];
    let result = nms(&boxes, 0.5);
    assert_eq!(result.len(), 2);
}

#[test]
fn nms_different_classes_not_suppressed() {
    // Two overlapping boxes of different classes should both survive
    let boxes = vec![
        bbox(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        bbox(1.0, 1.0, 11.0, 11.0, 0.8, 1),
    ];
    let result = nms(&boxes, 0.5);
    assert_eq!(result.len(), 2);
}

#[test]
fn nms_returns_sorted_by_score_descending() {
    let boxes = vec![
        bbox(0.0, 0.0, 10.0, 10.0, 0.5, 0),
        bbox(50.0, 50.0, 60.0, 60.0, 0.9, 0),
        bbox(100.0, 100.0, 110.0, 110.0, 0.7, 0),
    ];
    let result = nms(&boxes, 0.5);
    assert_eq!(result.len(), 3);
    assert!(result[0].score >= result[1].score);
    assert!(result[1].score >= result[2].score);
}
