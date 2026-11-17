import numpy as np

from pulsetrace.face import ROI_NAMES, Box, FaceTracker, roi_boxes, roi_means, skin_mask
from pulsetrace.synthetic import draw_face


def test_tracker_finds_synthetic_face():
    img, _, (cx, cy, fw, fh) = draw_face()
    tr = FaceTracker()
    box = tr.update(img.astype(np.uint8))
    assert box is not None
    assert abs((box.x + box.w / 2) - cx) < 30
    assert tr.detection_rate == 1.0


def test_tracker_handles_blank_frames():
    tr = FaceTracker()
    assert tr.update(np.zeros((240, 320, 3), np.uint8)) is None
    assert tr.detection_rate == 0.0


def test_tracker_keeps_last_box_when_face_lost():
    img, _, _ = draw_face()
    tr = FaceTracker(detect_every=1)
    first = tr.update(img.astype(np.uint8))
    again = tr.update(np.zeros_like(img, dtype=np.uint8))
    assert again is first or (again.x == first.x and again.w == first.w)


def test_roi_layout_inside_face_and_ordered():
    boxes = roi_boxes(Box(100, 100, 200, 200))
    assert set(boxes) == set(ROI_NAMES)
    for x0, y0, x1, y1 in boxes.values():
        assert 100 <= x0 < x1 <= 300 and 100 <= y0 < y1 <= 300
    assert boxes["forehead"][3] < boxes["left_cheek"][1]  # forehead above cheeks
    assert boxes["left_cheek"][2] < boxes["right_cheek"][0]  # cheeks don't overlap


def test_skin_mask_accepts_skin_rejects_dark_hair():
    skin = np.full((10, 10, 3), (120, 150, 200), np.uint8)
    hair = np.full((10, 10, 3), (40, 40, 50), np.uint8)
    assert skin_mask(skin).mean() > 0.9
    assert skin_mask(hair).mean() < 0.1


def test_roi_means_returns_rgb_and_none_off_frame():
    img = np.full((200, 200, 3), (10, 20, 30), np.uint8)
    out = roi_means(img, Box(0, 0, 200, 200), use_skin_mask=False)
    assert np.allclose(out["forehead"], [30, 20, 10])  # BGR -> RGB
    off = roi_means(img, Box(500, 500, 100, 100))
    assert all(v is None for v in off.values())
