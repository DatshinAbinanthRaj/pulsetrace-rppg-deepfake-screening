"""Face detection, tracking and skin regions of interest (forehead, left cheek, right cheek)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ROI_NAMES = ("forehead", "left_cheek", "right_cheek")

# ROI boxes as fractions of the face box: (x0, y0, x1, y1).
# "left"/"right" are image-left/image-right.
ROI_LAYOUT = {
    "forehead": (0.30, 0.12, 0.70, 0.28),
    "left_cheek": (0.17, 0.52, 0.38, 0.72),
    "right_cheek": (0.62, 0.52, 0.83, 0.72),
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    def as_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))


def roi_boxes(face: Box) -> dict[str, tuple[int, int, int, int]]:
    """Return {name: (x0, y0, x1, y1)} pixel boxes for each ROI."""
    out = {}
    for name, (fx0, fy0, fx1, fy1) in ROI_LAYOUT.items():
        out[name] = (
            int(face.x + fx0 * face.w),
            int(face.y + fy0 * face.h),
            int(face.x + fx1 * face.w),
            int(face.y + fy1 * face.h),
        )
    return out


def skin_mask(patch_bgr: np.ndarray) -> np.ndarray:
    """Loose YCrCb skin mask. Removes hair, eyebrows, glasses rims and background from an ROI."""
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    return (y > 60) & (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)


@dataclass
class FaceTracker:
    """Haar-cascade face detector with periodic re-detection and exponential smoothing.

    Smoothing matters: a jittery box moves the ROIs, and ROI movement shows up
    as colour noise that can swamp the pulse.
    """

    detect_every: int = 5
    smoothing: float = 0.7
    min_face_frac: float = 0.12
    _cascade: cv2.CascadeClassifier = field(init=False, repr=False)
    _fallback: cv2.CascadeClassifier = field(init=False, repr=False)
    _box: Box | None = field(default=None, init=False)
    _frame_idx: int = field(default=0, init=False)
    detections: int = field(default=0, init=False)
    attempts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._cascade = self._load("haarcascade_frontalface_default.xml")
        self._fallback = self._load("haarcascade_frontalface_alt2.xml")

    @staticmethod
    def _load(name: str) -> cv2.CascadeClassifier:
        path = cv2.data.haarcascades + name
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")
        return c

    def _detect(self, frame: np.ndarray) -> Box | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_side = int(min(frame.shape[:2]) * self.min_face_frac)
        kw = dict(scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        faces = self._cascade.detectMultiScale(gray, **kw)
        if len(faces) == 0:  # second chance: different cascade on contrast-equalised image (dim/backlit faces)
            faces = self._fallback.detectMultiScale(cv2.equalizeHist(gray), **kw)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        return Box(float(x), float(y), float(w), float(h))

    def update(self, frame: np.ndarray) -> Box | None:
        if self._box is None or self._frame_idx % self.detect_every == 0:
            self.attempts += 1
            found = self._detect(frame)
            if found is not None:
                self.detections += 1
                if self._box is None:
                    self._box = found
                else:
                    a = self.smoothing
                    b = self._box
                    self._box = Box(
                        a * b.x + (1 - a) * found.x,
                        a * b.y + (1 - a) * found.y,
                        a * b.w + (1 - a) * found.w,
                        a * b.h + (1 - a) * found.h,
                    )
        self._frame_idx += 1
        return self._box

    @property
    def detection_rate(self) -> float:
        return self.detections / self.attempts if self.attempts else 0.0


def roi_means(frame: np.ndarray, face: Box, use_skin_mask: bool = True) -> dict[str, np.ndarray | None]:
    """Mean (R, G, B) of skin pixels inside each ROI, or None if the ROI is unusable."""
    h, w = frame.shape[:2]
    out: dict[str, np.ndarray | None] = {}
    for name, (x0, y0, x1, y1) in roi_boxes(face).items():
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            out[name] = None
            continue
        patch = frame[y0:y1, x0:x1]
        pixels = patch.reshape(-1, 3)
        if use_skin_mask:
            m = skin_mask(patch).reshape(-1)
            if m.mean() > 0.25:  # enough skin; otherwise fall back to all pixels
                pixels = pixels[m]
        bgr = pixels.astype(np.float64).mean(axis=0)
        out[name] = bgr[::-1].copy()  # -> RGB
    return out
