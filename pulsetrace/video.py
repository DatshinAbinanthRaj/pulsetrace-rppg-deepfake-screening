"""Video input helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info = VideoInfo(
            fps=fps if 1.0 <= fps <= 240.0 else 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()
    return info


def iter_frames(path: str | Path, max_seconds: float | None = None, max_width: int = 640) -> Iterator[np.ndarray]:
    """Yield BGR frames, downscaled so width <= max_width (keeps the pipeline fast)."""
    info = probe(path)
    limit = int(max_seconds * info.fps) if max_seconds else None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Cannot open video: {path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            yield frame
            n += 1
            if limit is not None and n >= limit:
                break
    finally:
        cap.release()


def write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> Path:
    """Write frames losslessly-ish (MJPG at max quality in .avi) so tiny colour changes survive."""
    path = Path(path)
    if not frames:
        raise VideoError("No frames to write")
    h, w = frames[0].shape[:2]
    suffix = path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if suffix != ".mp4":
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    if not writer.isOpened():
        raise VideoError(f"Cannot write video: {path}")
    for f in frames:
        writer.write(f)
    writer.release()
    return path
