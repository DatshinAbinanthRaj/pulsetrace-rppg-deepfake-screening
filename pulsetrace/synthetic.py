"""Synthetic benchmark videos with known ground truth.

No real person appears anywhere: faces are drawn procedurally. "Real" clips carry
one shared pulse across the face (with realistic nuisances: lighting drift, head
motion, sensor noise, heart-rate variability). "Fake" clips imitate failure modes
reported for generated faces: no pulse, regions pulsing independently, or
frame-to-frame colour flicker from per-frame synthesis.

These clips test that the pipeline behaves as designed. They are NOT a substitute
for evaluation on real deepfake datasets (see docs/EVALUATION.md).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pulsetrace.video import write_video

KINDS = ("real", "no_pulse", "incoherent", "flicker")
SKIN_TONES_BGR = [(120, 150, 200), (95, 125, 175), (70, 100, 150), (140, 170, 220), (60, 85, 125)]
# Blood-volume pulse direction in normalised RGB (de Haan & van Leest 2014), stored as BGR.
PULSE_BGR = np.array([0.53, 0.77, 0.33])


@dataclass
class SyntheticClip:
    frames: list[np.ndarray]
    fps: float
    kind: str
    label: str  # "real" | "fake"
    bpm: float | None


def draw_face(h: int = 480, w: int = 640, skin=(120, 150, 200), scale: float = 1.0, bg=(70, 60, 55)):
    """Return (image_float, face_mask, geometry) for a procedurally drawn frontal face."""
    img = np.full((h, w, 3), bg, np.uint8)
    cx, cy = w // 2, h // 2
    fw, fh = int(110 * scale), int(145 * scale)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (fw, fh), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy - fh + int(25 * scale)), (fw + 5, int(55 * scale)), 0, 180, 360, (40, 40, 50), -1)
    for s in (-1, 1):
        ex, ey = cx + s * int(45 * scale), cy - int(30 * scale)
        cv2.ellipse(img, (ex, ey - int(25 * scale)), (int(30 * scale), int(7 * scale)), 0, 0, 360, (40, 45, 60), -1)
        cv2.ellipse(img, (ex, ey), (int(24 * scale), int(12 * scale)), 0, 0, 360, (235, 235, 235), -1)
        cv2.circle(img, (ex, ey), int(10 * scale), (50, 40, 30), -1)
    shade = tuple(int(c * 0.8) for c in skin)
    cv2.line(img, (cx, cy - int(20 * scale)), (cx - int(8 * scale), cy + int(35 * scale)), shade, max(2, int(5 * scale)))
    cv2.ellipse(img, (cx, cy + int(80 * scale)), (int(45 * scale), int(14 * scale)), 0, 0, 360, (70, 70, 150), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    return img.astype(np.float64), mask > 0, (cx, cy, fw, fh)


def _ppg_wave(t: np.ndarray, bpm: float, rng: np.random.Generator, hrv: float = 0.03) -> np.ndarray:
    f = bpm / 60.0 * (1 + hrv * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(f) * (t[1] - t[0]) + rng.uniform(0, 6.28)
    wave = np.sin(phase) + 0.35 * np.sin(2 * phase + 0.6)
    return wave / wave.std()


def make_clip(
    kind: str = "real",
    seconds: float = 12.0,
    fps: float = 30.0,
    bpm: float | None = None,
    seed: int = 0,
    pulse_amp: float = 0.006,
    noise_sigma: float = 1.5,
    motion_px: float = 2.0,
    light_drift: float = 0.03,
    size: tuple[int, int] = (480, 640),
) -> SyntheticClip:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rng = np.random.default_rng(seed)
    h, w = size
    skin = SKIN_TONES_BGR[seed % len(SKIN_TONES_BGR)]
    base, face, (cx, cy, fw, fh) = draw_face(h, w, skin, scale=rng.uniform(0.9, 1.1))
    n = int(seconds * fps)
    t = np.arange(n) / fps
    bpm = float(bpm if bpm is not None else rng.uniform(55, 110))

    yy, xx = np.mgrid[0:h, 0:w]
    forehead_line = cy - 0.3 * fh
    zones = {
        "forehead": face & (yy < forehead_line),
        "left": face & (yy >= forehead_line) & (xx < cx),
        "right": face & (yy >= forehead_line) & (xx >= cx),
    }
    waves: dict[str, np.ndarray] = {}
    if kind == "real":
        shared = _ppg_wave(t, bpm, rng)
        for i, z in enumerate(zones):  # small pulse-transit delay between regions
            waves[z] = np.interp(t - 0.02 * i, t, shared)
    elif kind == "incoherent":
        options = rng.permutation(np.arange(50, 130, 17))[: len(zones)]
        for z, b in zip(zones, options, strict=True):
            waves[z] = _ppg_wave(t, float(b), rng, hrv=0.15)
    elif kind == "flicker":
        for z in zones:
            waves[z] = rng.normal(0, 1, n)
    else:  # no_pulse
        for z in zones:
            waves[z] = np.zeros(n)

    light = 1 + light_drift * np.sin(2 * np.pi * rng.uniform(0.03, 0.08) * t) + 0.01 * (t / t[-1])
    mx = motion_px * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t)
    my = 0.7 * motion_px * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + 1.0)

    frames = []
    for i in range(n):
        f = base * light[i]
        for z, m in zones.items():
            if waves[z][i] != 0:
                f[m] *= 1 + pulse_amp * waves[z][i] * PULSE_BGR
        f += rng.normal(0, noise_sigma, f.shape)
        if motion_px:
            M = np.float32([[1, 0, mx[i]], [0, 1, my[i]]])
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return SyntheticClip(frames, fps, kind, "real" if kind == "real" else "fake", bpm if kind == "real" else None)


def generate_dataset(out_dir: str | Path, per_kind: int = 5, seconds: float = 12.0, seed: int = 0) -> Path:
    """Write a labelled benchmark (.avi clips + manifest.csv). Returns the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for kind in KINDS:
        reps = per_kind * 3 if kind == "real" else per_kind  # balance real vs fake
        for j in range(reps):
            clip = make_clip(kind, seconds=seconds, seed=seed + k)
            name = f"{kind}_{j:02d}.avi"
            write_video(out / name, clip.frames, clip.fps)
            rows.append({"path": name, "label": clip.label, "kind": kind, "gt_bpm": f"{clip.bpm:.1f}" if clip.bpm else ""})
            k += 1
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["path", "label", "kind", "gt_bpm"])
        wr.writeheader()
        wr.writerows(rows)
    return manifest
