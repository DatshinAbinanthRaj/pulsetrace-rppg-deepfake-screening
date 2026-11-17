"""End-to-end analysis: frames -> ROI colour traces -> pulse signals -> features -> score -> verdict."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pulsetrace import dsp
from pulsetrace.face import ROI_NAMES, Box, FaceTracker, roi_boxes, roi_means
from pulsetrace.video import VideoError, iter_frames, probe

VERDICT_REAL = "LIKELY_REAL"
VERDICT_FAKE = "LIKELY_FAKE"
VERDICT_REVIEW = "NEEDS_REVIEW"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class AnalysisConfig:
    method: str = "pos"  # green | pos | chrom
    min_seconds: float = 6.0
    max_seconds: float | None = 30.0
    min_fps: float = 10.0
    min_face_detection_rate: float = 0.5
    max_missing_frac: float = 0.2
    use_skin_mask: bool = True
    # Score -> verdict bands. Between the two thresholds the clip goes to a human.
    real_threshold: float = 0.60
    fake_threshold: float = 0.40
    # Sub-score shaping (calibrated on the synthetic benchmark, see docs/EVALUATION.md)
    snr_mid_db: float = 0.0
    snr_scale_db: float = 2.5
    bpm_tolerance: float = 12.0
    hr_std_tolerance: float = 20.0
    weights: dict[str, float] = field(
        default_factory=lambda: {"snr": 0.25, "agreement": 0.20, "correlation": 0.25, "coherence": 0.20, "stability": 0.10}
    )


@dataclass
class RoiResult:
    name: str
    bpm: float
    snr_db: float
    missing_frac: float


@dataclass
class AnalysisReport:
    verdict: str
    score: float | None
    bpm: float | None
    reasons: list[str]
    fps: float
    duration_s: float
    frames: int
    face_detection_rate: float
    method: str
    rois: list[RoiResult]
    features: dict[str, float]
    sub_scores: dict[str, float]
    elapsed_s: float
    # Arrays for plotting; excluded from JSON.
    signals: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    hr_trace: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    preview_frame: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("signals", "hr_trace", "preview_frame"):
            d.pop(k, None)
        return _clean(d)


def _clean(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 4)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _fill_gaps(series: list[np.ndarray | None]) -> tuple[np.ndarray | None, float]:
    """Linear-interpolate missing ROI samples. Returns (T,3) array and missing fraction."""
    n = len(series)
    ok = np.array([s is not None for s in series])
    miss = 1.0 - ok.mean() if n else 1.0
    if ok.sum() < 2:
        return None, miss
    idx = np.arange(n)
    vals = np.array([s if s is not None else [np.nan] * 3 for s in series], dtype=np.float64)
    for c in range(3):
        vals[:, c] = np.interp(idx, idx[ok], vals[ok, c])
    return vals, miss


def draw_rois(frame: np.ndarray, face: Box | None) -> np.ndarray:
    out = frame.copy()
    if face is None:
        return out
    x, y, w, h = face.as_int()
    cv2.rectangle(out, (x, y), (x + w, y + h), (120, 220, 120), 2)
    colours = {"forehead": (60, 200, 255), "left_cheek": (200, 120, 255), "right_cheek": (200, 120, 255)}
    for name, (x0, y0, x1, y1) in roi_boxes(face).items():
        cv2.rectangle(out, (x0, y0), (x1, y1), colours[name], 2)
        cv2.putText(out, name, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colours[name], 1, cv2.LINE_AA)
    return out


def analyze_frames(frames, fps: float, config: AnalysisConfig | None = None) -> AnalysisReport:
    """Analyse an iterable of BGR frames captured at `fps`."""
    cfg = config or AnalysisConfig()
    if cfg.method not in dsp.METHODS:
        raise ValueError(f"Unknown method {cfg.method!r}; choose from {sorted(dsp.METHODS)}")
    t0 = time.perf_counter()
    tracker = FaceTracker()
    traces: dict[str, list] = {n: [] for n in ROI_NAMES}
    preview, preview_box, n = None, None, 0
    for frame in frames:
        box = tracker.update(frame)
        if box is None:
            for name in ROI_NAMES:
                traces[name].append(None)
        else:
            for name, v in roi_means(frame, box, cfg.use_skin_mask).items():
                traces[name].append(v)
            if preview_box is None:
                preview, preview_box = frame, box
        if preview is None:
            preview = frame
        n += 1

    duration = n / fps if fps > 0 else 0.0
    reasons: list[str] = []
    base = dict(fps=fps, duration_s=duration, frames=n, face_detection_rate=tracker.detection_rate, method=cfg.method)

    def inconclusive(why: list[str], rois=None, features=None) -> AnalysisReport:
        return AnalysisReport(
            verdict=VERDICT_INCONCLUSIVE,
            score=None,
            bpm=None,
            reasons=why,
            rois=rois or [],
            features=features or {},
            sub_scores={},
            elapsed_s=time.perf_counter() - t0,
            preview_frame=draw_rois(preview, preview_box) if preview is not None else None,
            **base,
        )

    if fps < cfg.min_fps:
        reasons.append(f"Frame rate {fps:.1f} fps is below {cfg.min_fps:.0f} fps; the pulse cannot be sampled reliably.")
    if duration < cfg.min_seconds:
        reasons.append(f"Clip is {duration:.1f}s; at least {cfg.min_seconds:.0f}s is needed for a stable heart-rate estimate.")
    if tracker.detection_rate < cfg.min_face_detection_rate:
        reasons.append(f"A frontal face was found in only {tracker.detection_rate:.0%} of detection attempts.")
    if reasons:
        return inconclusive(reasons)

    signals: dict[str, np.ndarray] = {}
    rois: list[RoiResult] = []
    for name in ROI_NAMES:
        rgb, miss = _fill_gaps(traces[name])
        if rgb is None or miss > cfg.max_missing_frac:
            rois.append(RoiResult(name, float("nan"), float("-inf"), miss))
            continue
        raw = dsp.METHODS[cfg.method](rgb, fps)
        filt = dsp.bandpass(dsp.detrend(raw), fps)
        filt = filt / (filt.std() + 1e-12)
        signals[name] = filt
        bpm, snr = dsp.estimate_bpm(filt, fps)
        rois.append(RoiResult(name, bpm, snr, miss))

    if len(signals) < 2:
        return inconclusive(["Fewer than two usable skin regions (forehead/cheeks were occluded or out of frame)."], rois)

    combined = np.mean(np.stack(list(signals.values())), axis=0)
    signals["combined"] = combined
    bpm, snr_comb = dsp.estimate_bpm(combined, fps)
    hr_t, hr_b = dsp.bpm_over_time(combined, fps)

    names = [r for r in ROI_NAMES if r in signals]
    pairs = list(itertools.combinations(names, 2))
    corrs = [float(np.corrcoef(signals[a], signals[b])[0, 1]) for a, b in pairs]
    cohs = [dsp.band_coherence(signals[a], signals[b], fps, bpm / 60.0) for a, b in pairs]
    roi_bpms = [r.bpm for r in rois if not math.isnan(r.bpm)]
    roi_snrs = [r.snr_db for r in rois if math.isfinite(r.snr_db)]

    features = {
        "mean_roi_snr_db": float(np.mean(roi_snrs)) if roi_snrs else float("-inf"),
        "combined_snr_db": snr_comb,
        "roi_bpm_spread": float(max(roi_bpms) - min(roi_bpms)) if roi_bpms else float("inf"),
        "mean_pairwise_correlation": float(np.nanmean(corrs)),
        "mean_band_coherence": float(np.nanmean(cohs)),
        "hr_trace_std_bpm": float(np.nanstd(hr_b)),
    }
    sub = {
        "snr": _sigmoid((features["mean_roi_snr_db"] - cfg.snr_mid_db) / cfg.snr_scale_db),
        "agreement": float(np.clip(1 - features["roi_bpm_spread"] / cfg.bpm_tolerance, 0, 1)),
        "correlation": float(np.clip(features["mean_pairwise_correlation"], 0, 1)),
        "coherence": float(np.clip(np.nan_to_num(features["mean_band_coherence"]), 0, 1)),
        "stability": float(np.clip(1 - features["hr_trace_std_bpm"] / cfg.hr_std_tolerance, 0, 1)),
    }
    wsum = sum(cfg.weights.values())
    score = sum(cfg.weights[k] * sub[k] for k in cfg.weights) / wsum

    if score >= cfg.real_threshold:
        verdict = VERDICT_REAL
        reasons.append(f"Forehead and cheeks share a consistent pulse near {bpm:.0f} BPM.")
    elif score <= cfg.fake_threshold:
        verdict = VERDICT_FAKE
        reasons.append("No consistent heartbeat across facial regions; signals look like noise or disagree.")
    else:
        verdict = VERDICT_REVIEW
        reasons.append("Evidence is mixed; route to a human reviewer.")
    # Explain the weakest evidence so a reviewer knows where to look.
    weak = [k for k, v in sorted(sub.items(), key=lambda kv: kv[1]) if v < 0.5][:2]
    labels = {
        "snr": "weak pulse peak (low SNR)",
        "agreement": "regions report different heart rates",
        "correlation": "region signals are uncorrelated",
        "coherence": "low coherence at the heart-rate frequency",
        "stability": "heart rate jumps around over time",
    }
    reasons += [f"Weak evidence: {labels[k]}." for k in weak]

    return AnalysisReport(
        verdict=verdict,
        score=float(score),
        bpm=float(bpm),
        reasons=reasons,
        rois=rois,
        features=features,
        sub_scores=sub,
        elapsed_s=time.perf_counter() - t0,
        signals=signals,
        hr_trace=(hr_t, hr_b),
        preview_frame=draw_rois(preview, preview_box) if preview is not None else None,
        **base,
    )


def analyze_video(path: str | Path, config: AnalysisConfig | None = None) -> AnalysisReport:
    cfg = config or AnalysisConfig()
    path = Path(path)
    if not path.exists():
        raise VideoError(f"File not found: {path}")
    info = probe(path)
    return analyze_frames(iter_frames(path, max_seconds=cfg.max_seconds), info.fps, cfg)
