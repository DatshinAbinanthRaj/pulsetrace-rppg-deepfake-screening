"""Signal processing: rPPG projections, filtering and heart-rate estimation."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

# Human heart-rate band used throughout: 42-240 BPM.
HR_LOW_HZ = 0.7
HR_HIGH_HZ = 4.0


def detrend(x: np.ndarray) -> np.ndarray:
    return sps.detrend(np.asarray(x, dtype=np.float64), type="linear")


def bandpass(x: np.ndarray, fps: float, low: float = HR_LOW_HZ, high: float = HR_HIGH_HZ, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass ('filter out the rest')."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fps / 2.0
    high = min(high, 0.95 * nyq)
    sos = sps.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    padlen = min(len(x) - 1, 3 * (2 * order + 1))
    return sps.sosfiltfilt(sos, x, padlen=max(padlen, 0))


def green_signal(rgb: np.ndarray) -> np.ndarray:
    """Classic green-channel rPPG (the method shown in the video). rgb: (T, 3)."""
    g = rgb[:, 1]
    return g / (g.mean() + 1e-9) - 1.0


def pos_signal(rgb: np.ndarray, fps: float, window_s: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin projection (Wang et al., IEEE TBME 2017).

    More robust than raw green to lighting changes and small motion because it
    cancels intensity variation that is common to all three channels.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    win = max(int(round(window_s * fps)), 2)
    h = np.zeros(n)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(0, max(n - win + 1, 1)):
        seg = rgb[start : start + win]
        mean = seg.mean(axis=0)
        if np.any(mean <= 0):
            continue
        cn = seg / mean
        s = cn @ proj.T  # (win, 2)
        std2 = s[:, 1].std()
        alpha = s[:, 0].std() / std2 if std2 > 1e-12 else 0.0
        p = s[:, 0] + alpha * s[:, 1]
        h[start : start + len(seg)] += p - p.mean()
    return h


def chrom_signal(rgb: np.ndarray, fps: float) -> np.ndarray:
    """CHROM chrominance method (de Haan & Jeanne, IEEE TBME 2013)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    cn = rgb / (rgb.mean(axis=0) + 1e-9)
    x = 3 * cn[:, 0] - 2 * cn[:, 1]
    y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf, yf = bandpass(x, fps), bandpass(y, fps)
    alpha = xf.std() / (yf.std() + 1e-12)
    return xf - alpha * yf


METHODS = {"green": lambda rgb, fps: green_signal(rgb), "pos": pos_signal, "chrom": chrom_signal}


def spectrum(x: np.ndarray, fps: float, nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed, zero-padded power spectrum. Returns (freqs_hz, power)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    win = np.hanning(len(x))
    n = max(nfft, len(x))
    p = np.abs(np.fft.rfft(x * win, n=n)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    return f, p


def estimate_bpm(x: np.ndarray, fps: float) -> tuple[float, float]:
    """Return (bpm, snr_db). SNR follows de Haan: power at peak and first harmonic vs rest of band."""
    f, p = spectrum(x, fps)
    band = (f >= HR_LOW_HZ) & (f <= HR_HIGH_HZ)
    if not band.any() or p[band].sum() <= 0:
        return float("nan"), float("-inf")
    fb, pb = f[band], p[band]
    peak = fb[np.argmax(pb)]
    sig = (np.abs(fb - peak) <= 0.1) | (np.abs(fb - 2 * peak) <= 0.2)
    s, nz = pb[sig].sum(), pb[~sig].sum()
    snr = 10 * np.log10(s / nz) if nz > 0 else float("inf")
    return float(peak * 60.0), float(snr)


def bpm_over_time(x: np.ndarray, fps: float, window_s: float = 6.0, step_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window heart-rate trace. Returns (time_s, bpm)."""
    win, step = int(window_s * fps), max(int(step_s * fps), 1)
    if len(x) < win:
        bpm, _ = estimate_bpm(x, fps)
        return np.array([len(x) / fps / 2]), np.array([bpm])
    t, b = [], []
    for s in range(0, len(x) - win + 1, step):
        bpm, _ = estimate_bpm(x[s : s + win], fps)
        t.append((s + win / 2) / fps)
        b.append(bpm)
    return np.array(t), np.array(b)


def band_coherence(a: np.ndarray, b: np.ndarray, fps: float, center_hz: float, half_width_hz: float = 0.15) -> float:
    """Magnitude-squared coherence of two signals averaged around the heart-rate peak."""
    nper = int(min(len(a), max(64, 4 * fps)))
    f, c = sps.coherence(a, b, fs=fps, nperseg=nper)
    sel = np.abs(f - center_hz) <= max(half_width_hz, f[1] - f[0] if len(f) > 1 else half_width_hz)
    return float(np.nanmean(c[sel])) if sel.any() else float("nan")
