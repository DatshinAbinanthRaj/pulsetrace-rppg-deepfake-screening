import numpy as np
import pytest

from pulsetrace import dsp

FPS = 30.0


def sine(bpm, seconds=12, fps=FPS, noise=0.0, seed=0):
    t = np.arange(int(seconds * fps)) / fps
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * bpm / 60 * t) + rng.normal(0, noise, t.size)


@pytest.mark.parametrize("bpm", [48, 60, 72, 95, 130, 180])
def test_estimate_bpm_recovers_frequency(bpm):
    est, snr = dsp.estimate_bpm(sine(bpm, noise=0.3), FPS)
    assert abs(est - bpm) < 1.5
    assert snr > 3


def test_snr_is_low_for_noise():
    rng = np.random.default_rng(1)
    _, snr = dsp.estimate_bpm(dsp.bandpass(rng.normal(size=360), FPS), FPS)
    assert snr < 0


def test_bandpass_removes_out_of_band():
    t = np.arange(360) / FPS
    slow = np.sin(2 * np.pi * 0.1 * t)  # lighting drift
    fast = np.sin(2 * np.pi * 8 * t)  # flicker
    pulse = 0.2 * np.sin(2 * np.pi * 1.2 * t)
    out = dsp.bandpass(slow + fast + pulse, FPS)
    est, _ = dsp.estimate_bpm(out, FPS)
    assert abs(est - 72) < 1.5
    assert out.std() < 0.3


def test_bandpass_clamps_high_cut_for_low_fps():
    x = sine(60, fps=6.0)
    assert np.isfinite(dsp.bandpass(x, 6.0)).all()


def test_pos_cancels_common_intensity_change():
    """A brightness change that hits R, G, B equally must not dominate POS output."""
    t = np.arange(360) / FPS
    pulse = 0.004 * np.sin(2 * np.pi * 1.25 * t)
    light = 1 + 0.05 * np.sin(2 * np.pi * 0.9 * t)  # in-band lighting flicker at 54 BPM
    base = np.array([180.0, 140.0, 110.0])
    rgb = base * light[:, None] * (1 + pulse[:, None] * np.array([0.33, 0.77, 0.53]))
    pos_bpm, _ = dsp.estimate_bpm(dsp.bandpass(dsp.pos_signal(rgb, FPS), FPS), FPS)
    green_bpm, _ = dsp.estimate_bpm(dsp.bandpass(dsp.green_signal(rgb), FPS), FPS)
    assert abs(pos_bpm - 75) < 2  # POS finds the pulse
    assert abs(green_bpm - 54) < 2  # raw green is fooled by the flicker


def test_chrom_runs_and_finds_pulse():
    t = np.arange(360) / FPS
    pulse = 0.004 * np.sin(2 * np.pi * 1.5 * t)
    rgb = np.array([170.0, 130.0, 100.0]) * (1 + pulse[:, None] * np.array([0.33, 0.77, 0.53]))
    est, _ = dsp.estimate_bpm(dsp.chrom_signal(rgb, FPS), FPS)
    assert abs(est - 90) < 2


def test_bpm_over_time_shape_and_short_input():
    t, b = dsp.bpm_over_time(sine(80), FPS)
    assert len(t) == len(b) > 3
    assert np.nanmax(np.abs(b - 80)) < 3
    t2, b2 = dsp.bpm_over_time(sine(80, seconds=3), FPS)
    assert len(b2) == 1


def test_band_coherence_high_for_shared_signal_low_for_independent():
    a = sine(70, noise=0.2, seed=1)
    b = sine(70, noise=0.2, seed=2)
    rng = np.random.default_rng(3)
    c = dsp.bandpass(rng.normal(size=a.size), FPS)
    assert dsp.band_coherence(a, b, FPS, 70 / 60) > 0.8
    assert dsp.band_coherence(a, c, FPS, 70 / 60) < 0.5
