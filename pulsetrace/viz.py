"""Matplotlib figures for reports and the web UI."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pulsetrace import dsp  # noqa: E402
from pulsetrace.analysis import AnalysisReport  # noqa: E402

COLOURS = {"forehead": "#f5b700", "left_cheek": "#e05aa8", "right_cheek": "#8c6cf0", "combined": "#2fbf71"}


def signals_figure(rep: AnalysisReport):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for name, sig in rep.signals.items():
        t = np.arange(len(sig)) / rep.fps
        ax.plot(
            t,
            sig + (0 if name == "combined" else 0),
            lw=1.6 if name == "combined" else 0.9,
            color=COLOURS.get(name),
            label=name,
            alpha=1 if name == "combined" else 0.75,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised pulse signal")
    ax.set_title("Filtered skin signals per region (0.7-4 Hz)")
    ax.legend(loc="upper right", ncols=4, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def spectrum_figure(rep: AnalysisReport):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for name, sig in rep.signals.items():
        f, p = dsp.spectrum(sig, rep.fps)
        sel = (f >= 0.5) & (f <= 4.5)
        ax.plot(f[sel] * 60, p[sel] / p[sel].max(), color=COLOURS.get(name), label=name, lw=1.6 if name == "combined" else 0.9)
    ax.axvspan(dsp.HR_LOW_HZ * 60, dsp.HR_HIGH_HZ * 60, color="#2fbf71", alpha=0.06, label="human HR band")
    if rep.bpm:
        ax.axvline(rep.bpm, color="#2fbf71", ls="--", lw=1)
    ax.set_xlabel("beats per minute")
    ax.set_ylabel("relative power")
    ax.set_title("Frequency content: one shared peak = consistent heartbeat")
    ax.legend(loc="upper right", ncols=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def hr_trace_figure(rep: AnalysisReport):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    if rep.hr_trace is not None:
        t, b = rep.hr_trace
        ax.plot(t, b, marker="o", ms=3, color="#e0475b")
    ax.set_ylim(30, 250)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("BPM (6 s window)")
    ax.set_title("Heart-rate estimate over time")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
