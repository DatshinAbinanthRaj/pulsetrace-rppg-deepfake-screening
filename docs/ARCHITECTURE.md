# Architecture

```
video file / webcam / synthetic clip
        │
        ▼
┌──────────────────┐   video.py   decode, downscale to ≤640 px wide, cap at N seconds
│  Frame source    │
└────────┬─────────┘
         ▼
┌──────────────────┐   face.py    Haar cascade (default → alt2+equalised fallback),
│  Face tracker    │              re-detect every 5 frames, exponential box smoothing
└────────┬─────────┘
         ▼
┌──────────────────┐   face.py    forehead / left cheek / right cheek boxes,
│  Skin ROIs       │              YCrCb skin mask drops hair, brows, glasses
└────────┬─────────┘
         ▼  mean (R,G,B) per ROI per frame  →  3 traces of shape (T, 3)
┌──────────────────┐   dsp.py     green | POS (default) | CHROM projection,
│  rPPG signal     │              linear detrend, Butterworth 0.7–4 Hz band-pass
└────────┬─────────┘
         ▼
┌──────────────────┐   dsp.py     zero-padded FFT → peak BPM + SNR per ROI,
│  Features        │              pairwise correlation, band coherence at the HR peak,
└────────┬─────────┘              BPM agreement across ROIs, BPM stability over time
         ▼
┌──────────────────┐ analysis.py  weighted 0–1 authenticity score
│  Score + verdict │              ≥0.60 LIKELY_REAL · ≤0.40 LIKELY_FAKE · between: NEEDS_REVIEW
└────────┬─────────┘              quality gates → INCONCLUSIVE (short clip, low fps, no face)
         ▼
  CLI (cli.py) · Web UI (app/streamlit_app.py) · Live webcam (live.py) · Evaluation (evaluate.py)
```

## Design decisions

| Decision | Why |
|---|---|
| POS as default, green kept as an option | The video demonstrates the green channel. POS (Wang et al., 2017) cancels brightness changes common to all channels, so lighting flicker inside the heart-rate band does not masquerade as a pulse. `tests/test_dsp.py::test_pos_cancels_common_intensity_change` shows raw green locking onto a 54 BPM flicker while POS finds the 75 BPM pulse. |
| Cross-region consistency instead of "is there a pulse?" | A single noisy region can show a spurious spectral peak. Requiring forehead and both cheeks to agree on frequency, phase (correlation) and coherence is the core idea in FakeCatcher (Ciftci, Demir & Yin, TPAMI 2020) and is far harder to satisfy by accident. |
| Three-way verdict with an abstain band | Public testing of FakeCatcher (BBC, 2023) found real clips flagged as fake, worse on low-resolution video. A screening tool should defer uncertain clips to a person rather than guess. |
| Quality gates before scoring | Under ~6 s of video, under 10 fps, or no stable face, the spectrum cannot resolve a heart rate. Reporting INCONCLUSIVE is more honest than a low score. |
| Transparent hand-weighted score, not a trained classifier | No licensed deepfake dataset is bundled. A small set of interpretable features with documented weights can be audited and re-calibrated (`AnalysisConfig`) once real labelled data exists. |
| Haar cascades (bundled with OpenCV) | No model download, runs offline on CPU. Trade-off: weaker on profile faces and heavy occlusion than DNN detectors. |
| Procedural synthetic faces for tests | Ground truth is known exactly, and no real person's likeness is stored or published. |

## Module map

| Module | Responsibility |
|---|---|
| `pulsetrace/video.py` | probe, frame iterator, writer |
| `pulsetrace/face.py` | `FaceTracker`, ROI layout, skin mask, per-ROI mean colour |
| `pulsetrace/dsp.py` | green / POS / CHROM, band-pass, spectrum, BPM + SNR, coherence |
| `pulsetrace/analysis.py` | orchestration, features, score, verdict, `AnalysisReport` |
| `pulsetrace/synthetic.py` | labelled synthetic clips and benchmark generator |
| `pulsetrace/evaluate.py` | manifest evaluation, metrics, review queue, stress sweep |
| `pulsetrace/viz.py` | signal, spectrum, heart-rate plots |
| `pulsetrace/cli.py` | `pulsetrace analyze / synth / evaluate / stress / live` |
| `pulsetrace/live.py` | webcam overlay (local machine only) |
| `app/streamlit_app.py` | web UI with human-review logging |

## Privacy

Uploaded videos are written to a temporary file, analysed, and deleted in a `finally` block. The review log stores a 16-character SHA-256 prefix of the video, the model verdict and the reviewer's label, never frames or file names. Nothing is sent over the network.
