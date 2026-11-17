# Video → implementation feature matrix

Source: a ~2 min explainer reel on rPPG deepfake detection (Intel FakeCatcher). The reel is an animated explanation; it shows the idea and one code snippet (`import cv2`, `face = detect(frame)`) but no full implementation. Everything below was rebuilt from the described method and the cited paper.

| # | Feature shown in the video | Implemented | Where | How verified | Matches demo? | Notes / deviations |
|---|---|---|---|---|---|---|
| 1 | Heartbeat changes skin colour; read the pulse from video (rPPG) | Yes | `dsp.py` | `test_dsp.py` (BPM recovered within 1.5 BPM for 48–180 BPM) | Yes | |
| 2 | Step 1: find the face | Yes | `face.py::FaceTracker` | `test_face.py`, all analysis tests | Yes | Haar cascade + fallback cascade, smoothed box |
| 3 | Forehead and cheek regions | Yes | `face.py::ROI_LAYOUT`, `roi_means` | `test_roi_layout_inside_face_and_ordered` | Yes | Added skin mask to drop hair/brows |
| 4 | ~30 frames per second, each frame = pixels with R, G, B | Yes | `video.py`, `face.py` | file round-trip test at 30 fps | Yes | Any fps ≥10 accepted |
| 5 | Green channel shows the pulse best; track green frame by frame | Yes | `dsp.green_signal`, `--method green` | `test_real_clip_is_real_with_correct_bpm[green]` | Yes | POS is the default (more robust, see #7) |
| 6 | Connect values into a (noisy) signal | Yes | `analysis.py`, `viz.signals_figure` | UI screenshot, CLI `--plots` | Yes | |
| 7 | Lighting changes and head movement add noise | Yes (mitigated) | POS/CHROM, detrend, box smoothing | `test_pos_cancels_common_intensity_change`; synthetic clips include lighting drift + head motion | Yes | Improvement over the demo's raw-green approach |
| 8 | Frequencies in the signal; keep human heart-rate range, filter out the rest | Yes | `dsp.bandpass` (0.7–4 Hz = 42–240 BPM), `dsp.spectrum` | `test_bandpass_removes_out_of_band` | Yes | |
| 9 | What's left: heartbeat → BPM readout | Yes | `dsp.estimate_bpm`, `bpm_over_time` | BPM MAE 0.27 on synthetic benchmark | Yes | Also shows BPM over time |
| 10 | Real face: forehead and cheek signals match | Yes | correlation, coherence, BPM agreement features | real clips score 0.85–0.99 | Yes | |
| 11 | Deepfake: signals don't match → the giveaway | Yes | same features → LIKELY_FAKE | 3 fake types all flagged on benchmark | Yes, on synthetic data | Not validated on real deepfakes (no licensed dataset available offline) |
| 12 | Live detector feed showing a verdict | Partly | `pulsetrace live` (webcam), web UI for files | Code path written; **not run** (no camera in build environment) | Untested | Web UI verified with browser screenshots |
| 13 | "Not every deepfake is caught; some real clips get flagged" (BBC test) | Yes, as a design response | NEEDS_REVIEW band, INCONCLUSIVE gates, review queue, reviewer log | `test_compute_metrics_counts_and_abstention`, UI review-form test | n/a | Reported metric: real-flagged-as-fake rate |
| 14 | Intel's system also uses eye movement / learned classifier on PPG maps | No | n/a | n/a | n/a | Out of scope; would need a labelled deepfake dataset to train. Listed under future work |

## Added beyond the video

- POS and CHROM rPPG algorithms with a method switch
- Three-way verdict with an abstain band and plain-language reasons
- Evaluation harness: precision/recall/F1, ROC-AUC, coverage, real-flagged-as-fake rate, BPM MAE, review queue CSV
- Robustness stress sweep (weaker pulse, more noise)
- Synthetic benchmark generator with known ground truth, no real faces
- Web UI with human-review logging; JSON report export; CLI; CI workflow
