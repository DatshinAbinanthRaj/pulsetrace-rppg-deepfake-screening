# Test results

Run on 17 Nov 2026, Linux, Python 3.13, OpenCV 4.14, SciPy 1.18, Streamlit 1.67.

| Check | Command | Result |
|---|---|---|
| Dependency install | `pip install -e ".[dev]"` + `pip install streamlit` | OK |
| Lint | `ruff check .` | All checks passed |
| Format | `ruff format --check .` | clean |
| Unit + integration + end-to-end tests | `pytest -q` | **42 passed**, 0 failed (≈2 min) |
| Security: static analysis | `bandit -r pulsetrace app -s B101` | no issues |
| Security: dependency CVEs | `pip-audit -r requirements.txt` | no known vulnerabilities |
| Security: secrets | grep for key/secret/token/password assignments | none (project needs no credentials) |
| UI: scripted flow | `tests/test_app.py` (Streamlit AppTest: generate → analyse → verdict → save review) | 2 passed |
| UI: real browser | Streamlit server + headless Chrome screenshots | renders, verdicts correct ([real](assets/ui_real.png), [fake](assets/ui_fake.png)) |
| Performance | 12 s clip at 640×480, 30 fps, in the build sandbox | ≈4 s analysis per clip (≈3× faster than real time) |
| CLI | `pulsetrace analyze / synth / evaluate / stress` | all ran; outputs in `docs/results/` |

## Test inventory

| File | Covers |
|---|---|
| `tests/test_dsp.py` (13) | BPM recovery 48–180 BPM, SNR on noise, band-pass, low-fps clamp, POS vs green under lighting flicker, CHROM, BPM-over-time, coherence |
| `tests/test_face.py` (6) | detection, blank frames, losing the face, ROI geometry, skin mask, RGB order, off-frame ROIs |
| `tests/test_analysis.py` (15) | real → LIKELY_REAL with correct BPM (3 methods), 3 fake types → LIKELY_FAKE, ranking, short clip / no face / low fps → INCONCLUSIVE, bad method, JSON safety, file round-trip, missing and corrupt files |
| `tests/test_evaluate_cli.py` (6) | AUC, confusion counts and abstention, end-to-end synth→evaluate, errors recorded not raised, CLI JSON + plots, CLI exit code |
| `tests/test_app.py` (2) | web UI flow for a real and a fake clip, review log written |

## Bugs found and fixed during testing

1. **Skin mask accepted dark hair** (`test_skin_mask_accepts_skin_rejects_dark_hair`). Luma threshold raised from 40 to 60.
2. **Face lost after video encoding.** After MJPG encoding, face detection fell to 48 % and the clip was marked INCONCLUSIVE. Histogram equalisation was the cause. Fix: detect on the raw greyscale frame first, then fall back to a second cascade on the equalised frame. Detection is back to 100 % on the benchmark.
3. **UI widget lookup in tests** was fragile (index-based). Added stable widget keys.
4. **Evidence chart unreadable** when sub-scores were 0. Replaced it with a 0–1 progress table.

## Not tested

- `pulsetrace live` (webcam): no camera in the build environment. The code runs `analyze_frames` on a rolling 10 s buffer, so the analysis itself is covered, but the capture and overlay loop has not been run.
- Real deepfake datasets: see [EVALUATION.md](EVALUATION.md).
