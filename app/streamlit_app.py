"""PulseTrace web UI.  Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulsetrace import viz  # noqa: E402
from pulsetrace.analysis import AnalysisConfig, analyze_frames, analyze_video  # noqa: E402
from pulsetrace.synthetic import KINDS, make_clip  # noqa: E402
from pulsetrace.video import VideoError  # noqa: E402

REVIEW_LOG = Path(os.environ.get("PULSETRACE_REVIEW_LOG", "data/reviews.csv"))
VERDICT_STYLE = {
    "LIKELY_REAL": ("#1f9d55", "Consistent heartbeat found"),
    "LIKELY_FAKE": ("#d64545", "No consistent heartbeat"),
    "NEEDS_REVIEW": ("#d98b00", "Mixed evidence: human review"),
    "INCONCLUSIVE": ("#6b7280", "Cannot judge this clip"),
}

st.set_page_config(page_title="PulseTrace: rPPG deepfake screening", page_icon="🫀", layout="wide")
st.title("🫀 PulseTrace")
st.caption(
    "Deepfake screening from blood-flow signals (remote photoplethysmography). "
    "A screening aid with a human-review band, not a forensic verdict."
)

with st.sidebar:
    st.header("Settings")
    method = st.selectbox(
        "rPPG method", ["pos", "chrom", "green"], help="POS and CHROM resist lighting changes better than raw green."
    )
    max_s = st.slider("Seconds to analyse", 6, 60, 20)
    st.divider()
    st.markdown(
        "**How it works**\n\n1. Find the face, then forehead and both cheeks\n"
        "2. Average skin colour per region, every frame\n3. Keep only 42–240 BPM frequencies\n"
        "4. Real faces: all regions share one pulse. Many fakes: they don't"
    )

src = st.radio("Input", ["Upload a video", "Synthetic demo clip"], horizontal=True, key="src")
cfg = AnalysisConfig(method=method, max_seconds=float(max_s))
report, video_id = None, None

if src == "Upload a video":
    up = st.file_uploader(
        "Video with one frontal face (mp4, mov, avi, webm). Processed locally, not stored.",
        type=["mp4", "mov", "avi", "webm", "mkv"],
    )
    if up is not None and st.button("Analyse", type="primary"):
        data = up.getvalue()
        video_id = hashlib.sha256(data).hexdigest()[:16]
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            with st.spinner("Extracting pulse signals..."):
                report = analyze_video(tmp_path, cfg)
        except VideoError as exc:
            st.error(f"Could not read this video: {exc}")
        finally:
            os.unlink(tmp_path)
else:
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Clip type", KINDS, key="kind", help="real = one shared pulse; others imitate deepfake failure modes")
    seed = c2.number_input("Seed", 0, 9999, 7)
    if st.button("Generate and analyse", type="primary", key="gen"):
        with st.spinner("Rendering synthetic face and analysing..."):
            clip = make_clip(kind, seconds=12, seed=int(seed))
            report = analyze_frames(clip.frames, clip.fps, cfg)
            video_id = f"synthetic-{kind}-{seed}"
        st.info(f"Ground truth: **{clip.label.upper()}** ({kind})" + (f", {clip.bpm:.0f} BPM" if clip.bpm else ""))

if report is not None:
    st.session_state["report"] = report
    st.session_state["video_id"] = video_id

report = st.session_state.get("report")
report_weights = cfg.weights
if report is not None:
    colour, blurb = VERDICT_STYLE[report.verdict]
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{colour};color:white'>"
        f"<div style='font-size:1.6rem;font-weight:700'>{report.verdict.replace('_', ' ')}</div>"
        f"<div>{blurb}</div></div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Authenticity score",
        f"{report.score:.2f}" if report.score is not None else "n/a",
        help="0 = no consistent pulse, 1 = strong shared pulse",
    )
    m2.metric("Estimated pulse", f"{report.bpm:.0f} BPM" if report.bpm else "n/a")
    m3.metric("Face found", f"{report.face_detection_rate:.0%}")
    m4.metric("Analysed", f"{report.duration_s:.1f}s @ {report.fps:.0f} fps")
    for r in report.reasons:
        st.write("• " + r)

    left, right = st.columns([1, 2])
    if report.preview_frame is not None:
        left.image(
            cv2.cvtColor(report.preview_frame, cv2.COLOR_BGR2RGB), caption="Face and skin regions", use_container_width=True
        )
    if report.sub_scores:
        right.subheader("Evidence breakdown")
        right.dataframe(
            [{"evidence": k, "weight": report_weights.get(k, 0), "strength": v} for k, v in report.sub_scores.items()],
            hide_index=True,
            column_config={
                "strength": st.column_config.ProgressColumn("strength (0-1)", min_value=0.0, max_value=1.0, format="%.2f")
            },
        )
        right.dataframe(
            [{"region": r.name, "BPM": round(r.bpm, 1), "SNR dB": round(r.snr_db, 2)} for r in report.rois], hide_index=True
        )
    if report.signals:
        st.pyplot(viz.signals_figure(report))
        st.pyplot(viz.spectrum_figure(report))
        st.pyplot(viz.hr_trace_figure(report))

    st.download_button("Download JSON report", json.dumps(report.to_dict(), indent=2), file_name="pulsetrace_report.json")

    st.divider()
    st.subheader("Human review")
    st.caption("Log your judgement. Reviews build a labelled set for re-calibrating thresholds (see docs/EVALUATION.md).")
    with st.form("review"):
        lab = st.radio("Your label", ["real", "fake", "unsure"], horizontal=True, key="review_label")
        notes = st.text_input("Notes (optional)")
        if st.form_submit_button("Save review"):
            REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
            new = not REVIEW_LOG.exists()
            with REVIEW_LOG.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["timestamp_utc", "video_id", "model_verdict", "model_score", "reviewer_label", "notes"])
                w.writerow(
                    [
                        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        st.session_state.get("video_id"),
                        report.verdict,
                        report.score,
                        lab,
                        notes,
                    ]
                )
            st.success(f"Saved to {REVIEW_LOG}")
