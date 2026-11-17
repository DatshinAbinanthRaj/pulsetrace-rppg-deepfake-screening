"""Command-line interface: pulsetrace {analyze,synth,evaluate,stress,live}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pulsetrace import __version__
from pulsetrace.analysis import AnalysisConfig, analyze_video
from pulsetrace.video import VideoError


def _config(args) -> AnalysisConfig:
    return AnalysisConfig(method=args.method, max_seconds=args.max_seconds)


def cmd_analyze(args) -> int:
    try:
        rep = analyze_video(args.video, _config(args))
    except VideoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    d = rep.to_dict()
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print(f"Verdict : {rep.verdict}")
        print(f"Score   : {rep.score:.2f}" if rep.score is not None else "Score   : n/a")
        print(f"Pulse   : {rep.bpm:.0f} BPM" if rep.bpm else "Pulse   : n/a")
        print(f"Clip    : {rep.duration_s:.1f}s @ {rep.fps:.1f} fps, face found {rep.face_detection_rate:.0%}")
        for r in rep.rois:
            print(f"  {r.name:<12} bpm={r.bpm:6.1f}  snr={r.snr_db:6.2f} dB")
        for reason in rep.reasons:
            print(f"  - {reason}")
    if args.out:
        Path(args.out).write_text(json.dumps(d, indent=2))
    if args.plots:
        from pulsetrace import viz

        outdir = Path(args.plots)
        outdir.mkdir(parents=True, exist_ok=True)
        if rep.signals:
            viz.signals_figure(rep).savefig(outdir / "signals.png", dpi=120)
            viz.spectrum_figure(rep).savefig(outdir / "spectrum.png", dpi=120)
            viz.hr_trace_figure(rep).savefig(outdir / "hr_trace.png", dpi=120)
        if rep.preview_frame is not None:
            import cv2

            cv2.imwrite(str(outdir / "rois.png"), rep.preview_frame)
    return 0


def cmd_synth(args) -> int:
    from pulsetrace.synthetic import generate_dataset

    m = generate_dataset(args.out, per_kind=args.per_kind, seconds=args.seconds, seed=args.seed)
    print(f"Wrote benchmark manifest: {m}")
    return 0


def cmd_evaluate(args) -> int:
    from pulsetrace.evaluate import evaluate_manifest

    def progress(i, n, r):
        print(f"[{i}/{n}] {r.path}: {r.verdict} score={r.score if r.score is None else round(r.score, 2)}", file=sys.stderr)

    metrics = evaluate_manifest(args.manifest, args.out, _config(args), progress=progress)
    print(json.dumps({k: v for k, v in metrics.items()}, indent=2, default=str))
    print(f"Report: {Path(args.out) / 'report.md'}")
    return 0


def cmd_stress(args) -> int:
    from pulsetrace.evaluate import stress_sweep

    rows = stress_sweep(args.out, clips_per_cell=args.clips, config=_config(args))
    print((Path(args.out) / "stress_sweep.md").read_text())
    return 0 if rows else 1


def cmd_live(args) -> int:  # pragma: no cover - needs a camera
    from pulsetrace.live import run_live

    return run_live(camera=args.camera, method=args.method)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pulsetrace", description="rPPG (blood-flow) deepfake screening")
    p.add_argument("--version", action="version", version=f"pulsetrace {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--method", choices=["green", "pos", "chrom"], default="pos")
        sp.add_argument("--max-seconds", type=float, default=30.0)

    a = sub.add_parser("analyze", help="analyse one video")
    a.add_argument("video")
    a.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    a.add_argument("--out", help="write JSON report here")
    a.add_argument("--plots", help="write signal/spectrum/ROI images to this folder")
    common(a)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("synth", help="generate a labelled synthetic benchmark")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--per-kind", type=int, default=5)
    s.add_argument("--seconds", type=float, default=12.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    e = sub.add_parser("evaluate", help="evaluate on a labelled manifest CSV")
    e.add_argument("manifest")
    e.add_argument("--out", default="results/eval")
    common(e)
    e.set_defaults(func=cmd_evaluate)

    st = sub.add_parser("stress", help="robustness sweep on synthetic clips")
    st.add_argument("--out", default="results/stress")
    st.add_argument("--clips", type=int, default=4)
    common(st)
    st.set_defaults(func=cmd_stress)

    lv = sub.add_parser("live", help="live webcam heart-rate + verdict overlay (local machine)")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--method", choices=["green", "pos", "chrom"], default="pos")
    lv.set_defaults(func=cmd_live)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
