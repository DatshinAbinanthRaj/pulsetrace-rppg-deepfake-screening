"""Harder synthetic check: weak pulse + large head motion. Usage: python examples/motion_stress.py"""

from pulsetrace.analysis import analyze_frames
from pulsetrace.evaluate import ClipResult, compute_metrics
from pulsetrace.synthetic import KINDS, make_clip

print("| head motion (px) | coverage | accuracy (decided) | real flagged fake | fake recall | ROC-AUC | verdicts |")
print("|---|---|---|---|---|---|---|")
for motion in (8, 16, 30):
    res, k = [], 7000 + motion * 100
    for kind in KINDS:
        for _ in range(3 if kind == "real" else 1):
            c = make_clip(kind, seed=k, pulse_amp=0.0015, noise_sigma=4.0, motion_px=motion)
            k += 1
            r = analyze_frames(c.frames, c.fps)
            res.append(ClipResult(kind, c.label, kind, c.bpm, r.verdict, r.score, r.bpm, ""))
    m = compute_metrics(res)
    v = ", ".join(f"{r.kind}:{r.verdict.split('_')[-1][:4]}" for r in res)
    f = lambda x: "n/a" if x != x else f"{x:.2f}"  # noqa: E731
    print(f"| {motion} | {f(m['coverage'])} | {f(m['accuracy_on_decided'])} | {f(m['real_flagged_as_fake_rate'])} | {f(m['fake_recall'])} | {f(m['roc_auc_score'])} | {v} |")
