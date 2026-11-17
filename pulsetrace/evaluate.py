"""Evaluation harness: run the detector over a labelled manifest and report quality metrics.

Manifest CSV columns: path,label[,kind,gt_bpm]   (label is "real" or "fake")

Outputs (in out_dir):
  results.csv      one row per clip (verdict, score, features)
  metrics.json     headline metrics
  report.md        human-readable summary with confusion matrix
  review_queue.csv clips a human should look at (NEEDS_REVIEW, INCONCLUSIVE, misclassified)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pulsetrace.analysis import (
    VERDICT_FAKE,
    VERDICT_INCONCLUSIVE,
    VERDICT_REAL,
    VERDICT_REVIEW,
    AnalysisConfig,
    analyze_frames,
    analyze_video,
)


@dataclass
class ClipResult:
    path: str
    label: str
    kind: str
    gt_bpm: float | None
    verdict: str
    score: float | None
    bpm: float | None
    reasons: str
    error: str = ""


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Mann-Whitney AUC. labels: 1 = real (positive = authentic). Ties count half."""
    pos = [s for y, s in zip(labels, scores, strict=True) if y == 1]
    neg = [s for y, s in zip(labels, scores, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def compute_metrics(results: list[ClipResult]) -> dict:
    """'Positive' = flagged as fake. Decided clips only for precision/recall; abstentions reported separately."""
    n = len(results)
    decided = [r for r in results if r.verdict in (VERDICT_REAL, VERDICT_FAKE)]
    tp = sum(r.label == "fake" and r.verdict == VERDICT_FAKE for r in decided)
    fp = sum(r.label == "real" and r.verdict == VERDICT_FAKE for r in decided)
    tn = sum(r.label == "real" and r.verdict == VERDICT_REAL for r in decided)
    fn = sum(r.label == "fake" and r.verdict == VERDICT_REAL for r in decided)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not math.isnan(prec + rec) else float("nan")
    scored = [r for r in results if r.score is not None]
    auc = roc_auc([1 if r.label == "real" else 0 for r in scored], [r.score for r in scored])
    bpm_err = [abs(r.bpm - r.gt_bpm) for r in results if r.gt_bpm and r.bpm and r.verdict != VERDICT_INCONCLUSIVE]
    return {
        "clips": n,
        "decided": len(decided),
        "abstained_review": sum(r.verdict == VERDICT_REVIEW for r in results),
        "abstained_inconclusive": sum(r.verdict == VERDICT_INCONCLUSIVE for r in results),
        "errors": sum(bool(r.error) for r in results),
        "coverage": len(decided) / n if n else float("nan"),
        "accuracy_on_decided": (tp + tn) / len(decided) if decided else float("nan"),
        "fake_precision": prec,
        "fake_recall": rec,
        "fake_f1": f1,
        "real_flagged_as_fake_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "roc_auc_score": auc,
        "bpm_mae": float(np.mean(bpm_err)) if bpm_err else float("nan"),
        "confusion": {"tp_fake": tp, "fp_real_called_fake": fp, "tn_real": tn, "fn_fake_called_real": fn},
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def write_outputs(results: list[ClipResult], metrics: dict, out_dir: Path, title: str = "PulseTrace evaluation") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(ClipResult.__dataclass_fields__)
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)
    clean = json.loads(json.dumps(metrics, default=str).replace("NaN", "null"))
    (out_dir / "metrics.json").write_text(json.dumps(clean, indent=2))

    queue = [
        r
        for r in results
        if r.verdict in (VERDICT_REVIEW, VERDICT_INCONCLUSIVE)
        or r.error
        or (r.label == "real" and r.verdict == VERDICT_FAKE)
        or (r.label == "fake" and r.verdict == VERDICT_REAL)
    ]
    with (out_dir / "review_queue.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["path", "label", "verdict", "score", "why_queued", "reasons", "reviewer_label", "reviewer_notes"]
        )
        w.writeheader()
        for r in queue:
            why = (
                "error" if r.error else ("abstained" if r.verdict in (VERDICT_REVIEW, VERDICT_INCONCLUSIVE) else "misclassified")
            )
            w.writerow(
                {
                    "path": r.path,
                    "label": r.label,
                    "verdict": r.verdict,
                    "score": r.score,
                    "why_queued": why,
                    "reasons": r.reasons or r.error,
                    "reviewer_label": "",
                    "reviewer_notes": "",
                }
            )

    c = metrics["confusion"]
    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {_fmt(v)} |" for k, v in metrics.items() if k != "confusion"],
        "",
        "## Confusion matrix (decided clips; positive = fake)",
        "",
        "| | predicted FAKE | predicted REAL |",
        "|---|---|---|",
        f"| actual FAKE | {c['tp_fake']} | {c['fn_fake_called_real']} |",
        f"| actual REAL | {c['fp_real_called_fake']} | {c['tn_real']} |",
        "",
        f"Clips queued for human review: {len(queue)} (see review_queue.csv)",
        "",
    ]
    kinds = sorted({r.kind for r in results if r.kind})
    if kinds:
        lines += [
            "## By clip kind",
            "",
            "| kind | n | LIKELY_REAL | LIKELY_FAKE | NEEDS_REVIEW | INCONCLUSIVE | mean score |",
            "|---|---|---|---|---|---|---|",
        ]
        for k in kinds:
            rs = [r for r in results if r.kind == k]
            sc = [r.score for r in rs if r.score is not None]
            cnt = {v: sum(r.verdict == v for r in rs) for v in (VERDICT_REAL, VERDICT_FAKE, VERDICT_REVIEW, VERDICT_INCONCLUSIVE)}
            lines.append(
                f"| {k} | {len(rs)} | {cnt[VERDICT_REAL]} | {cnt[VERDICT_FAKE]} | {cnt[VERDICT_REVIEW]} | {cnt[VERDICT_INCONCLUSIVE]} | {np.mean(sc) if sc else float('nan'):.3f} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def evaluate_manifest(manifest: str | Path, out_dir: str | Path, config: AnalysisConfig | None = None, progress=None) -> dict:
    manifest = Path(manifest)
    base = manifest.parent
    results: list[ClipResult] = []
    with manifest.open() as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows):
        label = row["label"].strip().lower()
        if label not in ("real", "fake"):
            raise ValueError(f"Row {i + 2}: label must be 'real' or 'fake', got {row['label']!r}")
        p = Path(row["path"])
        p = p if p.is_absolute() else base / p
        gt = float(row["gt_bpm"]) if row.get("gt_bpm") else None
        try:
            rep = analyze_video(p, config)
            results.append(
                ClipResult(row["path"], label, row.get("kind", ""), gt, rep.verdict, rep.score, rep.bpm, " | ".join(rep.reasons))
            )
        except Exception as exc:  # keep going; failures are reported, not hidden
            results.append(
                ClipResult(row["path"], label, row.get("kind", ""), gt, VERDICT_INCONCLUSIVE, None, None, "", error=str(exc))
            )
        if progress:
            progress(i + 1, len(rows), results[-1])
    metrics = compute_metrics(results)
    write_outputs(results, metrics, Path(out_dir))
    return metrics


def stress_sweep(
    out_dir: str | Path,
    amps=(0.006, 0.003, 0.0015, 0.00075),
    noises=(1.5, 4.0),
    clips_per_cell: int = 4,
    seconds: float = 12.0,
    config: AnalysisConfig | None = None,
    seed: int = 1000,
) -> list[dict]:
    """Robustness sweep on synthetic clips: weaker pulse and more sensor noise (a stand-in for
    low-quality / compressed video). Returns one metrics dict per (amp, noise) cell."""
    from pulsetrace.synthetic import KINDS, make_clip

    out = Path(out_dir)
    rows = []
    k = seed
    for amp in amps:
        for noise in noises:
            results = []
            for kind in KINDS:
                reps = clips_per_cell * (3 if kind == "real" else 1)
                for _ in range(reps):
                    clip = make_clip(kind, seconds=seconds, seed=k, pulse_amp=amp, noise_sigma=noise)
                    k += 1
                    rep = analyze_frames(clip.frames, clip.fps, config)
                    results.append(
                        ClipResult(f"synthetic:{kind}:{k}", clip.label, kind, clip.bpm, rep.verdict, rep.score, rep.bpm, "")
                    )
            m = compute_metrics(results)
            rows.append({"pulse_amp": amp, "noise_sigma": noise, **{k2: v for k2, v in m.items() if k2 != "confusion"}})
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out / "stress_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Stress sweep (synthetic)",
        "",
        "Pulse amplitude is the relative skin-colour change per heartbeat; noise is per-pixel sensor noise (8-bit units).",
        "",
        "| pulse_amp | noise | coverage | accuracy (decided) | real flagged fake | fake recall | ROC-AUC | BPM MAE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['pulse_amp']} | {r['noise_sigma']} | {_fmt(r['coverage'])} | {_fmt(r['accuracy_on_decided'])} | {_fmt(r['real_flagged_as_fake_rate'])} | {_fmt(r['fake_recall'])} | {_fmt(r['roc_auc_score'])} | {_fmt(r['bpm_mae'])} |"
        )
    (out / "stress_sweep.md").write_text("\n".join(md) + "\n")
    return rows
