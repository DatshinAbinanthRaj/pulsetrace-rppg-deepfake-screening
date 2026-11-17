import csv
import json

from pulsetrace.cli import main
from pulsetrace.evaluate import ClipResult, compute_metrics, evaluate_manifest, roc_auc
from pulsetrace.synthetic import generate_dataset


def test_roc_auc_basic():
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([1, 0], [0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.1, 0.2]) != roc_auc([1, 1], [0.1, 0.2])  # NaN when one class missing


def test_compute_metrics_counts_and_abstention():
    rs = [
        ClipResult("a", "real", "", 70, "LIKELY_REAL", 0.9, 70, ""),
        ClipResult("b", "real", "", 80, "LIKELY_FAKE", 0.2, 90, ""),
        ClipResult("c", "fake", "", None, "LIKELY_FAKE", 0.1, None, ""),
        ClipResult("d", "fake", "", None, "NEEDS_REVIEW", 0.5, None, ""),
    ]
    m = compute_metrics(rs)
    assert m["confusion"] == {"tp_fake": 1, "fp_real_called_fake": 1, "tn_real": 1, "fn_fake_called_real": 0}
    assert m["coverage"] == 0.75
    assert m["real_flagged_as_fake_rate"] == 0.5
    assert m["abstained_review"] == 1
    assert m["bpm_mae"] == 5.0


def test_end_to_end_synth_evaluate(tmp_path):
    manifest = generate_dataset(tmp_path / "bench", per_kind=1, seconds=8, seed=40)
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 6 and {r["label"] for r in rows} == {"real", "fake"}
    m = evaluate_manifest(manifest, tmp_path / "out")
    assert m["errors"] == 0
    assert m["accuracy_on_decided"] >= 0.8
    for f in ["results.csv", "metrics.json", "report.md", "review_queue.csv"]:
        assert (tmp_path / "out" / f).exists()
    json.loads((tmp_path / "out" / "metrics.json").read_text())


def test_evaluate_records_errors_instead_of_crashing(tmp_path):
    (tmp_path / "m.csv").write_text("path,label\nmissing.mp4,real\n")
    m = evaluate_manifest(tmp_path / "m.csv", tmp_path / "o")
    assert m["errors"] == 1
    q = list(csv.DictReader((tmp_path / "o" / "review_queue.csv").open()))
    assert q[0]["why_queued"] == "error"


def test_cli_analyze_json_and_plots(tmp_path, capsys):
    manifest = generate_dataset(tmp_path / "b", per_kind=1, seconds=8, seed=60)
    clip = manifest.parent / "real_00.avi"
    assert main(["analyze", str(clip), "--json", "--plots", str(tmp_path / "plots"), "--out", str(tmp_path / "r.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] in {"LIKELY_REAL", "NEEDS_REVIEW", "LIKELY_FAKE"}
    assert (tmp_path / "plots" / "spectrum.png").exists() and (tmp_path / "plots" / "rois.png").exists()
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == out["verdict"]


def test_cli_analyze_missing_file_returns_2(capsys):
    assert main(["analyze", "nope.mp4"]) == 2
