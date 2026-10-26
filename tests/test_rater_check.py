"""C6 second-rater tool: the sample is stratified and blind, and the score is Cohen's kappa."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "scripts", "rater_check.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rater_check  # noqa: E402


def _exp(tmp_path):
    d = tmp_path / "exp"
    (d / "steps").mkdir(parents=True)
    classes = ["ok_call"] * 50 + ["missing_call"] * 30 + ["malformed"] * 5 + ["end_turn"] * 15
    with open(d / "steps" / "r1.jsonl", "w") as f:
        for i, c in enumerate(classes):
            f.write(json.dumps({"run_id": f"r{i}", "variant": "q4_0", "backend": "cuda", "repair": "none",
                                "failure_class": c, "raw_text": f"text {i}", "turn": 0, "step": i,
                                "decoded_calls": [], "schema_errors": []}) + "\n")
    return d


def test_sample_is_blind_and_covers_every_class(tmp_path):
    d = _exp(tmp_path)
    out, key = tmp_path / "s.csv", tmp_path / "k.csv"
    subprocess.run([sys.executable, TOOL, "sample", str(d), "--n", "30", "--out", str(out), "--key", str(key)],
                   check=True, capture_output=True)
    shown = list(csv.DictReader(open(out)))
    hidden = list(csv.DictReader(open(key)))
    assert len(shown) == len(hidden) == 30
    for col in ("failure_class", "variant", "backend", "repair", "run_id"):
        assert col not in shown[0]
    got = {r["failure_class"] for r in hidden}
    assert got == {"ok_call", "missing_call", "malformed", "end_turn"}           # rare classes included
    counts = {c: sum(r["failure_class"] == c for r in hidden) for c in got}
    assert counts["ok_call"] > counts["end_turn"] > counts["malformed"] >= 1      # roughly proportional


def test_score_reports_kappa(tmp_path):
    d = _exp(tmp_path)
    out, key = tmp_path / "s.csv", tmp_path / "k.csv"
    subprocess.run([sys.executable, TOOL, "sample", str(d), "--out", str(out), "--key", str(key)], check=True,
                   capture_output=True)
    rows = list(csv.DictReader(open(out)))
    truth = {r["item"]: r["failure_class"] for r in csv.DictReader(open(key))}
    for r in rows:                                                   # the rater agrees on every item
        r["label"] = truth[r["item"]]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p = subprocess.run([sys.executable, TOOL, "score", str(out), str(key)], check=True, capture_output=True,
                       text=True)
    assert "raw agreement 1.00, Cohen's kappa 1.00" in p.stdout


def test_cohen_kappa_known_value():
    po, k = rater_check.cohen_kappa(list("aabb"), list("abab"))
    assert po == 0.5 and abs(k - 0.0) < 1e-9
