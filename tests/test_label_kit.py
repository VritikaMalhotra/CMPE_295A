"""H2 label kit: sheets are blank and blind, the key holds the hidden fields, kappa is right on synthetic labels.

Every label in this file is synthetic and lives only inside the test: no rater label is ever written to the repo.
"""
import csv
import json
import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import label_kit  # noqa: E402
from testbed.classifier import CLASSES  # noqa: E402


def _step(i, variant, raw, calls, results=(), known=True, errs=(), stop="eos", gt=True, last=True, step=0):
    return {"run_id": f"r{i}", "config_key": "c", "task_id": f"t{i}", "seed": 1, "turn": 0, "step": step,
            "variant": variant, "backend": "cuda", "repair": "none", "constraint": "none", "prompt_tokens": 1,
            "completion_tokens": 1, "ttft_ms": None, "decode_ms": None, "latency_ms": 1.0, "stop_reason": stop,
            "raw_text": raw, "n_tool_call_blocks": len(calls), "decoded_calls": list(calls),
            "execution_results": list(results), "known_tools": known, "schema_errors": list(errs),
            "turn_has_ground_truth": gt, "is_last_step_of_turn": last, "forced_quit": False,
            "benchmark": "", "failure_class": "old"}


@pytest.fixture
def run_dirs(tmp_path):
    dirs = []
    for d, variant in (("ref", "bf16"), ("q4", "q4_0")):
        p = tmp_path / d / "steps"
        p.mkdir(parents=True)
        rows = []
        for i in range(60):
            k = i % 4
            if k == 0:
                rows.append(_step(i, variant, "<tool_call>x</tool_call>", ["ls()"], ['{"a": 1}']))
            elif k == 1:
                rows.append(_step(i, variant, "The total is 4 MB.", []))
            elif k == 2:
                rows.append(_step(i, variant, "<tool_call>", ["cd(folder='x')"], ['{"error": "no"}']))
            else:
                rows.append(_step(i, variant, "Which folder?", []))
        for r in rows:                         # one step file per episode, as the logger writes them
            with open(p / f"{r['run_id']}.jsonl", "w") as f:
                f.write(json.dumps(r) + "\n")
        (tmp_path / d / "episodes").mkdir()
        for r in rows:
            (tmp_path / d / "episodes" / f"{r['run_id']}.json").write_text("{}")
        with open(tmp_path / d / "specs.jsonl", "w") as f:   # old runs: benchmark only in the specs
            f.write("\n".join(json.dumps({"run_id": r["run_id"], "benchmark": "bfcl_multi_turn_base"}) for r in rows))
        dirs.append(str(tmp_path / d))
    return dirs


def test_export_blank_blind_sheets_and_separate_key(run_dirs, tmp_path):
    out, key = tmp_path / "kit", tmp_path / "key"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "label_kit.py"), "export", "--runs", *run_dirs,
                        "--out-dir", str(out), "--key-dir", str(key), "--n-sample", "10", "--n-pilot", "40"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name, n in (("rater_sample_30", 10), ("pilot_300", 40)):
        rows = list(csv.DictReader(open(out / f"{name}.csv")))
        assert len(rows) == n
        assert all(row["label"] == "" and row["note"] == "" for row in rows)            # blank
        hidden = {"variant", "run_id", "machine_label", "precision", "failure_class", "task_id"}
        assert not hidden & set(rows[0])                                                  # blind
        krows = list(csv.DictReader(open(key / f"key_{name}.csv")))
        assert [k["item"] for k in krows] == [x["item"] for x in rows]
        assert {k["precision"] for k in krows} == {"16bit", "4bit"}                       # both precisions
        assert all(k["machine_label"] in CLASSES for k in krows)
        assert all(row["benchmark"] == "bfcl_multi_turn_base" for row in rows)            # filled from specs
        # the classifier relabels from the logged fields: these synthetic steps give exactly these four classes
        assert {k["machine_label"] for k in krows} <= {"ok_call", "missing_call_direct_answer",
                                                       "tool_error_payload", "missing_call"}
    s = {x["item"] for x in csv.DictReader(open(out / "rater_sample_30.csv"))}
    p = {x["item"] for x in csv.DictReader(open(out / "pilot_300.csv"))}
    assert not s & p
    m = json.load(open(out / "manifest.json"))
    assert set(m["sheets"]) == {"rater_sample_30", "pilot_300"} and "strata" not in json.dumps(m)


def test_kappa_on_synthetic_labels():
    rng = random.Random(0)
    truth = [rng.choice(["ok_call", "exec_error", "reply", "unknown_tool"]) for _ in range(200)]
    a = {f"P-{i:03d}": t for i, t in enumerate(truth)}
    assert label_kit.agreement(a, dict(a))["kappa"] == pytest.approx(1.0)
    # flip 20% of rater b's labels at random: kappa must fall below the 0.80 target but stay well above 0
    b = {i: (rng.choice([c for c in ("ok_call", "exec_error", "reply", "unknown_tool") if c != v])
             if rng.random() < 0.2 else v) for i, v in a.items()}
    r = label_kit.agreement(a, b)
    assert 0.5 < r["kappa"] < 0.8 and not r["meets_0.80"]
    # hand-checked 2x2 case: po = 0.7, pe = 0.5 -> kappa 0.4
    x = {str(i): v for i, v in enumerate(["reply"] * 5 + ["ok_call"] * 5)}
    y = {str(i): v for i, v in enumerate(["reply"] * 4 + ["ok_call"] + ["reply"] * 2 + ["ok_call"] * 3)}
    assert label_kit.agreement(x, y)["kappa"] == pytest.approx(0.4)


def test_kappa_refuses_unlabelled_or_unknown_labels():
    with pytest.raises(ValueError):
        label_kit.agreement({"1": "reply", "2": ""}, {"1": "reply", "2": "ok_call"})
    with pytest.raises(ValueError):
        label_kit.agreement({"1": "reply"}, {"1": "wrong_tool"})
