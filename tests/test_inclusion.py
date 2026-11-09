"""Inclusion set: registered rule over sampled reference trials; greedy pass ignored; C11 filter."""
import json
import os

import pytest

from testbed.inclusion import build, filter_tasks

B = "bfcl_multi_turn_base"


def _make(d, results, greedy_valid=None):
    """results: {task: [valid per sampled trial or None = unfinished]}; greedy_valid: {task: bool}."""
    os.makedirs(f"{d}/episodes", exist_ok=True)
    specs = []
    for task, trials in results.items():
        for k, v in enumerate(trials):
            rid = f"{task}-s{k}"
            specs.append({"run_id": rid, "variant": "bf16", "backend": "cuda", "repair": "none", "benchmark": B,
                          "task_id": task, "decoding": {"temperature": 0.7}})
            if v is not None:
                json.dump({"run_id": rid, "valid": v}, open(f"{d}/episodes/{rid}.json", "w"))
    for task, v in (greedy_valid or {}).items():
        rid = f"{task}-greedy"
        specs.append({"run_id": rid, "variant": "bf16", "backend": "cuda", "repair": "none", "benchmark": B,
                      "task_id": task, "decoding": {"temperature": 0.0}})
        json.dump({"run_id": rid, "valid": v}, open(f"{d}/episodes/{rid}.json", "w"))
    with open(f"{d}/specs.jsonl", "w") as f:
        f.write("\n".join(json.dumps(s) for s in specs) + "\n")


def test_rule_and_greedy_ignored(tmp_path):
    d = str(tmp_path)
    _make(d, {"t1": [False, False, True, False], "t2": [False] * 4, "t3": [True] * 4},
          greedy_valid={"t2": True})                              # a greedy pass alone does not include t2
    r = build(d)
    assert r["benchmarks"][B]["included"] == ["t1", "t3"]
    assert r["benchmarks"][B]["excluded"] == ["t2"]
    assert r["trials_per_task"] == 4 and r["complete"] and r["single_turn_only"]   # 2 < 120


def test_incomplete_reference_refused(tmp_path):
    d = str(tmp_path)
    _make(d, {"t1": [False, None, False, False]})
    with pytest.raises(ValueError, match="unfinished"):
        build(d)
    assert build(d, allow_incomplete=True)["benchmarks"][B]["incomplete"] == ["t1"]


def test_filter_tasks(tmp_path):
    p = tmp_path / "inc.json"
    json.dump({"benchmarks": {B: {"included": ["t3", "t1"], "excluded": ["t2"], "incomplete": []}}}, open(p, "w"))
    assert filter_tasks({B: ["t1", "t2", "t3"]}, str(p)) == {B: ["t1", "t3"]}       # original order kept
    with pytest.raises(ValueError, match="no entry"):
        filter_tasks({B: ["t1"], "tau2_verified_airline": ["7"]}, str(p))
