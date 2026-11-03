"""train/compare_loss.py: the S7 walltime check (uninterrupted run A versus cut-and-resumed run B)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
import compare_loss  # noqa: E402


def _run(d, losses):
    d.mkdir(parents=True)
    with open(d / "loss.jsonl", "w") as f:
        for i, l in enumerate(losses, 1):
            f.write(json.dumps({"step": i, "loss": l, "t": 0.0, "peak_alloc_mib": 9000 + i}) + "\n")
    return str(d)


B_LOG = """\
job 123 on g7; CUDA_VISIBLE_DEVICES=0
S7 run B tag s7 steps 6 limit 10:00 restart 0 start 1
step 1 loss 0.5
walltime near at 600: stopping qad_toy and requeueing 123
job 123 on g7; CUDA_VISIBLE_DEVICES=0
S7 run B tag s7 steps 6 limit 10:00 restart 1 start 700
{"event": "resumed", "from_step": 5, "t": 701.0}
{"event": "done", "peak_alloc_mib": 9100, "peak_reserved_mib": 9500}
S7 run B finished at 900
"""


def test_identical_and_resumed_passes(tmp_path):
    a = _run(tmp_path / "A", [0.5, 0.4, 0.3, 0.25, 0.2, 0.19])
    b = _run(tmp_path / "B", [0.5, 0.4, 0.3, 0.25, 0.2, 0.19])
    log = tmp_path / "b.out"
    log.write_text(B_LOG)
    r = compare_loss.compare(a, b, [str(log)])
    assert r["identical"] and r["b_was_cut_and_resumed"] and r["pass"]
    assert r["b_log"]["resumed_from"] == [5] and r["b_log"]["walltime_signals"] == 1
    assert r["max_abs_diff"] == 0.0 and r["first_diff_step"] is None


def test_any_difference_fails(tmp_path):
    a = _run(tmp_path / "A", [0.5, 0.4, 0.3])
    b = _run(tmp_path / "B", [0.5, 0.4000001, 0.3])
    r = compare_loss.compare(a, b)
    assert not r["identical"] and not r["pass"] and r["first_diff_step"] == 2


def test_missing_steps_fail(tmp_path):
    a = _run(tmp_path / "A", [0.5, 0.4, 0.3])
    b = _run(tmp_path / "B", [0.5, 0.4])
    r = compare_loss.compare(a, b)
    assert r["missing_in_b"] == [3] and not r["pass"]


def test_uninterrupted_b_is_not_a_walltime_test(tmp_path):
    a = _run(tmp_path / "A", [0.5, 0.4])
    b = _run(tmp_path / "B", [0.5, 0.4])
    log = tmp_path / "b.out"
    log.write_text("S7 run B tag s7 steps 2 limit 10:00 restart 0 start 1\n")
    r = compare_loss.compare(a, b, [str(log)])
    assert r["identical"] and not r["b_was_cut_and_resumed"] and not r["pass"]
