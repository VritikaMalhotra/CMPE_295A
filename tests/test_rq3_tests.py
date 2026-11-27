"""Registered RQ3 tests (analysis/rq3_tests.py): gap shares, D, bootstrap, triplet pairing, undefined gaps."""
import numpy as np
import pandas as pd

from analysis.rq3_tests import gap_test, shares_from_means, triplet_clusters

MODELS = ["qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct"]


def test_shares_formula():
    # S: BF16 .50, Q4 .30, repair .46 -> 0.8 of the success gap; F: BF16 .20, Q4 .30, repair .28 -> 0.2 of the failure gap
    r = shares_from_means(np.array([0.50, 0.30, 0.46, 0.20, 0.30, 0.28]))
    assert np.isclose(r["share_success"], 0.8) and np.isclose(r["share_failure"], 0.2) and np.isclose(r["D"], 0.6)


def _clusters(share_s, share_f, n=150, seed=0, gap_s=0.2, gap_f=0.1):
    rng = np.random.default_rng(seed)
    rows = []
    for m in MODELS:
        for t in range(n):
            sb = np.clip(rng.normal(0.5, 0.1), 0, 1)
            fb = np.clip(rng.normal(0.2, 0.05), 0, 1)
            rows.append({"model": m, "benchmark": "b", "task_id": f"t{t}",
                         "success_bf": sb + rng.normal(0, 0.05), "success_q4": sb - gap_s + rng.normal(0, 0.05),
                         "success_rep": sb - gap_s * (1 - share_s) + rng.normal(0, 0.05),
                         "rate_bf": fb + rng.normal(0, 0.02), "rate_q4": fb + gap_f + rng.normal(0, 0.02),
                         "rate_rep": fb + gap_f * (1 - share_f) + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)


def test_planted_difference_detected_and_null_not():
    r = gap_test(_clusters(0.8, 0.2), np.random.default_rng(1))
    assert r["testable"] and not r["unstable"]
    assert r["p"] < 0.01 and r["D_ci95"][0] < 0.6 < r["D_ci95"][1] and r["D_ci95"][0] > 0
    r0 = gap_test(_clusters(0.5, 0.5, seed=4), np.random.default_rng(2))
    assert r0["p"] > 0.05 and r0["D_ci95"][0] < 0 < r0["D_ci95"][1]


def test_no_gap_is_not_testable_and_flagged():
    r = gap_test(_clusters(0.5, 0.5, gap_s=-0.05), np.random.default_rng(3))
    assert not r["testable"] and r["p"] == 1.0 and r["D_ci95"] is None
    tiny = gap_test(_clusters(0.5, 0.5, gap_s=0.004, seed=5), np.random.default_rng(4))
    assert tiny["unstable"]


def test_triplet_pairing_keeps_trials_all_three_finished():
    rows = []
    for t in range(4):
        for s in (5, 6):
            for c in ("BF16", "Q4_0", "R1"):
                if c == "R1" and t == 3 and s == 6:
                    continue                                  # one repair trial unfinished
                rows.append({"model": "qwen3-0.6b", "config": c, "benchmark": "b", "task_id": f"t{t}", "seed": s,
                             "success": float(c == "BF16"), "rate": {"BF16": 0.1, "Q4_0": 0.3, "R1": 0.2}[c],
                             "f_any": 1, "n": 10})
    c, n = triplet_clusters(pd.DataFrame(rows), ["qwen3-0.6b"], "R1", (5, 6))
    assert n == 7 and len(c) == 4
    assert np.allclose(c.rate_q4, 0.3) and np.allclose(c.success_bf, 1.0)
