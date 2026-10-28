"""C10: the M1 'planted effect' exit criterion and basic test properties."""
import numpy as np
import pandas as pd

from testbed.stats import compare, mcnemar_exact, planted_effect_check


def test_planted_effect_detected_and_null_not():
    r = planted_effect_check()
    assert r["detected"] and r["ci_covers_realized"]
    assert not r["null_false_positive"]


def test_mcnemar_symmetric_and_exact():
    a = np.array([1] * 10 + [0] * 2 + [1] * 8)
    b = np.array([0] * 10 + [1] * 2 + [1] * 8)
    r = mcnemar_exact(a, b)
    assert (r["n10"], r["n01"]) == (10, 2)
    assert abs(r["p"] - 0.0386) < 1e-3          # exact binomial, 12 discordant, 10 vs 2
    assert mcnemar_exact(b, a)["p"] == r["p"]


def test_compare_pairs_on_task_and_seed():
    rows = []
    for t in range(10):
        for s in (1, 2):
            rows.append({"task_id": f"t{t}", "seed": s, "variant": "bf16", "backend": "cuda", "repair": "none", "valid": True})
            rows.append({"task_id": f"t{t}", "seed": s, "variant": "q4_0", "backend": "cuda", "repair": "none", "valid": t % 2 == 0})
    c = compare(pd.DataFrame(rows), {"variant": "bf16"}, {"variant": "q4_0"})
    assert c["n_pairs"] == 20 and c["acc_a"] == 1.0 and c["acc_b"] == 0.5
    assert abs(c["ci_diff"] - 0.5) < 1e-9
