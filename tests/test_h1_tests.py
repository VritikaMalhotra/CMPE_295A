"""Registered H1 tests (analysis/h1_tests.py): planted effects, null calibration, Holm, GEE, pairing."""
import numpy as np
import pandas as pd
import pytest

from analysis.h1_tests import (contrast, holm, loc_design, paired, perm_test_paired, poisson_gee, wald, wilson,
                               DEPTHS)


def test_perm_detects_planted_shift_and_not_null():
    rng = np.random.default_rng(1)
    d = rng.normal(0.03, 0.1, 300)                       # planted mean shift, about 5 SE
    assert perm_test_paired(d, rng=np.random.default_rng(2))["p"] < 0.001
    null = rng.normal(0.0, 0.1, 300)
    assert perm_test_paired(null, rng=np.random.default_rng(3))["p"] > 0.01
    assert perm_test_paired(np.zeros(50))["p"] == 1.0


def test_perm_null_calibration():
    """Under H0 the rejection rate at 0.05 is about 0.05 (200 null datasets; binomial 99% band)."""
    rng = np.random.default_rng(7)
    rej = sum(perm_test_paired(rng.normal(0, 1, 60), n_perm=999, rng=rng)["p"] < 0.05 for _ in range(200))
    assert 2 <= rej <= 22


def test_perm_matches_exact_enumeration():
    d = np.array([0.3, -0.1, 0.2, 0.25, 0.05, -0.02, 0.4, 0.1])
    signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).reshape(len(d), -1).T
    exact = np.mean(np.abs(signs @ d) >= abs(d.sum()) - 1e-12)
    p = perm_test_paired(d, n_perm=200_000, rng=np.random.default_rng(0))["p"]
    assert abs(p - exact) < 0.004


def test_holm_known_values():
    # textbook example: p = .01, .04, .03, .005 -> adjusted .03, .06, .06, .02
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert holm([0.9, 0.8]) == [1.0, 1.0]


def test_wilson_known_value():
    lo, hi = wilson(10, 100)
    assert abs(lo - 0.0552) < 1e-3 and abs(hi - 0.1744) < 1e-3


def _steps(n_tasks=300, n_eps=4, beta_q=0.4, beta_qcall=0.5, beta_qdeep=0.6, seed=0):
    """Step table with planted log-rate effects: 4-bit main, 4-bit x call, 4-bit x d7+; task random effects."""
    rng = np.random.default_rng(seed)
    rows = []
    a = rng.normal(0, 0.5, n_tasks)
    for m_i, m in enumerate(["m0", "m1"]):
        for t in range(n_tasks):
            for var in ("bf16", "q4_0"):
                q = var != "bf16"
                for e in range(n_eps):
                    for k in range(10):
                        call = k % 2 == 0
                        depth = DEPTHS[min(3, [0, 0, 1, 1, 2, 2, 2, 3, 3, 3][k])]
                        eta = (-4.0 + a[t] + 0.2 * m_i + 0.3 * call + 0.1 * DEPTHS.index(depth)
                               + q * (beta_q + beta_qcall * call + beta_qdeep * (depth == "d7+")))
                        rows.append({"model": m, "variant": var, "task_id": f"t{t}", "benchmark": "b",
                                     "call": call, "depth": depth, "fail": rng.random() < np.exp(eta)})
    return pd.DataFrame(rows)


def test_gee_recovers_planted_interactions():
    s = _steps()
    X, names = loc_design(s, ["m0", "m1"])
    g = poisson_gee(s.fail.values.astype(float), X, (s.benchmark + s.task_id).values)
    assert g["converged"]
    wc = wald(g["beta"], g["cov"], [names.index("prec4:call")])
    assert wc["p"] < 0.01 and wc["rr_ci95"][0] < np.exp(0.5) < wc["rr_ci95"][1]
    wd = wald(g["beta"], g["cov"], [names.index(f"prec4:depth_{d}") for d in DEPTHS[1:]])
    assert wd["p"] < 0.01 and wd["estimates"][-1] > 0
    s0 = _steps(beta_qcall=0.0, beta_qdeep=0.0, seed=3)
    X0, _ = loc_design(s0, ["m0", "m1"])
    g0 = poisson_gee(s0.fail.values.astype(float), X0, (s0.benchmark + s0.task_id).values)
    assert wald(g0["beta"], g0["cov"], [names.index("prec4:call")])["p"] > 0.01


def test_gee_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    s = _steps(n_tasks=60, n_eps=2)
    X, _ = loc_design(s, ["m0", "m1"])
    y = s.fail.values.astype(float)
    groups = pd.factorize(s.benchmark + s.task_id)[0]
    g = poisson_gee(y, X, groups)
    ref = sm.GEE(y, X, groups=groups, family=sm.families.Poisson(),
                 cov_struct=sm.cov_struct.Independence()).fit()
    assert np.allclose(g["beta"], ref.params, atol=1e-6)
    assert np.allclose(np.sqrt(np.diag(g["cov"])), ref.bse, rtol=1e-4)


def _ep(rows):
    return pd.DataFrame(rows, columns=["run_id", "model", "variant", "backend", "repair", "benchmark", "task_id",
                                       "seed", "f_any", "f_tep", "n"])


def test_pairing_keeps_only_trials_both_finished_and_rates_are_task_means():
    rows = []
    for t in range(3):
        for s in (5, 6):
            rows.append([f"r{t}{s}", "m", "bf16", "cuda", "none", "b", f"t{t}", s, 1, 0, 10])
            if not (t == 2 and s == 6):                     # one 4-bit trial unfinished
                rows.append([f"q{t}{s}", "m", "q4_0", "cuda", "none", "b", f"t{t}", s, 2 + s - 5, 0, 10])
    j = paired(_ep(rows), "m", "cuda", "q4_0", [5, 6])
    assert len(j) == 5
    c = contrast(j, "any_failure", np.random.default_rng(0), [5, 6])
    assert c["pairs"] == 5 and c["tasks"] == 3
    # per-task 4-bit means: t0 = t1 = (0.2 + 0.3)/2 = 0.25, t2 = 0.2; reference 0.1 everywhere
    assert np.isclose(c["mean_diff"], ((0.15 + 0.15 + 0.1) / 3))
    assert np.isclose(c["pooled"]["ratio"], (12 / 50) / (5 / 50))
