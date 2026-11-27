"""Registered RQ2 tests (analysis/rq2_tests.py): TOST, stratified ratio CI, config mapping, planted contrasts."""
import numpy as np
import pandas as pd

from analysis.rq2_tests import (CONFIGS, CONTRASTS, perm_one_sided, run_contrast, stratified_ratio_ci, tost_paired,
                                mcnemar_greedy)


def test_perm_one_sided_matches_exact_enumeration():
    d = np.array([0.3, -0.1, 0.2, 0.25, 0.05, -0.02, 0.4, 0.1])
    signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).reshape(len(d), -1).T
    exact = np.mean(signs @ d >= d.sum() - 1e-12)
    p = perm_one_sided(d, "greater", 200_000, np.random.default_rng(0))
    assert abs(p - exact) < 0.004
    assert perm_one_sided(-d, "less", 200_000, np.random.default_rng(1)) - exact < 0.004


def test_tost_equivalent_vs_shifted():
    rng = np.random.default_rng(0)
    eq = rng.normal(0.0, 0.05, 300)                       # true diff 0, SE 0.003: well inside +/- 0.03
    r = tost_paired(eq, 0.03, np.random.default_rng(1))
    assert r["p"] < 0.01 and -0.03 < r["ci90"][0] < r["ci90"][1] < 0.03
    sh = rng.normal(0.06, 0.05, 300)                      # true diff 0.06: outside the margin
    assert tost_paired(sh, 0.03, np.random.default_rng(2))["p"] > 0.5
    wide = rng.normal(0.0, 0.5, 30)                       # too noisy to show equivalence
    assert tost_paired(wide, 0.03, np.random.default_rng(3))["p"] > 0.05


def test_stratified_ratio_ci():
    rng = np.random.default_rng(0)
    n = np.full(200, 40.0)
    fb = rng.binomial(40, 0.30, 200).astype(float)
    fa = rng.binomial(40, 0.21, 200).astype(float)        # planted ratio 0.7
    strata = np.repeat(["m0", "m1"], 100)
    r = stratified_ratio_ci(fa, n, fb, n, strata, np.random.default_rng(1))
    assert np.isclose(r["ratio"], fa.sum() / fb.sum())
    assert r["ratio_ci95"][0] < 0.7 < r["ratio_ci95"][1] and r["ratio_ci95"][1] < 1


def _ep(effect=0.6, n_tasks=120, seed=0):
    """Episode table for three models with every configuration; failures ~ Poisson(n * rate), treatment rates
    scaled by ``effect`` for the configurations that §6 predicts lower; success Bernoulli."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in ("qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct"):
        base = rng.uniform(0.15, 0.35, n_tasks)
        for name in CONFIGS:
            mult = 1.0 if name in ("Q4_0", "train-longer", "Q5_0", "R2b") else effect
            for t in range(n_tasks):
                for s in (0, 5, 6, 7, 8):
                    n = int(rng.integers(5, 15))
                    f = int(rng.poisson(n * base[t] * mult))
                    rows.append({"run_id": f"{m}-{name}-{t}-{s}", "model": m, "config": name, "benchmark": "b",
                                 "task_id": f"t{t}", "seed": s, "f_any": f, "f_format": f // 3, "n": n,
                                 "valid": rng.random() < 0.5})
    return pd.DataFrame(rows)


def test_planted_reduction_detected_and_null_not():
    ep = _ep(effect=0.7)
    models = ["qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct"]
    rng = np.random.default_rng(0)
    r = run_contrast(ep, models, 4, "Q8_0", "Q4_0", "any", "lower", (5, 6, 7, 8), rng)
    assert r["clusters"] == 360 and r["pairs"] == 1440
    assert r["p"] < 0.001 and r["pooled"]["ratio_ci95"][0] < 0.7 < r["pooled"]["ratio_ci95"][1]
    assert set(r["per_model"]) == set(models)
    r0 = run_contrast(ep, models, 5, "R2b", "Q5_0", "any", "two-sided", (5, 6, 7, 8), rng)
    assert r0["p"] > 0.01


def test_equivalence_with_inclusion_and_all_task_sensitivity():
    ep = _ep()
    models = ["qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct"]
    inc = {m: {("b", f"t{t}") for t in range(20)} for m in models}
    r = run_contrast(ep, models, 7, "R2c", "Q8_0", "success", "equivalence", (5, 6, 7, 8),
                     np.random.default_rng(0), inc)
    assert r["clusters"] == 60 and r["all_tasks"]["clusters"] == 360
    assert 0 <= r["p"] <= 1 and r["margin"] == 0.03


def test_mcnemar_uses_greedy_pairs_only():
    ep = _ep()
    s = mcnemar_greedy(ep, ["qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct"], "R1", "Q4_0", 0)
    assert s["pairs"] == 360 and 0 <= s["p"] <= 1


def test_contrast_list_is_the_registered_one():
    assert [c[0] for c in CONTRASTS] == list(range(1, 11))
    assert [(c[1], c[2]) for c in CONTRASTS] == [("R1", "Q4_0"), ("R1", "Q4_0"), ("Q5_0", "Q4_0"), ("Q8_0", "Q4_0"),
                                                 ("R2b", "Q5_0"), ("R2c", "Q4_0"), ("R2c", "Q8_0"), ("QAD-KL", "Q4_0"),
                                                 ("QAD-KL+CE", "Q4_0"), ("QAD-KL+CE", "train-longer")]
    assert [c[4] for c in CONTRASTS].count("lower") == 7
