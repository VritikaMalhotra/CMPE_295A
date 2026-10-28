"""C10 Analysis & Stats (M1 draft): paired comparisons of episode success between configurations.

Owner: Vritika Malhotra (draft by Yashashav for review).

* ``mcnemar_exact``  - two-sided exact McNemar test on paired binary outcomes (same task x seed).
* ``paired_bootstrap_ci`` - percentile bootstrap CI for the difference in success rate,
                         resampling tasks (clusters) so the two seeds of a task stay together.
* ``planted_effect_check`` - synthetic data with a known drop; the test must detect it and the CI
                         must cover it. This is the M1 "toy stats + planted effect" exit criterion.
The frozen M2 analysis plan replaces these defaults if the pre-registration says otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: paired 0/1 outcomes. Returns discordant counts and the exact two-sided p-value."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n10 = int(np.sum(a & ~b))          # right under a, wrong under b
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)
    return {"n10": n10, "n01": n01, "p": p}


def paired_bootstrap_ci(df: pd.DataFrame, a_col: str, b_col: str, cluster: str = "task_id",
                        n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for mean(a) - mean(b), resampling clusters with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster)]
    diffs = np.array([g[a_col].astype(float).values - g[b_col].astype(float).values for g in groups], dtype=object)
    sums = np.array([d.sum() for d in diffs])
    counts = np.array([len(d) for d in diffs])
    est = sums.sum() / counts.sum()
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(est), "lo": float(lo), "hi": float(hi), "n_clusters": len(groups)}


def compare(episodes: pd.DataFrame, ref: dict, other: dict) -> dict:
    """Pair episodes of two configurations (dicts of column filters) on (task_id, seed)."""
    def pick(f):
        m = np.ones(len(episodes), bool)
        for k, v in f.items():
            m &= episodes[k].values == v
        return episodes[m][["task_id", "seed", "valid"]]
    a, b = pick(ref), pick(other)
    j = a.merge(b, on=["task_id", "seed"], suffixes=("_a", "_b"))
    if j.empty:
        return {"n_pairs": 0}
    out = {"n_pairs": int(len(j)), "acc_a": float(j.valid_a.mean()), "acc_b": float(j.valid_b.mean())}
    out.update(mcnemar_exact(j.valid_a.values, j.valid_b.values))
    out.update({f"ci_{k}": v for k, v in paired_bootstrap_ci(j, "valid_a", "valid_b").items()})
    return out


def planted_effect_check(n_tasks: int = 200, n_seeds: int = 2, p_ref: float = 0.6, drop: float = 0.15,
                         seed: int = 0) -> dict:
    """Simulate a reference config and a damaged one with a known success-rate drop; check recovery."""
    rng = np.random.default_rng(seed)
    task_p = np.clip(rng.normal(p_ref, 0.15, n_tasks), 0.02, 0.98)        # per-task difficulty
    rows = []
    for t in range(n_tasks):
        for s in range(n_seeds):
            a = rng.random() < task_p[t]
            b = a and (rng.random() >= drop / max(task_p.mean(), 1e-9))     # damage flips some successes
            rows.append({"task_id": f"t{t}", "seed": s, "valid_a": int(a), "valid_b": int(b)})
    df = pd.DataFrame(rows)
    true_diff = float(df.valid_a.mean() - df.valid_b.mean())
    mc = mcnemar_exact(df.valid_a.values, df.valid_b.values)
    ci = paired_bootstrap_ci(df, "valid_a", "valid_b", n_boot=4000)
    null = df.assign(valid_b=df.valid_a)                                      # no effect -> must not detect
    mc0 = mcnemar_exact(null.valid_a.values, null.valid_b.values)
    return {"planted_drop": drop, "realized_diff": true_diff, "mcnemar_p": mc["p"], "ci": ci,
            "detected": mc["p"] < 0.05 and ci["lo"] > 0, "ci_covers_realized": ci["lo"] <= true_diff <= ci["hi"],
            "null_p": mc0["p"], "null_false_positive": mc0["p"] < 0.05}
