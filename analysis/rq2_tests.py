#!/usr/bin/env python
"""Registered H2 tests, family RQ2 (pre-registration https://osf.io/xwq5y, §6), written after registration.

As for H1 (``analysis/h1_tests.py``, deviation note ``docs/m3_h1_tests_deviation.md``), the code frozen at
``1e13897`` has no RQ2 tests. This module adds them as §6 states, reusing the H1 module's permutation test, Holm,
Wilson and loader, and the frozen ``damage_table.ratio_ci`` and ``stats.mcnemar_exact``. Written and checked on
development data and synthetic data before it read any C3 cell (``docs/m3_rq2_tests_deviation.md``).

Family RQ2, 10 contrasts, Q4_0 base, pooled over the three core models; the cluster is task x model, model is a
stratum; outcome any failure per step unless stated; Holm within the family; per-model estimates for every
contrast:

    1 R1 vs Q4_0, format failures (malformed + unknown_tool + invalid_args)   R1 lower
    2 R1 vs Q4_0, any failure                                                 two-sided
    3 Q5_0 vs Q4_0                                                            Q5_0 lower
    4 Q8_0 vs Q4_0                                                            Q8_0 lower
    5 R2b vs Q5_0 (matched 5.5 bpw)                                           two-sided
    6 R2c vs Q4_0                                                             R2c lower
    7 R2c vs Q8_0: equivalence on episode success, TOST +/- 3 points          "matches" only if TOST rejects
    8 QAD-KL vs Q4_0                                                          QAD lower
    9 QAD-KL+CE vs Q4_0                                                       QAD lower
   10 QAD-KL+CE vs train-longer control (both Q4_0)                           QAD lower

* Per-cluster rate: the mean over the cluster's paired sampled trials (seeds 5-8) of each episode's
  failures / steps (§4, as in H1). Test: paired permutation, 10,000 random sign flips of the per-cluster
  differences (treatment - comparison), two-sided. "Lower": supported if Holm p < 0.05 and the pooled ratio < 1.
  Two-sided contrasts: "significant" if Holm p < 0.05, direction from the pooled ratio.
* Pooled ratio of rates (treatment / comparison) with a paired cluster bootstrap CI stratified by model (10,000
  resamples of tasks within each model).
* Contrast 7: per-cluster mean episode success over the paired sampled trials; TOST with margin 0.03 as two
  one-sided permutation tests on the shifted differences, p_TOST = max of the two; "matches" if Holm p_TOST < 0.05.
  Tasks: the registered inclusion sets (§5, secondary success analysis), with an all-task sensitivity.
* Secondary (report only, Holm within its own 10): McNemar's exact test on greedy-pass episode success (seed 0),
  pairs = model x task, the frozen ``stats.mcnemar_exact``.

    python analysis/rq2_tests.py --dir <run dir> [--dir ...] [--inclusion results/m3/inclusion_holdout_<m>.json ...] \\
        --json results/m3/rq2_holdout.json --md docs/m3_rq2_holdout.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from analysis.damage_table import ratio_ci  # noqa: E402  (frozen)
from analysis.h1_tests import ALPHA, N_BOOT, holm, load_dir, perm_test_paired, sha, wilson  # noqa: E402
from testbed.stats import mcnemar_exact  # noqa: E402  (frozen)

CORE_MODELS = ("qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct")
FORMAT_LABELS = {"malformed", "unknown_tool", "invalid_args"}
# (variant, repair) of each configuration in the run specs
CONFIGS = {"Q4_0": ("q4_0", "none"), "R1": ("q4_0", "r1"), "Q5_0": ("q5_0", "r2"), "Q8_0": ("q8_0", "r2"),
           "R2b": ("r2b", "r2"), "R2c": ("r2c_tool", "r2"), "QAD-KL": ("r3_qad_kl", "r3"),
           "QAD-KL+CE": ("r3_qad_klce", "r3"), "train-longer": ("r3_ctrl", "r3")}
CONTRASTS = [(1, "R1", "Q4_0", "format", "lower"), (2, "R1", "Q4_0", "any", "two-sided"),
             (3, "Q5_0", "Q4_0", "any", "lower"), (4, "Q8_0", "Q4_0", "any", "lower"),
             (5, "R2b", "Q5_0", "any", "two-sided"), (6, "R2c", "Q4_0", "any", "lower"),
             (7, "R2c", "Q8_0", "success", "equivalence"), (8, "QAD-KL", "Q4_0", "any", "lower"),
             (9, "QAD-KL+CE", "Q4_0", "any", "lower"), (10, "QAD-KL+CE", "train-longer", "any", "lower")]
TOST_MARGIN = 0.03
SOURCES = ("analysis/rq2_tests.py", "analysis/h1_tests.py", "analysis/damage_table.py", "testbed/stats.py",
           "testbed/classifier.py")
KEY = ["benchmark", "task_id", "seed"]


# ---------------------------------------------------------------- statistics

def perm_one_sided(d: np.ndarray, alternative: str, n_perm: int, rng: np.random.Generator) -> float:
    """One-sided sign-flip permutation p for mean(d): 'greater' (H1: mean > 0) or 'less'."""
    d = np.asarray(d, float)
    obs = d.mean()
    signs = rng.integers(0, 2, size=(n_perm, len(d)), dtype=np.int8) * 2 - 1
    null = (signs @ d) / len(d)
    hits = np.sum(null >= obs - 1e-12) if alternative == "greater" else np.sum(null <= obs + 1e-12)
    return float((1 + int(hits)) / (n_perm + 1))


def tost_paired(d: np.ndarray, margin: float, rng: np.random.Generator, n_perm: int = 10_000) -> dict:
    """Equivalence of mean(d) to 0 within +/- margin: H0a mean <= -margin (test d + margin > 0) and H0b
    mean >= margin (test d - margin < 0), each a one-sided sign-flip permutation test; p_TOST = max."""
    d = np.asarray(d, float)
    pa = perm_one_sided(d + margin, "greater", n_perm, rng)
    pb = perm_one_sided(d - margin, "less", n_perm, rng)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.quantile(d[idx].mean(1), [0.05, 0.95])            # the 90% CI that matches a TOST at 0.05
    return {"mean_diff": float(d.mean()), "p_lower": pa, "p_upper": pb, "p": max(pa, pb), "n": int(len(d)),
            "ci90": [float(lo), float(hi)], "margin": margin}


def stratified_ratio_ci(fa, na, fb, nb, strata, rng) -> dict:
    """Ratio of pooled rates a / b and its percentile CI from a cluster bootstrap that resamples clusters within each
    stratum (model), 10,000 resamples; plus the difference per 100 steps."""
    fa, na, fb, nb = (np.asarray(x, float) for x in (fa, na, fb, nb))
    strata = np.asarray(strata)
    ra, rb = fa.sum() / na.sum(), fb.sum() / nb.sum()
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    idx = np.concatenate([g[rng.integers(0, len(g), size=(N_BOOT, len(g)))] for g in groups], axis=1)
    ba, bb = fa[idx].sum(1) / na[idx].sum(1), fb[idx].sum(1) / nb[idx].sum(1)
    out = {"rate_a_per100": 100 * ra, "rate_b_per100": 100 * rb, "events_a": int(fa.sum()), "events_b": int(fb.sum()),
           "steps_a": int(na.sum()), "steps_b": int(nb.sum()), "ratio": float(ra / rb) if rb > 0 else None,
           "diff_per100": float(100 * (ra - rb))}
    ok = bb > 0
    out["ratio_ci95"] = ([float(x) for x in np.quantile(ba[ok] / bb[ok], [0.025, 0.975])]
                         if rb > 0 and ok.mean() > 0.99 else None)
    out["diff_ci95_per100"] = [float(x) for x in np.quantile(100 * (ba - bb), [0.025, 0.975])]
    return out


# ---------------------------------------------------------------- data

def episode_table(steps: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    s = steps.assign(fmt=steps.label.isin(FORMAT_LABELS))
    agg = s.groupby("run_id").agg(f_any=("fail", "sum"), f_format=("fmt", "sum"), n=("fail", "size")).reset_index()
    e = eps.merge(agg, on="run_id", how="left").fillna({"f_any": 0, "f_format": 0, "n": 0})
    if len(set(CONFIGS.values())) != len(CONFIGS):
        raise ValueError("two configuration names map to the same (variant, repair)")
    e["config"] = None
    for name, (v, r) in CONFIGS.items():
        e.loc[(e.variant == v) & (e.repair == r) & (e.backend == "cuda"), "config"] = name
    return e[e.config.notna()]


def pair(ep: pd.DataFrame, model: str, a: str, b: str, seeds, tasks: set | None = None) -> pd.DataFrame:
    sel = ep[(ep.model == model) & ep.seed.isin(seeds)]
    if tasks is not None:
        sel = sel[[(bm, t) in tasks for bm, t in zip(sel.benchmark, sel.task_id)]]
    j = sel[sel.config == a].merge(sel[sel.config == b], on=KEY, suffixes=("_a", "_b"))
    return j[(j.n_a > 0) & (j.n_b > 0)]


def clusters(ep, models, a, b, seeds, f, tasks_by_model=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (model, benchmark, task) cluster: mean episode rate (or success) and pooled counts; plus the pairs."""
    rows, pairs = [], []
    for m in models:
        j = pair(ep, m, a, b, seeds, None if tasks_by_model is None else tasks_by_model.get(m, set()))
        if j.empty:
            continue
        j = j.assign(model=m)
        if f == "success":
            j = j.assign(x_a=j.valid_a.astype(float), x_b=j.valid_b.astype(float), c_a=0.0, c_b=0.0)
        else:
            j = j.assign(x_a=j[f + "_a"] / j.n_a, x_b=j[f + "_b"] / j.n_b, c_a=j[f + "_a"], c_b=j[f + "_b"])
        pairs.append(j)
        t = j.groupby(["benchmark", "task_id"]).agg(x_a=("x_a", "mean"), x_b=("x_b", "mean"), f_a=("c_a", "sum"),
                                                    n_a=("n_a", "sum"), f_b=("c_b", "sum"), n_b=("n_b", "sum"))
        rows.append(t.reset_index().assign(model=m))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows, ignore_index=True), pd.concat(pairs, ignore_index=True)


def per_seed(pairs: pd.DataFrame, f: str, seeds) -> dict | None:
    vals = []
    for s in seeds:
        p = pairs[pairs.seed == s]
        if len(p):
            if f == "success":
                vals.append((100 * p.valid_a.mean(), 100 * p.valid_b.mean()))
            else:
                vals.append((100 * p[f + "_a"].sum() / p.n_a.sum(), 100 * p[f + "_b"].sum() / p.n_b.sum()))
    if len(vals) < 2:
        return None
    v = np.array(vals)
    return {"a": v[:, 0].tolist(), "b": v[:, 1].tolist(), "a_mean": float(v[:, 0].mean()),
            "a_sd": float(v[:, 0].std(ddof=1)), "b_mean": float(v[:, 1].mean()), "b_sd": float(v[:, 1].std(ddof=1))}


# ---------------------------------------------------------------- contrasts

def run_contrast(ep, models, cid, a, b, outcome, kind, seeds, rng, inclusion=None) -> dict:
    f = {"format": "f_format", "any": "f_any", "success": "success"}[outcome]
    row = {"id": cid, "treatment": a, "comparison": b, "outcome": outcome, "kind": kind}
    tasks = inclusion if (outcome == "success" and inclusion is not None) else None
    c, pairs = clusters(ep, models, a, b, seeds, f, tasks)
    if c.empty:
        return {**row, "missing": True, "p": 1.0}
    row.update({"clusters": int(len(c)), "pairs": int(len(pairs)),
                "clusters_by_model": c.groupby("model").size().to_dict(), "per_seed": per_seed(pairs, f, seeds)})
    d = (c.x_a - c.x_b).values
    if kind == "equivalence":
        row.update(tost_paired(d, TOST_MARGIN, rng))
        row["success_a"], row["success_b"] = float(c.x_a.mean()), float(c.x_b.mean())
        if inclusion is not None:                                     # all-task sensitivity (§5)
            c_all, _ = clusters(ep, models, a, b, seeds, f, None)
            row["all_tasks"] = tost_paired((c_all.x_a - c_all.x_b).values, TOST_MARGIN, rng)
            row["all_tasks"]["clusters"] = int(len(c_all))
        return row
    row.update(perm_test_paired(d, rng=rng))
    row["pooled"] = stratified_ratio_ci(c.f_a, c.n_a, c.f_b, c.n_b, c.model, rng)
    row["per_model"] = {}
    for m, g in c.groupby("model"):
        pm = ratio_ci(g.f_b.values.astype(float), g.n_b.values.astype(float), g.f_a.values.astype(float),
                      g.n_a.values.astype(float), rng)            # frozen: "4bit" slot = treatment
        pm["p"] = perm_test_paired((g.x_a - g.x_b).values, rng=rng)["p"]
        pm["clusters"] = int(len(g))
        row["per_model"][m] = pm
    return row


def mcnemar_greedy(ep, models, a, b, greedy_seed) -> dict:
    js = [pair(ep, m, a, b, [greedy_seed]) for m in models]
    js = [j for j in js if not j.empty]
    if not js:
        return {"pairs": 0, "p": 1.0}
    j = pd.concat(js, ignore_index=True)
    r = mcnemar_exact(j.valid_a.values, j.valid_b.values)        # n10: success under treatment only
    va, vb = j.valid_a.astype(bool), j.valid_b.astype(bool)
    return {"pairs": int(len(j)), **r, "success_a": float(va.mean()), "success_b": float(vb.mean()),
            "wilson_a": wilson(int(va.sum()), len(j)), "wilson_b": wilson(int(vb.sum()), len(j))}


def analyse(dirs, inclusion_files, seeds, greedy_seed, models=CORE_MODELS, seed=295) -> dict:
    parts = [load_dir(d) for d in dirs]
    steps = pd.concat([p[0] for p in parts], ignore_index=True)
    ep = episode_table(steps, pd.concat([p[1] for p in parts], ignore_index=True))
    models = [m for m in models if m in set(ep.model)]
    inclusion = None
    if inclusion_files:
        inclusion = {}
        for p in inclusion_files:
            inc = json.load(open(p))
            m = os.path.basename(p).split("inclusion_")[-1].replace("holdout_", "")[:-len(".json")]
            m = next((x for x in models if x.startswith(m)), m)
            inclusion[m] = {(bm, t) for bm, b in inc["benchmarks"].items() for t in b["included"]}
    rng = np.random.default_rng(seed)
    rows = [run_contrast(ep, models, cid, a, b, o, k, seeds, rng, inclusion) for cid, a, b, o, k in CONTRASTS]
    adj = holm([r["p"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        if r.get("missing"):
            r["verdict"] = "missing"
        elif r["kind"] == "lower":
            r["verdict"] = "supported" if p < ALPHA and r["pooled"]["ratio"] < 1 else "not supported"
        elif r["kind"] == "two-sided":
            r["verdict"] = (("significant, lower" if r["pooled"]["ratio"] < 1 else "significant, higher")
                            if p < ALPHA else "not significant")
        else:
            r["verdict"] = "matches (equivalent)" if p < ALPHA else "equivalence not shown"
    sec = [mcnemar_greedy(ep, models, a, b, greedy_seed) for _, a, b, _, _ in CONTRASTS]
    for s, p in zip(sec, holm([s["p"] for s in sec])):
        s["p_holm"] = p
    for r, s in zip(rows, sec):
        r["secondary_mcnemar_greedy_success"] = s
    return {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "sources_sha256_16": {p: sha(p) for p in SOURCES}, "runs": [p[2] for p in parts], "models": models,
            "seeds": list(seeds), "greedy_seed": greedy_seed, "rng_seed": seed,
            "configs_present": sorted(ep.config.unique().tolist()),
            "inclusion": {m: len(v) for m, v in (inclusion or {}).items()},
            "method": "§6 RQ2 as registered; cluster = task x model, model stratum; per-cluster rate = mean over "
                      "paired sampled trials of failures/steps; paired permutation (10,000 sign flips); Holm within "
                      "the family; pooled ratio with a model-stratified cluster bootstrap CI; contrast 7 TOST +/- 3 "
                      "points on success (permutation, inclusion set; all-task sensitivity); secondary McNemar on "
                      "greedy success (frozen stats.mcnemar_exact), Holm within its own 10", "contrasts": rows}


# ---------------------------------------------------------------- report

def _f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _p(x):
    return "–" if x is None else ("<0.0001" if x < 1e-4 else f"{x:.4f}")


def write_md(res: dict, path: str, json_path: str, label: str) -> None:
    L = [f"# H2 registered tests, family RQ2: {label}\n",
         f"Generated {res['generated']} by `analysis/rq2_tests.py` (sources SHA-256/16: "
         + ", ".join(f"`{k}` {v}" for k, v in res["sources_sha256_16"].items()) + f"). JSON: `{json_path}`.\n",
         "Method: " + res["method"] + f". Models: {', '.join(res['models'])}. Sampled seeds {res['seeds']}; greedy "
         f"seed {res['greedy_seed']}.\n"]
    rows = res["contrasts"]
    n_sup = sum(r.get("verdict") == "supported" for r in rows)
    L.append(f"## RQ2: {n_sup} of {sum(r['kind'] == 'lower' for r in rows)} directional contrasts supported\n")
    L += ["| # | Contrast | Outcome | Clusters / pairs | Treatment per 100 steps | Comparison per 100 steps | "
          "Ratio [95% CI] | p | p (Holm) | Verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = f"{r['treatment']} vs {r['comparison']}"
        if r.get("missing"):
            L.append(f"| {r['id']} | {name} | {r['outcome']} | 0 | – | – | – | – | – | missing |")
            continue
        s = r["per_seed"]
        if r["kind"] == "equivalence":
            ta = f"{_f(s['a_mean'], 1)} ± {_f(s['a_sd'], 1)} % success" if s else "–"
            tb = f"{_f(s['b_mean'], 1)} ± {_f(s['b_sd'], 1)} % success" if s else "–"
            rr = (f"Δ success {_f(100 * r['mean_diff'], 1)} pts, 90% CI [{_f(100 * r['ci90'][0], 1)}, "
                  f"{_f(100 * r['ci90'][1], 1)}], margin ± 3")
        else:
            po = r["pooled"]
            ta, tb = (f"{_f(s['a_mean'])} ± {_f(s['a_sd'])}", f"{_f(s['b_mean'])} ± {_f(s['b_sd'])}") if s else ("–", "–")
            ci = po["ratio_ci95"]
            rr = f"{_f(po['ratio'])} [{_f(ci[0]) if ci else '–'}, {_f(ci[1]) if ci else '–'}]"
        L.append(f"| {r['id']} | {name} | {r['outcome']} | {r['clusters']} / {r['pairs']} | {ta} | {tb} | {rr} | "
                 f"{_p(r['p'])} | {_p(r['p_holm'])} | **{r['verdict']}** |")
    L.append("\nRates: mean ± 1 SD over the four sampled seeds of the pooled (all three models) rate. Contrast 7 uses "
             "the registered inclusion sets (tasks the BF16 reference solves at least once); its p is the TOST p.\n")
    eq = next((r for r in rows if r["kind"] == "equivalence" and not r.get("missing")), None)
    if eq and "all_tasks" in eq:
        a = eq["all_tasks"]
        L.append(f"Contrast 7, all-task sensitivity ({a['clusters']} clusters): Δ success {_f(100 * a['mean_diff'], 1)} "
                 f"pts, 90% CI [{_f(100 * a['ci90'][0], 1)}, {_f(100 * a['ci90'][1], 1)}], TOST p {_p(a['p'])}.\n")
    L += ["## Per-model estimates\n", "| # | Contrast | Model | Clusters | Treatment / comparison per 100 steps | "
          "Ratio [95% CI] | p (unadjusted) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        for m, pm in (r.get("per_model") or {}).items():
            ci = pm["ratio_ci95"]
            L.append(f"| {r['id']} | {r['treatment']} vs {r['comparison']} | {m} | {pm['clusters']} | "
                     f"{_f(pm['rate_4bit_per100'])} / {_f(pm['rate_ref_per100'])} | {_f(pm['ratio'])} "
                     f"[{_f(ci[0]) if ci else '–'}, {_f(ci[1]) if ci else '–'}] | {_p(pm['p'])} |")
    L += ["\n## Secondary: McNemar's exact test on greedy-pass success (report only)\n",
          "| # | Contrast | Pairs | Success treatment [95% Wilson] | Success comparison [95% Wilson] | "
          "Discordant (treatment only / comparison only) | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["secondary_mcnemar_greedy_success"]
        if not s["pairs"]:
            L.append(f"| {r['id']} | {r['treatment']} vs {r['comparison']} | 0 | – | – | – | – | – |")
            continue
        wa, wb = s["wilson_a"], s["wilson_b"]
        L.append(f"| {r['id']} | {r['treatment']} vs {r['comparison']} | {s['pairs']} | {_f(100 * s['success_a'], 1)} % "
                 f"[{_f(100 * wa[0], 1)}, {_f(100 * wa[1], 1)}] | {_f(100 * s['success_b'], 1)} % [{_f(100 * wb[0], 1)}, "
                 f"{_f(100 * wb[1], 1)}] | {s['n10']} / {s['n01']} | {_p(s['p'])} | {_p(s['p_holm'])} |")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--inclusion", action="append", default=[])
    ap.add_argument("--seeds", default="5,6,7,8")
    ap.add_argument("--greedy-seed", type=int, default=0)
    ap.add_argument("--config-map", help="JSON {name: [variant, repair]} overriding CONFIGS (development check only)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--label", default="confirmatory holdout (tests implemented after registration)")
    ap.add_argument("--rng-seed", type=int, default=295)
    a = ap.parse_args()
    if a.config_map:
        CONFIGS.clear()
        CONFIGS.update({k: tuple(v) for k, v in json.loads(a.config_map).items()})
    res = analyse(a.dir, a.inclusion, tuple(int(s) for s in a.seeds.split(",")), a.greedy_seed, seed=a.rng_seed)
    res["label"], res["configs"] = a.label, {k: list(v) for k, v in CONFIGS.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, default=float)
        f.write("\n")
    if a.md:
        write_md(res, a.md, a.json, a.label)
    print("wrote", a.json, a.md or "")


if __name__ == "__main__":
    main()
