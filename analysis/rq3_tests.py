#!/usr/bin/env python
"""Registered H3 tests, family RQ3 (pre-registration https://osf.io/xwq5y, §1 and §6), written after registration.

H3 (§1): "Repairs close more of the success gap than of the failure gap." §6: "For R1, Q8_0 and QAD-KL+CE (family
RQ3, 3 tests), the paired bootstrap over tasks of (share of the BF16->Q4_0 success gap closed) - (share of the
any-failure gap closed); supported if the 95% CI excludes 0 in the predicted direction." Holm within RQ3 (§6).

Choices §6 leaves open (deviation note ``docs/m3_rq3_tests_deviation.md``):

* Pooled over the three core models, as RQ2: a cluster is a task within one model, the model a stratum; the
  bootstrap resamples tasks within each model (10,000 resamples).
* A cluster enters only with the sampled trials (seeds 5-8) that BF16, Q4_0 and the repair all finished. Per
  cluster: mean episode success and mean episode any-failure rate (failures / steps) over those trials. Pooled
  quantities are means over clusters.
* share_success = (S_repair - S_Q4_0) / (S_BF16 - S_Q4_0); share_failure = (F_Q4_0 - F_repair) / (F_Q4_0 - F_BF16);
  D = share_success - share_failure, predicted D > 0.
* Decision: bootstrap two-sided p for D = 0, p = 2 min(#(D* <= 0) + 1, #(D* >= 0) + 1) / (B + 1), capped at 1; Holm
  over the three; supported if Holm p < 0.05 and D > 0. The unadjusted 95% percentile CI is shown with it.
* All tasks (§5 limits the inclusion rule to the secondary episode-success analysis); the inclusion sets as a
  sensitivity. A gap whose point estimate is <= 0 makes the share undefined ("not testable"); replicates with a
  denominator <= 0 are counted, and the test is flagged unstable when they exceed 2.5%.

    python analysis/rq3_tests.py --dir <run dir> [--dir ...] [--inclusion results/m3/inclusion_holdout_<m>.json ...] \\
        --json results/m3/rq3_holdout.json --md docs/m3_rq3_holdout.md
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
from analysis.h1_tests import ALPHA, N_BOOT, holm, load_dir, sha  # noqa: E402

CORE_MODELS = ("qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct")
CONFIGS = {"BF16": ("bf16", "none"), "Q4_0": ("q4_0", "none"), "R1": ("q4_0", "r1"), "Q8_0": ("q8_0", "r2"),
           "QAD-KL+CE": ("r3_qad_klce", "r3")}
REPAIRS = ("R1", "Q8_0", "QAD-KL+CE")
UNSTABLE_SHARE = 0.025
SOURCES = ("analysis/rq3_tests.py", "analysis/h1_tests.py", "testbed/classifier.py")
KEY = ["benchmark", "task_id", "seed"]


def episode_table(steps: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    if len(set(CONFIGS.values())) != len(CONFIGS):
        raise ValueError("two configuration names map to the same (variant, repair)")
    agg = steps.groupby("run_id").agg(f_any=("fail", "sum"), n=("fail", "size")).reset_index()
    e = eps.merge(agg, on="run_id", how="left").fillna({"f_any": 0, "n": 0})
    e["config"] = None
    for name, (v, r) in CONFIGS.items():
        e.loc[(e.variant == v) & (e.repair == r) & (e.backend == "cuda"), "config"] = name
    e = e[e.config.notna() & (e.n > 0)]
    return e.assign(rate=e.f_any / e.n, success=e.valid.astype(float))


def triplet_clusters(ep, models, repair, seeds, tasks_by_model=None) -> tuple[pd.DataFrame, int]:
    """Per (model, benchmark, task): mean success and mean failure rate under BF16, Q4_0 and the repair, over the
    sampled trials all three finished; plus the number of such trials."""
    rows, n_trials = [], 0
    for m in models:
        sel = ep[(ep.model == m) & ep.seed.isin(seeds)]
        if tasks_by_model is not None:
            keep = tasks_by_model.get(m, set())
            sel = sel[[(b, t) in keep for b, t in zip(sel.benchmark, sel.task_id)]]
        parts = {c: sel[sel.config == c].set_index(KEY)[["success", "rate"]] for c in ("BF16", "Q4_0", repair)}
        idx = parts["BF16"].index.intersection(parts["Q4_0"].index).intersection(parts[repair].index)
        if len(idx) == 0:
            continue
        n_trials += len(idx)
        j = pd.concat({"bf": parts["BF16"].loc[idx], "q4": parts["Q4_0"].loc[idx], "rep": parts[repair].loc[idx]},
                      axis=1)
        j.columns = [f"{var}_{cfg}" for cfg, var in j.columns]          # e.g. ("bf", "success") -> success_bf
        g = j.reset_index().groupby(["benchmark", "task_id"]).mean(numeric_only=True).drop(columns=["seed"])
        rows.append(g.reset_index().assign(model=m))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), n_trials


COLS = ["success_bf", "success_q4", "success_rep", "rate_bf", "rate_q4", "rate_rep"]


def shares_from_means(M: np.ndarray) -> dict:
    """M[..., 6] = means of COLS; returns arrays of the gaps, the shares and D (broadcast over leading axes)."""
    sb, sq, sr, fb, fq, fr = (M[..., i] for i in range(6))
    gs, gf = sb - sq, fq - fb
    with np.errstate(divide="ignore", invalid="ignore"):
        ss, sf = (sr - sq) / gs, (fq - fr) / gf
    return {"gap_success": gs, "gap_failure": gf, "share_success": ss, "share_failure": sf, "D": ss - sf}


def gap_test(c: pd.DataFrame, rng: np.random.Generator) -> dict:
    X = c[COLS].values.astype(float)
    est = shares_from_means(X.mean(0))
    out = {k: float(v) for k, v in est.items()}
    out["clusters"] = int(len(c))
    out["means"] = dict(zip(COLS, map(float, X.mean(0))))
    groups = [np.flatnonzero(c.model.values == m) for m in pd.unique(c.model)]
    idx = np.concatenate([g[rng.integers(0, len(g), size=(N_BOOT, len(g)))] for g in groups], axis=1)
    bs = shares_from_means(X[idx].mean(1))
    bad = (bs["gap_success"] <= 0) | (bs["gap_failure"] <= 0)
    out["replicates_nonpositive_gap"] = float(bad.mean())
    out["unstable"] = bool(bad.mean() > UNSTABLE_SHARE)
    out["testable"] = bool(est["gap_success"] > 0 and est["gap_failure"] > 0)
    for k in ("gap_success", "gap_failure"):
        out[k + "_ci95"] = [float(x) for x in np.quantile(bs[k], [0.025, 0.975])]
    if not out["testable"]:
        out.update({"p": 1.0, "D_ci95": None, "share_success_ci95": None, "share_failure_ci95": None})
        return out
    D = bs["D"][~bad]
    out["D_ci95"] = [float(x) for x in np.quantile(D, [0.025, 0.975])]
    for k in ("share_success", "share_failure"):
        out[k + "_ci95"] = [float(x) for x in np.quantile(bs[k][~bad], [0.025, 0.975])]
    B = len(D)
    out["p"] = float(min(1.0, 2 * min(np.sum(D <= 0) + 1, np.sum(D >= 0) + 1) / (B + 1)))
    return out


def per_seed(ep, models, repair, seeds) -> dict:
    out = {}
    for c in ("BF16", "Q4_0", repair):
        s_vals, f_vals = [], []
        for s in seeds:
            e = ep[(ep.config == c) & ep.model.isin(models) & (ep.seed == s)]
            if len(e):
                s_vals.append(100 * e.success.mean())
                f_vals.append(100 * e.f_any.sum() / e.n.sum())
        out[c] = {"success_mean": float(np.mean(s_vals)), "success_sd": float(np.std(s_vals, ddof=1)),
                  "fail_per100_mean": float(np.mean(f_vals)), "fail_per100_sd": float(np.std(f_vals, ddof=1))}
    return out


def analyse(dirs, inclusion_files, seeds, models=CORE_MODELS, seed=295) -> dict:
    parts = [load_dir(d) for d in dirs]
    ep = episode_table(pd.concat([p[0] for p in parts], ignore_index=True),
                       pd.concat([p[1] for p in parts], ignore_index=True))
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
    rows = []
    for rep in REPAIRS:
        c, n = triplet_clusters(ep, models, rep, seeds)
        if c.empty:
            rows.append({"repair": rep, "missing": True, "p": 1.0})
            continue
        r = {"repair": rep, "trials": n, **gap_test(c, rng), "per_seed": per_seed(ep, models, rep, seeds)}
        r["per_model"] = {m: gap_test(g, rng) for m, g in c.groupby("model")}
        if inclusion is not None:
            ci, ni = triplet_clusters(ep, models, rep, seeds, inclusion)
            r["inclusion_sensitivity"] = {"trials": ni, **gap_test(ci, rng)} if not ci.empty else {"clusters": 0}
        rows.append(r)
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
        if r.get("missing"):
            r["verdict"] = "missing"
        elif not r["testable"]:
            r["verdict"] = "not testable (gap <= 0)"
        else:
            r["verdict"] = "supported" if p < ALPHA and r["D"] > 0 else "not supported"
            r["ci_excludes_0_predicted"] = bool(r["D_ci95"][0] > 0)
    return {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "sources_sha256_16": {p: sha(p) for p in SOURCES}, "runs": [p[2] for p in parts], "models": models,
            "seeds": list(seeds), "rng_seed": seed, "configs_present": sorted(ep.config.unique().tolist()),
            "inclusion": {m: len(v) for m, v in (inclusion or {}).items()},
            "method": "§6 RQ3 as registered; pooled over the core models (cluster task x model, model stratum); "
                      "trials finished under BF16, Q4_0 and the repair; per-cluster mean success and mean failure "
                      "rate; D = share of success gap closed - share of failure gap closed; model-stratified "
                      "bootstrap, 10,000 resamples; Holm over 3; all tasks (inclusion sets as sensitivity)",
            "tests": rows}


def _f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _p(x):
    return "–" if x is None else ("<0.0001" if x < 1e-4 else f"{x:.4f}")


def _ci(c, d=2):
    return "–" if not c else f"[{_f(c[0], d)}, {_f(c[1], d)}]"


def write_md(res, path, json_path, label) -> None:
    L = [f"# H3 registered tests, family RQ3: {label}\n",
         f"Generated {res['generated']} by `analysis/rq3_tests.py` (sources SHA-256/16: "
         + ", ".join(f"`{k}` {v}" for k, v in res["sources_sha256_16"].items()) + f"). JSON: `{json_path}`.\n",
         "Method: " + res["method"] + f". Models: {', '.join(res['models'])}. Sampled seeds {res['seeds']}.\n"]
    rows = res["tests"]
    L.append(f"## RQ3: {sum(r.get('verdict') == 'supported' for r in rows)} of {len(rows)} supported "
             "(Holm, alpha 0.05, D > 0)\n")
    L += ["| Repair | Clusters / trials | Success gap BF16 − Q4_0 (pts) [95% CI] | Failure gap Q4_0 − BF16 (per 100 "
          "steps) [95% CI] | Share of success gap closed [95% CI] | Share of failure gap closed [95% CI] | "
          "D [95% CI] | p | p (Holm) | Verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            L.append(f"| {r['repair']} | 0 | – | – | – | – | – | – | – | missing |")
            continue
        flag = " (unstable)" if r["unstable"] else ""
        L.append(f"| {r['repair']} | {r['clusters']} / {r['trials']} | {_f(100 * r['gap_success'], 1)} "
                 f"{_ci([100 * x for x in r['gap_success_ci95']], 1)} | {_f(100 * r['gap_failure'])} "
                 f"{_ci([100 * x for x in r['gap_failure_ci95']])} | {_f(r['share_success'])} {_ci(r['share_success_ci95'])} | "
                 f"{_f(r['share_failure'])} {_ci(r['share_failure_ci95'])} | {_f(r['D'])} {_ci(r['D_ci95'])} | "
                 f"{_p(r['p'])} | {_p(r['p_holm'])} | **{r['verdict']}**{flag} |")
    L.append("\nShares: 1 = the repair closes the whole BF16 → Q4_0 gap, 0 = none of it, < 0 = worse than Q4_0. "
             "Replicates with a non-positive gap are dropped from the share and D intervals (counted below).\n")
    L += ["### Rates by configuration (mean ± 1 SD over the four sampled seeds, three models pooled)\n",
          "| Repair | BF16 success % | Q4_0 success % | Repair success % | BF16 failures / 100 | Q4_0 failures / 100 | "
          "Repair failures / 100 | Replicates with gap ≤ 0 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            continue
        s = r["per_seed"]
        cell = lambda c, k: f"{_f(s[c][k + '_mean'], 1)} ± {_f(s[c][k + '_sd'], 1)}"
        L.append(f"| {r['repair']} | {cell('BF16', 'success')} | {cell('Q4_0', 'success')} | "
                 f"{cell(r['repair'], 'success')} | {cell('BF16', 'fail_per100')} | {cell('Q4_0', 'fail_per100')} | "
                 f"{cell(r['repair'], 'fail_per100')} | {_f(100 * r['replicates_nonpositive_gap'], 1)} % |")
    L += ["\n## Per-model estimates (report only)\n",
          "| Repair | Model | Clusters | Success gap (pts) | Failure gap (per 100) | Share success | Share failure | "
          "D [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for m, g in (r.get("per_model") or {}).items():
            L.append(f"| {r['repair']} | {m} | {g['clusters']} | {_f(100 * g['gap_success'], 1)} | "
                     f"{_f(100 * g['gap_failure'])} | {_f(g['share_success'])} | {_f(g['share_failure'])} | "
                     f"{_f(g['D'])} {_ci(g['D_ci95'])}{'' if g['testable'] else ' (gap ≤ 0)'} |")
    L += ["\n## Sensitivity: inclusion sets (tasks the BF16 reference solves at least once; report only)\n",
          "| Repair | Clusters | Success gap (pts) | Share success | Share failure | D [95% CI] | p (unadjusted) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r.get("inclusion_sensitivity")
        if not s or not s.get("clusters"):
            continue
        L.append(f"| {r['repair']} | {s['clusters']} | {_f(100 * s['gap_success'], 1)} | {_f(s['share_success'])} | "
                 f"{_f(s['share_failure'])} | {_f(s['D'])} {_ci(s['D_ci95'])} | {_p(s['p'])} |")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--inclusion", action="append", default=[])
    ap.add_argument("--seeds", default="5,6,7,8")
    ap.add_argument("--config-map", help="JSON {name: [variant, repair]} overriding CONFIGS (development check only)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--label", default="confirmatory holdout (tests implemented after registration)")
    ap.add_argument("--rng-seed", type=int, default=295)
    a = ap.parse_args()
    if a.config_map:
        CONFIGS.clear()
        CONFIGS.update({k: tuple(v) for k, v in json.loads(a.config_map).items()})
    res = analyse(a.dir, a.inclusion, tuple(int(s) for s in a.seeds.split(",")), seed=a.rng_seed)
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
