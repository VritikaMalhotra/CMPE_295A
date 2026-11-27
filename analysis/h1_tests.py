#!/usr/bin/env python
"""Registered H1 tests (pre-registration https://osf.io/xwq5y, §6), written after registration.

The code frozen at registration (commit 1e13897, §7) has the effect sizes (``analysis/damage_table.py``) and the
mixed-effects models, but not the H1 tests themselves. This module adds them exactly as §6 states and imports the
frozen pieces instead of copying them (``damage_table.ratio_ci``, ``classifier.classify_step``,
``mixed_effects.depth_bin``). It was written and checked on development data and planted effects before it read
any holdout cell (deviation note: ``docs/m3_h1_tests_deviation.md``).

Families (Holm within each, two-sided alpha = 0.05; a direction is checked on the estimate):

* ``H1``      18 tests: model x format (CUDA), paired permutation test (10,000 sign flips) on per-task any-failure
              rates, BF16 vs 4-bit. Supported if Holm p < 0.05 and the ratio of pooled rates > 1.
* ``H1-tep``  15 tests: the same for ``tool_error_payload``, five models (not SmolLM3-3B, underpowered, §3).
* ``H1-cpu``   6 tests: Qwen3-0.6B and 1.7B x format, CPU, greedy pass, paired permutation over tasks; plus the
              CUDA vs CPU difference in the BF16 -> 4-bit change with a bootstrap CI over tasks (no direction).
* ``H1-loc``   6 tests: per format, a Poisson GEE of step failures on precision x step type and precision x depth
              bin, pooled over models (model as a fixed effect), clustered by task, independence working
              correlation, robust (sandwich) covariance. Step type: Wald z on precision x call; depth: joint Wald
              chi-square (3 df) on precision x depth. Supported if Holm p < 0.05 and the interaction is larger in
              tool-call steps (coefficient > 0) / in the deepest bin (precision x d7+ > 0).
* ``H1-sens`` 18 tests: the H1 family on the 200 miss_func and miss_param holdout tasks (§8).

Per-task rate (§4: "sampled trials are averaged within a task before paired tests"): the mean over the task's
paired trials of (failures / steps) of each episode. Pairing (§5): a trial enters a contrast only when both
configurations finished it (benchmark, task, seed); pair counts are reported. Every step is relabelled with the
frozen classifier (codebook v0), as the frozen damage table does.

    python analysis/h1_tests.py --dir <run dir> [--dir ...] --seeds 5,6,7,8 --greedy-seed 0 \\
        --json results/m3/h1_holdout.json --md docs/m3_h1_holdout.md [--label "..."]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from analysis.damage_table import ratio_ci  # noqa: E402  (frozen)
from testbed.classifier import FAILURES, classify_step  # noqa: E402  (frozen)
from testbed.mixed_effects import depth_bin  # noqa: E402  (frozen)
from testbed.spec import StepRow  # noqa: E402

N_PERM = 10_000
N_BOOT = 10_000
ALPHA = 0.05
FORMATS = ("q4_0", "q4_k_m", "iq4_xs")
REF = "bf16"
TEP = "tool_error_payload"
TEP_EXCLUDED_MODELS = ("smollm3-3b",)                 # §3/§4: tool_error_payload not powered for SmolLM3-3B
CPU_MODELS = ("qwen3-0.6b", "qwen3-1.7b")
CALL_LABELS = {"malformed", "unknown_tool", "invalid_args", "exec_error", "tool_error_payload", "ok_call"}
DEPTHS = ("d0-1", "d2-3", "d4-6", "d7+")
SENS_BENCHMARKS = ("bfcl_multi_turn_miss_func", "bfcl_multi_turn_miss_param")
FIELDS = set(StepRow.__dataclass_fields__)
LIST_COLS = ("decoded_calls", "execution_results", "schema_errors")
SOURCES = ("analysis/h1_tests.py", "analysis/damage_table.py", "testbed/classifier.py", "testbed/mixed_effects.py")


# ---------------------------------------------------------------- statistics (pure functions, tested on their own)

def perm_test_paired(d: np.ndarray, n_perm: int = N_PERM, rng: np.random.Generator | None = None) -> dict:
    """Two-sided paired permutation test of mean(d) = 0 by random sign flips of the per-task differences.
    p = (1 + #{|mean perm| >= |mean obs|}) / (n_perm + 1)."""
    d = np.asarray(d, float)
    rng = rng or np.random.default_rng(0)
    obs = float(d.mean()) if len(d) else 0.0
    if len(d) == 0 or not np.any(d):
        return {"mean_diff": obs, "p": 1.0, "n_tasks": int(len(d)), "n_perm": n_perm}
    signs = rng.integers(0, 2, size=(n_perm, len(d)), dtype=np.int8) * 2 - 1
    null = (signs @ d) / len(d)
    p = (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / (n_perm + 1)
    return {"mean_diff": obs, "p": float(p), "n_tasks": int(len(d)), "n_perm": n_perm}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def poisson_gee(y: np.ndarray, X: np.ndarray, groups: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Poisson GEE (log link) with an independence working correlation: the Poisson GLM estimate by Newton-IRLS
    and the robust sandwich covariance summed over clusters (no small-sample correction; the statsmodels GEE
    default). ``X`` must hold an intercept column first."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-12))
    converged = False
    for it in range(max_iter):
        mu = np.exp(X @ beta)
        H = X.T @ (X * mu[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    mu = np.exp(X @ beta)
    bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    codes, uniq = pd.factorize(pd.Series(groups))
    U = np.zeros((len(uniq), X.shape[1]))
    np.add.at(U, codes, X * (y - mu)[:, None])
    cov = bread @ (U.T @ U) @ bread
    return {"beta": beta, "cov": cov, "n_clusters": int(len(uniq)), "n_obs": int(len(y)), "iterations": it + 1,
            "converged": converged}


def wald(beta: np.ndarray, cov: np.ndarray, idx: list[int]) -> dict:
    """Wald test that beta[idx] = 0: z (one coefficient) or chi-square (several)."""
    from scipy import stats
    b = beta[idx]
    V = cov[np.ix_(idx, idx)]
    if len(idx) == 1:
        se = float(np.sqrt(V[0, 0]))
        z = float(b[0] / se)
        return {"estimate": float(b[0]), "se": se, "z": z, "p": float(2 * stats.norm.sf(abs(z))),
                "rate_ratio": float(np.exp(b[0])),
                "rr_ci95": [float(np.exp(b[0] - 1.959963984540054 * se)), float(np.exp(b[0] + 1.959963984540054 * se))]}
    chi2 = float(b @ np.linalg.solve(V, b))
    return {"estimates": b.tolist(), "chi2": chi2, "df": len(idx), "p": float(stats.chi2.sf(chi2, len(idx))),
            "rate_ratios": np.exp(b).tolist()}


# ---------------------------------------------------------------- loading

def _relabel(df: pd.DataFrame) -> list[str]:
    out = []
    for r in df.to_dict("records"):
        for c in LIST_COLS:
            if isinstance(r.get(c), str):
                r[c] = json.loads(r[c])
        out.append(classify_step(StepRow(**{k: r.get(k) for k in FIELDS if k in r})))
    return out


def load_dir(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(steps, episodes, info) of the finished episodes of one run directory, from its Parquet tables (rebuilt by the
    frozen report from the per-episode files). Steps carry the relabelled class, step type and depth bin."""
    specs = pd.read_json(os.path.join(run_dir, "specs.jsonl"), lines=True)
    eps = pd.read_parquet(os.path.join(run_dir, "episodes.parquet"))
    steps = pd.read_parquet(os.path.join(run_dir, "steps.parquet"))
    sp = specs[["run_id", "model", "benchmark"]]
    eps = eps.merge(sp, on="run_id", how="inner")
    steps = steps.drop(columns=[c for c in ("benchmark",) if c in steps]).merge(sp, on="run_id", how="inner")
    steps = steps[steps.run_id.isin(eps.run_id)].copy()
    labels = _relabel(steps)
    info = {"dir": run_dir, "episodes": int(len(eps)), "steps": int(len(steps)),
            "label_changed_vs_logged": int(sum(a != b for a, b in zip(labels, steps.failure_class)))}
    steps["label"] = labels
    steps["fail"] = steps.label.isin(FAILURES)
    steps["tep"] = steps.label == TEP
    steps["call"] = (steps.n_tool_call_blocks.fillna(0) > 0) | steps.label.isin(CALL_LABELS)
    steps = steps.sort_values(["run_id", "turn", "step"])
    steps["depth"] = steps.groupby("run_id").cumcount().map(depth_bin)
    mp = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        info.update({k: m.get(k) for k in ("experiment", "experiment_hash", "testbed_git")})
    keep = ["run_id", "model", "variant", "backend", "repair", "benchmark", "task_id", "seed", "turn", "step",
            "label", "fail", "tep", "call", "depth"]
    return steps[keep], eps[["run_id", "model", "variant", "backend", "repair", "benchmark", "task_id", "seed",
                             "n_steps", "valid"]], info


def episode_table(steps: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """One row per finished episode: failures (any, tool_error_payload) and steps."""
    agg = steps.groupby("run_id").agg(f_any=("fail", "sum"), f_tep=("tep", "sum"), n=("fail", "size")).reset_index()
    e = eps.merge(agg, on="run_id", how="left").fillna({"f_any": 0, "f_tep": 0, "n": 0})
    return e


# ---------------------------------------------------------------- contrasts

KEY = ["benchmark", "task_id", "seed"]


def paired(ep: pd.DataFrame, model: str, backend: str, variant: str, seeds, benchmarks=None) -> pd.DataFrame:
    sel = ep[(ep.model == model) & (ep.backend == backend) & (ep.repair == "none") & ep.seed.isin(seeds)]
    if benchmarks is not None:
        sel = sel[sel.benchmark.isin(benchmarks)]
    a, b = sel[sel.variant == REF], sel[sel.variant == variant]
    j = a.merge(b, on=KEY, suffixes=("_ref", "_q"))
    return j[(j.n_ref > 0) & (j.n_q > 0)]


def contrast(j: pd.DataFrame, outcome: str, rng: np.random.Generator, seeds) -> dict:
    """Permutation test on per-task mean episode rates, plus the frozen pooled-rate ratio with its bootstrap CI."""
    f = "f_any" if outcome == "any_failure" else "f_tep"
    j = j.assign(r_ref=j[f + "_ref"] / j.n_ref, r_q=j[f + "_q"] / j.n_q)
    t = j.groupby(["benchmark", "task_id"]).agg(r_ref=("r_ref", "mean"), r_q=("r_q", "mean"),
                                                f_ref=(f + "_ref", "sum"), n_ref=("n_ref", "sum"),
                                                f_q=(f + "_q", "sum"), n_q=("n_q", "sum"))
    out = perm_test_paired((t.r_q - t.r_ref).values, rng=rng)
    out.update({"pairs": int(len(j)), "tasks": int(len(t)), "task_mean_rate_ref": float(t.r_ref.mean()),
                "task_mean_rate_4bit": float(t.r_q.mean())})
    # report-only sensitivity (declared before the holdout run): the same test on the within-task pooled rate
    # (failures / steps summed over the task's paired trials), which weights long episodes more; never decides
    sens = perm_test_paired((t.f_q / t.n_q - t.f_ref / t.n_ref).values, rng=rng)
    out["sensitivity_task_pooled_rate"] = {"mean_diff": sens["mean_diff"], "p": sens["p"]}
    out["pooled"] = ratio_ci(t.f_ref.values.astype(float), t.n_ref.values.astype(float),
                             t.f_q.values.astype(float), t.n_q.values.astype(float), rng)
    per_seed = []
    for s in seeds:
        js = j[j.seed == s]
        if len(js):
            per_seed.append((100 * js[f + "_ref"].sum() / js.n_ref.sum(), 100 * js[f + "_q"].sum() / js.n_q.sum()))
    if len(per_seed) > 1:
        a = np.array(per_seed)
        out["per_seed_per100"] = {"ref": a[:, 0].tolist(), "4bit": a[:, 1].tolist(),
                                  "ref_mean": float(a[:, 0].mean()), "ref_sd": float(a[:, 0].std(ddof=1)),
                                  "4bit_mean": float(a[:, 1].mean()), "4bit_sd": float(a[:, 1].std(ddof=1))}
    else:                                   # greedy: one trial, SD = 0; Wilson on the pooled step proportion
        out["wilson95_step_rate"] = {"ref": wilson(int(t.f_ref.sum()), int(t.n_ref.sum())),
                                     "4bit": wilson(int(t.f_q.sum()), int(t.n_q.sum()))}
    return out


def family(rows: list[dict], name: str, ratio_key: str = "pooled") -> dict:
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        ratio = r[ratio_key]["ratio"]
        r["supported"] = bool(a < ALPHA and ratio is not None and ratio > 1)
    return {"family": name, "m": len(rows), "n_supported": int(sum(r["supported"] for r in rows)), "tests": rows}


def damage_family(ep, models, seeds, outcome, rng, benchmarks=None, backend="cuda") -> list[dict]:
    rows = []
    for m in models:
        for v in FORMATS:
            j = paired(ep, m, backend, v, seeds, benchmarks)
            if j.empty:
                rows.append({"model": m, "format": v, "pairs": 0, "p": 1.0, "pooled": {"ratio": None},
                             "missing": True})
                continue
            rows.append({"model": m, "format": v, **contrast(j, outcome, rng, seeds)})
    return rows


def cpu_vs_cuda(ep, model, variant, seed, rng) -> dict:
    """Per task: (4-bit - BF16 on CPU) - (4-bit - BF16 on CUDA), greedy pass; mean with a bootstrap CI over tasks."""
    def delta(backend):
        j = paired(ep, model, backend, variant, [seed], ["bfcl_multi_turn_base"])
        j = j.assign(d=j.f_any_q / j.n_q - j.f_any_ref / j.n_ref)
        return j.set_index(["benchmark", "task_id"]).d
    dc, dg = delta("cpu"), delta("cuda")
    both = dc.index.intersection(dg.index)
    if len(both) == 0:
        return {"tasks": 0}
    x = (dc.loc[both] - dg.loc[both]).values
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"tasks": int(len(x)), "mean_diff_of_change": float(x.mean()), "ci95": [float(lo), float(hi)],
            "cpu_change": float(dc.loc[both].mean()), "cuda_change": float(dg.loc[both].mean())}


def loc_design(s: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, list[str]]:
    q = (s.variant != REF).values.astype(float)
    call = s.call.values.astype(float)
    cols, names = [np.ones(len(s)), q, call], ["intercept", "prec4", "call"]
    for d in DEPTHS[1:]:
        cols.append((s.depth == d).values.astype(float)); names.append(f"depth_{d}")
    for m in models[1:]:
        cols.append((s.model == m).values.astype(float)); names.append(f"model_{m}")
    cols.append(q * call); names.append("prec4:call")
    for d in DEPTHS[1:]:
        cols.append(q * (s.depth == d).values); names.append(f"prec4:depth_{d}")
    return np.column_stack(cols), names


def loc_family(steps: pd.DataFrame, ep: pd.DataFrame, models, seeds) -> list[dict]:
    rows = []
    for v in FORMATS:
        keep = []
        for m in models:                                          # paired trials only (§5), as everywhere else
            j = paired(ep, m, "cuda", v, seeds)
            keep += j.run_id_ref.tolist() + j.run_id_q.tolist()
        s = steps[steps.run_id.isin(set(keep))]
        if s.empty:
            continue
        X, names = loc_design(s, list(models))
        g = poisson_gee(s.fail.values.astype(float), X, (s.benchmark + "/" + s.task_id).values)
        i_call = [names.index("prec4:call")]
        i_dep = [names.index(f"prec4:depth_{d}") for d in DEPTHS[1:]]
        wc, wd = wald(g["beta"], g["cov"], i_call), wald(g["beta"], g["cov"], i_dep)
        base = {"format": v, "steps": g["n_obs"], "clusters": g["n_clusters"], "converged": g["converged"],
                "coefficients": dict(zip(names, map(float, g["beta"])))}
        rows.append({**base, "test": "precision x step type", **wc, "direction_ok": wc["estimate"] > 0,
                     "pooled": {"ratio": wc["rate_ratio"]}})
        rows.append({**base, "test": "precision x depth", **wd,
                     "direction_ok": wd["estimates"][-1] > 0,
                     "pooled": {"ratio": float(np.exp(wd["estimates"][-1]))}})
    return rows


# ---------------------------------------------------------------- driver

def sha(path: str) -> str:
    return hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()[:16]


def analyse(dirs: list[str], seeds: tuple, greedy_seed: int | None, seed: int = 295) -> dict:
    parts = [load_dir(d) for d in dirs]
    steps = pd.concat([p[0] for p in parts], ignore_index=True)
    ep = episode_table(steps, pd.concat([p[1] for p in parts], ignore_index=True))
    models = sorted(ep[(ep.backend == "cuda")].model.unique())
    rng = np.random.default_rng(seed)
    res = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "sources_sha256_16": {p: sha(p) for p in SOURCES}, "runs": [p[2] for p in parts],
           "seeds": list(seeds), "greedy_seed": greedy_seed, "models": models, "rng_seed": seed,
           "method": "§6 as registered; per-task rate = mean over paired trials of failures/steps; paired "
                     "permutation (10,000 sign flips), Holm within family; pooled ratio and paired cluster "
                     "bootstrap CI from the frozen damage_table.ratio_ci; classifier codebook v0", "families": {}}
    fam = res["families"]
    fam["H1"] = family(damage_family(ep, models, seeds, "any_failure", rng), "H1")
    tep_models = [m for m in models if m not in TEP_EXCLUDED_MODELS]
    fam["H1-tep"] = family(damage_family(ep, tep_models, seeds, TEP, rng), "H1-tep")
    rows = loc_family(steps, ep, models, seeds)
    fl = family(rows, "H1-loc")
    for r in fl["tests"]:
        r["supported"] = bool(r["p_holm"] < ALPHA and r["direction_ok"])
    fl["n_supported"] = int(sum(r["supported"] for r in fl["tests"]))
    fam["H1-loc"] = fl
    if greedy_seed is not None and (ep.backend == "cpu").any():
        cpu_models = [m for m in CPU_MODELS if m in set(ep[ep.backend == "cpu"].model)]
        fc = family(damage_family(ep, cpu_models, [greedy_seed], "any_failure", rng,
                                  ["bfcl_multi_turn_base"], backend="cpu"), "H1-cpu")
        for r in fc["tests"]:
            r["cpu_vs_cuda"] = cpu_vs_cuda(ep, r["model"], r["format"], greedy_seed, rng)
        fam["H1-cpu"] = fc
    if set(SENS_BENCHMARKS) & set(ep.benchmark):
        fam["H1-sens"] = family(damage_family(ep, models, seeds, "any_failure", rng, SENS_BENCHMARKS), "H1-sens")
    return res


def _f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _p(x):
    return "–" if x is None else ("<0.0001" if x < 1e-4 else f"{x:.4f}")


def write_md(res: dict, path: str, json_path: str, label: str) -> None:
    L = [f"# H1 registered tests: {label}\n",
         f"Generated {res['generated']} by `analysis/h1_tests.py` (sources SHA-256/16: "
         + ", ".join(f"`{k}` {v}" for k, v in res["sources_sha256_16"].items()) + f"). JSON: `{json_path}`.\n",
         "Method: " + res["method"] + f". Sampled seeds {res['seeds']}; greedy seed {res['greedy_seed']}.\n"]
    for key in ("H1", "H1-tep", "H1-sens", "H1-cpu"):
        f = res["families"].get(key)
        if not f:
            continue
        L.append(f"## {key}: {f['n_supported']} of {f['m']} supported (Holm, alpha 0.05, ratio > 1)\n")
        greedy = key == "H1-cpu"
        hdr = ("| Model | Format | Tasks / pairs | BF16 per 100 steps | 4-bit per 100 steps | Ratio [95% CI] | p | p (Holm) "
               "| Supported | p, task-pooled rate (report only) |")
        if greedy:
            hdr += " CPU − CUDA change [95% CI] |"
        L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for r in f["tests"]:
            if r.get("missing"):
                L.append(f"| {r['model']} | {r['format']} | 0 | – | – | – | – | – | missing | – |" + (" – |" if greedy else ""))
                continue
            po = r["pooled"]
            ci = po.get("ratio_ci95")
            if greedy:
                w = r["wilson95_step_rate"]
                ref = f"{_f(po['rate_ref_per100'])} [{_f(100 * w['ref'][0])}, {_f(100 * w['ref'][1])}]"
                q = f"{_f(po['rate_4bit_per100'])} [{_f(100 * w['4bit'][0])}, {_f(100 * w['4bit'][1])}]"
            else:
                s = r["per_seed_per100"]
                ref, q = (f"{_f(s['ref_mean'])} ± {_f(s['ref_sd'])}", f"{_f(s['4bit_mean'])} ± {_f(s['4bit_sd'])}")
            row = (f"| {r['model']} | {r['format']} | {r['tasks']} / {r['pairs']} | {ref} | {q} | {_f(po['ratio'])} "
                   f"[{_f(ci[0]) if ci else '–'}, {_f(ci[1]) if ci else '–'}] | {_p(r['p'])} | {_p(r['p_holm'])} | "
                   f"{'**yes**' if r['supported'] else 'no'} | {_p(r['sensitivity_task_pooled_rate']['p'])} |")
            if greedy:
                c = r["cpu_vs_cuda"]
                row += (f" {_f(100 * c['mean_diff_of_change'])} [{_f(100 * c['ci95'][0])}, {_f(100 * c['ci95'][1])}] |"
                        if c.get("tasks") else " – |")
            L.append(row)
        L.append("\nRates: sampled seeds, mean ± 1 SD of the per-seed pooled rate; greedy, pooled rate with a 95% "
                 "Wilson interval on the step proportion (SD = 0). CPU − CUDA: per 100 steps, per-task means.\n")
    f = res["families"].get("H1-loc")
    if f:
        L.append(f"## H1-loc: {f['n_supported']} of {f['m']} supported (Holm, alpha 0.05, direction)\n")
        L += ["| Format | Test | Steps / clusters | Statistic | Interaction RR (call; d7+) | p | p (Holm) | Direction | Supported |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in f["tests"]:
            stat = f"z = {_f(r['z'])}" if "z" in r else f"χ²({r['df']}) = {_f(r['chi2'])}"
            rr = (f"{_f(r['rate_ratio'])} [{_f(r['rr_ci95'][0])}, {_f(r['rr_ci95'][1])}]" if "rate_ratio" in r
                  else ", ".join(f"{d} {_f(x)}" for d, x in zip(DEPTHS[1:], r["rate_ratios"])))
            L.append(f"| {r['format']} | {r['test']} | {r['steps']} / {r['clusters']} | {stat} | {rr} | {_p(r['p'])} | "
                     f"{_p(r['p_holm'])} | {'ok' if r['direction_ok'] else 'opposite'} | "
                     f"{'**yes**' if r['supported'] else 'no'} |")
        L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--seeds", default="5,6,7,8")
    ap.add_argument("--greedy-seed", type=int, default=0)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--label", default="confirmatory holdout (tests implemented after registration)")
    ap.add_argument("--rng-seed", type=int, default=295)
    a = ap.parse_args()
    res = analyse(a.dir, tuple(int(s) for s in a.seeds.split(",")), a.greedy_seed, a.rng_seed)
    res["label"] = a.label
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, default=float)
        f.write("\n")
    if a.md:
        write_md(res, a.md, a.json, a.label)
    print("wrote", a.json, a.md or "")


if __name__ == "__main__":
    main()
