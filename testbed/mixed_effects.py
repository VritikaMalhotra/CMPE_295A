"""C10: run the registered mixed-effects models (analysis/mixed_effects.R) and check them on planted effects.

* ``run_mixed_effects`` - export the needed columns of the C5 step/episode tables to CSV, call the R script
                          with ``Rscript`` and return its JSON as a dict.
* ``simulate``          - step and episode tables in the C5 schema with known effects of 4-bit precision.
* ``planted_effect_check`` - simulate, fit, and check that each planted effect is recovered (95% CI covers it,
                          p < 0.05) and that a no-effect dataset is not flagged.

R 4.4 with lme4 must be on PATH (cluster: ``module load gnu14 R/4.4.1`` and
``R_LIBS_USER=/scratch/cmpe295-bertin/envs/R-4.4.1-lib``). ``RSCRIPT`` overrides the executable.
On the cluster use a shell *without* ``scripts/hpc_env.sh``: its NVIDIA HPC SDK module blocks the gnu14
compiler module, and lme4's Rcpp then fails to load (GLIBCXX_3.4.32 missing from the system libstdc++).
Checked 2026-11-08: ``. /etc/profile.d/lmod.sh; module purge; module load gnu14 R/4.4.1`` -> 3 tests pass.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_SCRIPT = os.path.join(REPO, "analysis", "mixed_effects.R")

STEP_COLS = ["run_id", "task_id", "seed", "variant", "backend", "repair", "constraint", "benchmark", "config_key",
             "turn", "step", "n_tool_call_blocks", "failure_class", "failure", "step_type", "turn_depth", "precision"]
EPISODE_COLS = ["run_id", "task_id", "seed", "variant", "backend", "repair", "constraint", "config_key",
                "valid", "success", "n_steps"]


def rscript() -> str | None:
    return os.environ.get("RSCRIPT") or shutil.which("Rscript")


def lme4_available() -> bool:
    exe = rscript()
    if not exe:
        return False
    r = subprocess.run([exe, "-e", "quit(status = !requireNamespace('lme4', quietly = TRUE))"],
                       capture_output=True)
    return r.returncode == 0


def _load(t) -> pd.DataFrame:
    if isinstance(t, pd.DataFrame):
        return t
    return pd.read_parquet(t) if str(t).endswith(".parquet") else pd.read_csv(t)


# Registered depth bins (decision 8, Ekant 2026-11-08; advisor to confirm): the step's index within the episode,
# 0-1, 2-3, 4-6, 7+ (about equal quarters of the Oct 28 BF16 steps for all three Qwen3 sizes).
DEPTH_BINS = ((0, "d0-1"), (2, "d2-3"), (4, "d4-6"), (7, "d7+"))


def depth_bin(episode_step: int) -> str:
    label = DEPTH_BINS[0][1]
    for lo, name in DEPTH_BINS:
        if episode_step >= lo:
            label = name
    return label


def with_depth_bins(steps: pd.DataFrame) -> pd.DataFrame:
    """Adds ``turn_depth`` = the registered bin of each step's index within its episode (C5 rows are per
    episode in (turn, step) order; the analysis column keeps the R script's name)."""
    s = steps.sort_values(["run_id", "turn", "step"]).copy()
    s["turn_depth"] = s.groupby("run_id").cumcount().map(depth_bin)
    return s


def run_mixed_effects(steps, episodes, out: str | None = None, configuration: str = "variant",
                      ref: str | None = None, filters: dict | None = None, depth_breaks=(0, 1, 2, 4),
                      dispersion_threshold: float = 1.5, models: str = "A,B,C", timeout: int = 3600,
                      depth: str = "episode_step") -> dict:
    """steps/episodes: DataFrames or paths to steps.parquet / episodes.parquet (or CSV). Returns the JSON.
    ``depth="episode_step"`` (registered) bins each step's index within the episode (``DEPTH_BINS``);
    ``depth="turn"`` lets the R script bin the turn index at ``depth_breaks`` instead."""
    exe = rscript()
    if not exe:
        raise RuntimeError("Rscript not found (cluster: module load gnu14 R/4.4.1)")
    s, e = _load(steps), _load(episodes)
    if depth == "episode_step" and "turn_depth" not in s:
        s = with_depth_bins(s)
    with tempfile.TemporaryDirectory() as d:
        sp, ep = os.path.join(d, "steps.csv"), os.path.join(d, "episodes.csv")
        s[[c for c in STEP_COLS if c in s]].to_csv(sp, index=False)
        e[[c for c in EPISODE_COLS + [configuration] if c in e]].loc[:, lambda x: ~x.columns.duplicated()] \
            .to_csv(ep, index=False)
        out_path = out or os.path.join(d, "mixed_effects.json")
        cmd = [exe, R_SCRIPT, "--steps", sp, "--episodes", ep, "--out", out_path,
               "--configuration", configuration, "--depth-breaks", ",".join(map(str, depth_breaks)),
               "--dispersion-threshold", str(dispersion_threshold), "--models", models]
        if ref:
            cmd += ["--ref", ref]
        for k, v in (filters or {}).items():
            cmd += ["--filter", f"{k}={v}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"mixed_effects.R failed ({r.returncode}):\n{r.stderr[-4000:]}")
        with open(out_path) as f:
            return json.load(f)


def simulate(n_tasks: int = 120, n_seeds: int = 4, beta_4bit: float = 0.6, beta_4bit_text: float = -0.4,
             beta_4bit_deep: float = 0.4, log_or_success: float = -0.8, seed: int = 0):
    """Two configurations (bf16 reference, q4_0) on the same tasks and seeds, in the C5 schema.

    Step failure is logit-linear (the model B form): base + task + episode intercepts + step type + depth bin
    + 4-bit x (main + text-step interaction + deepest-bin interaction). "end" steps never fail, as under
    classifier v0. Episode success is independent, logit-linear with a task intercept (the model C form).
    Returns (steps, episodes, truth); truth.A is the rate ratio implied by the simulated design.
    """
    rng = np.random.default_rng(seed)
    a_task = rng.normal(0, 0.6, n_tasks)          # step-failure task intercepts
    c_task = rng.normal(0, 1.0, n_tasks)          # success task intercepts
    depth_eff = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}  # bins t0, t1, t2-3, t4+
    steps, eps, p_ref, p_q4 = [], [], [], []
    for t in range(n_tasks):
        for s in range(n_seeds):
            for variant in ("bf16", "q4_0"):
                q4 = variant == "q4_0"
                rid = f"{variant}-t{t}-s{s}"
                e_ep = rng.normal(0, 0.4)
                n = 0
                for turn in range(int(rng.integers(2, 8))):
                    dbin = 0 if turn == 0 else 1 if turn == 1 else 2 if turn <= 3 else 3
                    for k in range(int(rng.integers(1, 5))):
                        kind = rng.choice(["call", "text", "end"], p=[0.6, 0.3, 0.1])
                        if kind == "end":
                            fail, pr, pq = False, 0.0, 0.0
                        else:
                            eta = -2.5 + a_task[t] + e_ep + (-0.3 if kind == "text" else 0.0) + depth_eff[dbin]
                            dq = beta_4bit + (beta_4bit_text if kind == "text" else 0.0) + \
                                (beta_4bit_deep if dbin == 3 else 0.0)
                            pr, pq = 1 / (1 + np.exp(-eta)), 1 / (1 + np.exp(-(eta + dq)))
                            fail = rng.random() < (pq if q4 else pr)
                        p_ref.append(pr)
                        p_q4.append(pq)
                        fc = {"call": "invalid_args" if fail else "ok_call",
                              "text": "missing_call" if fail else "reply", "end": "end_turn"}[kind]
                        steps.append({"run_id": rid, "task_id": f"t{t}", "seed": s, "variant": variant,
                                      "turn": turn, "step": k, "n_tool_call_blocks": int(kind == "call"),
                                      "failure_class": fc})
                        n += 1
                succ = rng.random() < 1 / (1 + np.exp(-(0.5 + c_task[t] + (log_or_success if q4 else 0.0))))
                eps.append({"run_id": rid, "task_id": f"t{t}", "seed": s, "variant": variant,
                            "valid": bool(succ), "n_steps": n})
    truth = {
        "A": {"configurationq4_0": float(np.log(np.sum(p_q4) / np.sum(p_ref)))},
        "B": {"precision4bit": beta_4bit, "precision4bit:step_typetext": beta_4bit_text,
              "precision4bit:turn_deptht4+": beta_4bit_deep},
        "C": {"configurationq4_0": log_or_success},
    }
    return pd.DataFrame(steps), pd.DataFrame(eps), truth


def _check(res: dict, truth: dict) -> dict:
    out = {}
    for m, terms in truth.items():
        model = res["models"][m]
        fixed = {r["term"]: r for r in model.get("fixed", [])}
        for term, true in terms.items():
            r = fixed.get(term)
            out[f"{m}:{term}"] = {
                "truth": true, "estimate": r and r["estimate"], "ci": r and [r["ci_low"], r["ci_high"]],
                "p": r and r["p"], "covered": bool(r) and r["ci_low"] <= true <= r["ci_high"],
                "significant": bool(r) and r["p"] < 0.05}
    return out


def planted_effect_check(seed: int = 0, **sim) -> dict:
    """Fit the planted dataset and a no-effect dataset; every planted term must be covered and significant,
    and no null term may be significant."""
    s, e, truth = simulate(seed=seed, **sim)
    res = run_mixed_effects(s, e, depth="turn")       # the simulation plants its effect by turn
    terms = _check(res, truth)
    s0, e0, truth0 = simulate(seed=seed + 1, beta_4bit=0.0, beta_4bit_text=0.0, beta_4bit_deep=0.0,
                              log_or_success=0.0, **{k: v for k, v in sim.items() if k in ("n_tasks", "n_seeds")})
    res0 = run_mixed_effects(s0, e0, depth="turn")
    null = _check(res0, truth0)
    return {
        "terms": terms, "null_terms": null,
        "status": {m: res["models"][m]["status"] for m in res["models"]},
        "family_A": res["models"]["A"]["family"], "dispersion_A": res["models"]["A"]["poisson_dispersion"],
        "notes": res["notes"],
        "recovered": all(v["covered"] and v["significant"] for v in terms.values()),
        "null_false_positive": any(v["significant"] for v in null.values()),
        "null_covered": all(v["covered"] for v in null.values()),
        "result": res,
    }
