#!/usr/bin/env python
"""C10: one run of the registered mixed-effects models on real logs: a BF16 reference against one 4-bit cell.

Reads ``steps.parquet`` / ``episodes.parquet`` of both run directories (the report job writes them), keeps the
sampled trials (seeds 1-4; the greedy pass is analysed separately, pre-registration §4), labels each task by
benchmark (tau2 task ids repeat across domains), and fits models A/B/C of ``analysis/mixed_effects.R`` with the
reference variant as baseline. Writes the R script's JSON plus provenance (run directories, experiment hashes,
row counts, R and lme4 versions) to ``--out``. Development data: the pre-registration is not filed.

    python scripts/mixed_effects_real.py --ref $TESTBED_OUT/ref_qwen3_0p6b/cd314594fb \\
        --cell $TESTBED_OUT/h1_qwen3_0p6b_q4_0/c200505b35 --out results/m2/mixed_effects_qwen3-0.6b.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testbed import mixed_effects  # noqa: E402

SAMPLED_SEEDS = (1, 2, 3, 4)


def load(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_parquet(os.path.join(run_dir, "steps.parquet"))
    e = pd.read_parquet(os.path.join(run_dir, "episodes.parquet"))
    bench = s.groupby("run_id")["benchmark"].first()
    e = e.assign(benchmark=e["run_id"].map(bench))
    return s, e


def prepare(ref_dir: str, cell_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts = [load(ref_dir), load(cell_dir)]
    s = pd.concat([p[0] for p in parts], ignore_index=True)
    e = pd.concat([p[1] for p in parts], ignore_index=True)
    s = s[s["seed"].isin(SAMPLED_SEEDS)].copy()
    e = e[e["seed"].isin(SAMPLED_SEEDS)].copy()
    for df in (s, e):
        df["task_id"] = df["benchmark"].fillna("?").astype(str) + ":" + df["task_id"].astype(str)
    # pair on (task, seed): keep only episodes both configurations finished
    keys = e.groupby(["task_id", "seed"])["variant"].nunique()
    both = keys[keys == 2].index
    e = e.set_index(["task_id", "seed"]).loc[both].reset_index()
    s = s[s["run_id"].isin(set(e["run_id"]))]
    info = {"variants": sorted(e["variant"].unique().tolist()), "n_episodes": int(len(e)), "n_steps": int(len(s)),
            "n_pairs": int(len(both)), "n_tasks": int(e["task_id"].nunique()),
            "episodes_by_variant": e["variant"].value_counts().to_dict(),
            "steps_by_variant": s["variant"].value_counts().to_dict()}
    return s, e, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="BF16 reference run directory")
    ap.add_argument("--cell", required=True, help="4-bit run directory (same tasks and seeds)")
    ap.add_argument("--ref-variant", default="bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    s, e, info = prepare(a.ref, a.cell)
    res = mixed_effects.run_mixed_effects(s, e, configuration="variant", ref=a.ref_variant, timeout=6 * 3600)
    rv = subprocess.run([mixed_effects.rscript(), "-e",
                         "cat(R.version.string, '|', as.character(packageVersion('lme4')))"],
                        capture_output=True, text=True).stdout.split("|")
    out = {"what": "Registered mixed-effects models A/B/C on real logs (development data, preview; "
                   "the pre-registration is not filed)",
           "reference_dir": a.ref, "cell_dir": a.cell,
           "reference_experiment": os.path.basename(a.ref.rstrip("/")),
           "cell_experiment": os.path.basename(a.cell.rstrip("/")),
           "seeds": list(SAMPLED_SEEDS), "depth": "episode_step bins " + str(mixed_effects.DEPTH_BINS),
           "R": rv[0].strip() if rv else None, "lme4": rv[1].strip() if len(rv) > 1 else None,
           **info, "wall_s": None, "result": res}
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
        f.write("\n")
    print(f"wrote {a.out}: {info['n_episodes']} episodes, {info['n_steps']} steps, "
          f"models {[(m, r.get('status')) for m, r in res.get('models', {}).items()]}")


if __name__ == "__main__":
    main()
