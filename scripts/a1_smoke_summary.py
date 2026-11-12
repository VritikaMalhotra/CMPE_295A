#!/usr/bin/env python
"""A1 evidence: what each prompt-handler smoke run did, read from its run directory (never typed in).

For every spec: benchmark, task, seed, whether an episode was logged, its verdict (``valid``), steps, error type,
and for tau2 the reward in ``tau2_sims/``; plus the testbed commit(s) in the run's manifests. "Ran end to end" means
an episode was logged and scored by the official evaluator; "valid" means the task was solved.

    python scripts/a1_smoke_summary.py <run dir> [...] --out results/m2/a1_smoke_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time


def summary(run_dir: str) -> dict:
    specs = [json.loads(l) for l in open(os.path.join(run_dir, "specs.jsonl")) if l.strip()]
    eps = {}
    for p in glob.glob(os.path.join(run_dir, "episodes", "*.json")):
        e = json.load(open(p))
        eps[e["run_id"]] = e
    commits = sorted({json.load(open(p)).get("testbed_git", {}).get("commit", "unknown")
                      for p in glob.glob(os.path.join(run_dir, "manifests", "*.json"))})
    rows = []
    for s in specs:
        e = eps.get(s["run_id"])
        sim = os.path.join(run_dir, "tau2_sims", s["run_id"] + ".json")
        rows.append({"benchmark": s["benchmark"], "task_id": s["task_id"], "seed": s["seed"],
                     "model": s.get("model"), "ran_end_to_end": e is not None,
                     "valid": e["valid"] if e else None, "n_steps": e["n_steps"] if e else None,
                     "error_type": e["error_type"] if e else None,
                     "tau2_reward": json.load(open(sim)).get("reward") if os.path.exists(sim) else None})
    return {"run_dir": run_dir, "testbed_commits": commits,
            "specs_written": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(
                os.path.join(run_dir, "specs.jsonl")))),
            "episodes": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [summary(d) for d in a.run_dirs]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "label": "development data (A1 smoke runs)",
           "runs": runs}
    json.dump(out, open(a.out, "w"), indent=1)
    for r in runs:
        print(r["run_dir"], [c[:7] for c in r["testbed_commits"]])
        for e in r["episodes"]:
            print(f"  {e['benchmark']} {e['task_id']} s{e['seed']}: end_to_end={e['ran_end_to_end']} "
                  f"valid={e['valid']} steps={e['n_steps']} {e['error_type'] or ''}")


if __name__ == "__main__":
    main()
