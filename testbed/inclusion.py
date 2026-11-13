"""Inclusion set from a 16-bit reference run (registered rule, Workbook 2 Ch. 7).

A task is excluded from a model's confirmatory tests when the 16-bit reference fails it in every sampled
trial (it shows missing capability, not quantization damage). The greedy pass does not count: only
configurations with temperature > 0. A trial excluded by the infrastructure rule (``retries/``) does not
count; the task is judged on its other trials (none left = incomplete). A task whose sampled trials are not all finished is "incomplete" and
is neither included nor excluded, so the set is written only when the reference is complete (unless
``allow_incomplete``). A model with fewer than 120 included multi-turn tasks moves to single-turn-only
confirmatory analysis; the file records that flag.

  python -m testbed.inclusion --run-dir $TESTBED_OUT/ref_qwen3-0.6b/<hash> --out results/m2/inclusion_qwen3-0.6b.json

An experiment config limits its tasks to the set with ``experiment.inclusion: <path>`` (C11, ``run._plan``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

MIN_MULTI_TURN = 120
RULE = "excluded if the 16-bit reference fails the task in all sampled trials (temperature > 0)"


def build(run_dir: str, variant: str = "bf16", backend: str | None = None, allow_incomplete: bool = False) -> dict:
    specs = [json.loads(line) for line in open(os.path.join(run_dir, "specs.jsonl")) if line.strip()]
    sampled = [s for s in specs if s["variant"] == variant and s["repair"] == "none"
               and float(s.get("decoding", {}).get("temperature", 0.0)) > 0
               and (backend is None or s["backend"] == backend)]
    if not sampled:
        raise ValueError(f"no sampled {variant} reference specs in {run_dir}")
    eps = {}
    for p in glob.glob(os.path.join(run_dir, "episodes", "*.json")):
        e = json.load(open(p))
        eps[e["run_id"]] = e
    planned: dict = defaultdict(list)
    for s in sampled:
        planned[(s["benchmark"], s["task_id"])].append(s["run_id"])
    from .logger import MAX_INFRA_ATTEMPTS, retry_count
    out: dict = {}
    dropped = []                    # trials excluded by the infrastructure rule: the task is judged on the rest
    for (bench, task), rids in sorted(planned.items()):
        b = out.setdefault(bench, {"included": [], "excluded": [], "incomplete": []})
        gone = [r for r in rids if r not in eps and retry_count(run_dir, r) >= MAX_INFRA_ATTEMPTS]
        dropped += [{"benchmark": bench, "task_id": task, "run_id": r} for r in gone]
        rids = [r for r in rids if r not in gone]
        done = [eps[r] for r in rids if r in eps]
        if not rids or len(done) < len(rids):
            b["incomplete"].append(task)
        elif any(e["valid"] for e in done):
            b["included"].append(task)
        else:
            b["excluded"].append(task)
    n_incomplete = sum(len(b["incomplete"]) for b in out.values())
    if n_incomplete and not allow_incomplete:
        raise ValueError(f"{n_incomplete} tasks have unfinished reference trials; finish the run first")
    multi = sum(len(b["included"]) for k, b in out.items() if "multi_turn" in k or k.startswith("tau2"))
    manifest = {}
    mp = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        manifest = {k: m.get(k) for k in ("experiment", "experiment_hash", "testbed_git")}
    return {"rule": RULE, "reference": {"run_dir": run_dir, "variant": variant, "backend": backend, **manifest},
            "trials_per_task": max(len(r) for r in planned.values()),
            "trials_excluded": dropped,
            "benchmarks": out,
            "n_included": {k: len(b["included"]) for k, b in out.items()},
            "n_excluded": {k: len(b["excluded"]) for k, b in out.items()},
            "multi_turn_included": multi,
            "single_turn_only": multi < MIN_MULTI_TURN,
            "complete": n_incomplete == 0}


def filter_tasks(tasks: dict[str, list[str]], inclusion_path: str) -> dict[str, list[str]]:
    """Keep only included tasks, per benchmark, in the original order. A benchmark missing from the file
    is an error: the set must cover every benchmark the experiment runs."""
    inc = json.load(open(inclusion_path))["benchmarks"]
    missing = [b for b in tasks if b not in inc]
    if missing:
        raise ValueError(f"inclusion file {inclusion_path} has no entry for {missing}")
    return {b: [t for t in ids if t in set(inc[b]["included"])] for b, ids in tasks.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = build(a.run_dir, a.variant, a.backend, a.allow_incomplete)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"included {res['n_included']}, excluded {res['n_excluded']}; single-turn only: {res['single_turn_only']}")


if __name__ == "__main__":
    main()
