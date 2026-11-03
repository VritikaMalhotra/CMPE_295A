#!/usr/bin/env python3
"""S10 check (Workbook 2 Table 35): after a Slurm array run with a short walltime, every planned episode is
present exactly once and at least one array task was cut by the walltime, requeued and resumed.

Reads the experiment directory that testbed.run writes (registry layout):
  specs.jsonl            planned specs (run_id, config_key, ...)
  episodes/<run_id>.json finished episodes          steps/<run_id>.jsonl  their step rows (turn, step)
  slurm_logs/*.out       one log per array task (array.sbatch.tmpl prints "restart N" and "walltime near")

    python scripts/check_s10.py <run_dir> [--time 00:06:00 --note "..."] [--out s10_check.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

START = re.compile(r"^config (\S+) task (\S+) on (\S+), restart (\d+)")


def check(run_dir: str) -> dict:
    planned = [json.loads(l) for l in open(os.path.join(run_dir, "specs.jsonl")) if l.strip()]
    plan_ids = [s["run_id"] for s in planned]
    by_config = collections.Counter(s["config_key"] for s in planned)

    done_files = glob.glob(os.path.join(run_dir, "episodes", "*.json"))
    done = {}
    bad_episodes = []
    for p in done_files:
        rid = os.path.basename(p)[:-5]
        try:
            ep = json.load(open(p))
        except ValueError:
            bad_episodes.append(rid)
            continue
        if ep.get("run_id", rid) != rid:
            bad_episodes.append(rid)
        done[rid] = ep

    dup_steps = {}
    for rid in done:
        sp = os.path.join(run_dir, "steps", f"{rid}.jsonl")
        if not os.path.exists(sp):
            continue
        keys = collections.Counter()
        for line in open(sp):
            if line.strip():
                r = json.loads(line)
                keys[(r.get("turn"), r.get("step"))] += 1
        d = sum(n - 1 for n in keys.values() if n > 1)
        if d:
            dup_steps[rid] = d

    tasks = collections.defaultdict(lambda: {"starts": 0, "max_restart": 0, "walltime_signals": 0, "hosts": set()})
    for log in sorted(glob.glob(os.path.join(run_dir, "slurm_logs", "*.out"))):
        name = os.path.basename(log)[:-4]
        t = tasks[name]
        for line in open(log, errors="replace"):
            m = START.match(line)
            if m:
                t["config"] = m.group(1)
                t["starts"] += 1
                t["hosts"].add(m.group(3))
                t["max_restart"] = max(t["max_restart"], int(m.group(4)))
            elif line.startswith("walltime near"):
                t["walltime_signals"] += 1
    task_rows = {k: {**v, "hosts": sorted(v["hosts"])} for k, v in tasks.items()}

    missing = sorted(set(plan_ids) - set(done))
    unplanned = sorted(set(done) - set(plan_ids))
    dup_plan = [k for k, n in collections.Counter(plan_ids).items() if n > 1]
    requeued = [k for k, v in task_rows.items() if v["max_restart"] >= 1 and v["walltime_signals"] >= 1]
    leftovers = [os.path.relpath(p, run_dir) for p in glob.glob(os.path.join(run_dir, "*", "*.part"))]
    return {
        "run_dir": run_dir,
        "planned": len(plan_ids),
        "configs": len(by_config),
        "episodes_per_config": dict(by_config),
        "done": len(done),
        "missing": missing,
        "unplanned": unplanned,
        "duplicate_planned_run_ids": dup_plan,
        "bad_episode_files": bad_episodes,
        "duplicate_step_rows": dup_steps,
        "leftover_part_files": leftovers,     # a run stopped mid-episode leaves its steps .part: harmless
        "array_tasks": task_rows,
        "tasks_requeued_by_walltime": requeued,
        "pass": not missing and not unplanned and not dup_plan and not bad_episodes and not dup_steps
                and bool(requeued),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--time", help="slurm.time used for the array (recorded)")
    ap.add_argument("--note", help="how the time limit was chosen (recorded)")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = check(a.run_dir)
    if a.time:
        res["slurm_time"] = a.time
    if a.note:
        res["time_choice"] = a.note
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
