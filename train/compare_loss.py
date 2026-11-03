#!/usr/bin/env python3
"""S7 check: compare the per-step losses of an uninterrupted toy-QAD run (A) with a run that was cut by the
Slurm walltime, requeued and resumed from its checkpoint (B). Pass when every step is present in both and the
losses are identical, and B's job log shows that it really was cut and resumed.

    python train/compare_loss.py <A_dir> <B_dir> [--b-log <slurm .out of B> ...] [--out s7_compare.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re


def read_losses(run_dir: str) -> dict[int, dict]:
    rows = {}
    with open(os.path.join(run_dir, "loss.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["step"])] = r
    return rows


def read_log_events(paths: list[str]) -> dict:
    ev = {"walltime_signals": 0, "resumed_from": [], "restarts_seen": [], "done": None, "limits": []}
    for p in paths:
        for line in open(p, errors="replace"):
            if line.startswith("walltime near"):
                ev["walltime_signals"] += 1
            m = re.search(r"limit (\S+) restart (\d+)", line)
            if m:
                ev["limits"].append(m.group(1))
                ev["restarts_seen"].append(int(m.group(2)))
            if line.startswith("{"):
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("event") == "resumed":
                    ev["resumed_from"].append(j.get("from_step"))
                elif j.get("event") == "done":
                    ev["done"] = j
    return ev


def compare(a_dir: str, b_dir: str, b_logs: list[str] | None = None) -> dict:
    a, b = read_losses(a_dir), read_losses(b_dir)
    common = sorted(set(a) & set(b))
    diffs = [abs(a[s]["loss"] - b[s]["loss"]) for s in common]
    first_diff = next((s for s in common if a[s]["loss"] != b[s]["loss"]), None)
    out = {
        "a_dir": a_dir,
        "b_dir": b_dir,
        "steps_a": len(a),
        "steps_b": len(b),
        "missing_in_b": sorted(set(a) - set(b)),
        "extra_in_b": sorted(set(b) - set(a)),
        "max_abs_diff": max(diffs) if diffs else None,
        "first_diff_step": first_diff,
        "identical": bool(common) and set(a) == set(b) and first_diff is None,
        "peak_alloc_mib_a": max((r.get("peak_alloc_mib", 0) for r in a.values()), default=None),
        "peak_alloc_mib_b": max((r.get("peak_alloc_mib", 0) for r in b.values()), default=None),
    }
    if b_logs:
        ev = read_log_events(b_logs)
        out["b_log"] = ev
        out["b_was_cut_and_resumed"] = ev["walltime_signals"] >= 1 and bool(ev["resumed_from"]) \
            and max(ev["restarts_seen"], default=0) >= 1
        out["pass"] = out["identical"] and out["b_was_cut_and_resumed"]
    else:
        out["pass"] = out["identical"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a_dir")
    ap.add_argument("b_dir")
    ap.add_argument("--b-log", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    res = compare(args.a_dir, args.b_dir, args.b_log)
    text = json.dumps(res, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
