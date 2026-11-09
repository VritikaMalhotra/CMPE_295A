"""NFR-5 throughput report from the pilot's Slurm accounting and episode files.

  python scripts/throughput_report.py --jobs $TESTBED_OUT/throughput_jobs.txt --out-root $TESTBED_OUT \
      --out results/m2/throughput.json

Per level N: array tasks that ran, the most that ran at the same time, queue wait (start - submit), the
wall-clock span from the first start to the last end, finished episodes, and episodes per hour over that span.
Target (Workbook NFR-5): >= 8 concurrent jobs and >= 120 episodes per hour; below 4 concurrent jobs the
registered reduction ladder applies.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import statistics
import subprocess

TARGET_CONCURRENT, TARGET_EPH, LADDER_BELOW = 8, 120, 4


def _t(s: str):
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None                                   # "Unknown", "None"


def sacct(jobid: str) -> list[dict]:
    out = subprocess.run(["sacct", "-j", jobid, "-X", "-n", "-P", "-o", "JobID,Submit,Start,End,State,NodeList"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        jid, sub, start, end, state, nodes = line.split("|")
        rows.append({"job": jid, "submit": _t(sub), "start": _t(start), "end": _t(end), "state": state,
                     "nodes": nodes})
    return rows


def max_overlap(intervals: list[tuple]) -> int:
    ev = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def level(label: str, n: int, jobid: str, out_root: str) -> dict:
    rows = [r for r in sacct(jobid) if r["start"] and r["end"]]
    eps = []
    for p in glob.glob(os.path.join(out_root, f"throughput_{label}", "*", "episodes", "*.json")):
        eps.append(json.load(open(p)))
    if not rows:
        return {"level": label, "n": n, "array": jobid, "tasks_ran": 0, "episodes": len(eps)}
    first, last = min(r["start"] for r in rows), max(r["end"] for r in rows)
    span_h = (last - first).total_seconds() / 3600
    waits = [(r["start"] - r["submit"]).total_seconds() / 60 for r in rows if r["submit"]]
    return {
        "level": label, "n": n, "array": jobid, "tasks_ran": len(rows),
        "max_concurrent": max_overlap([(r["start"], r["end"]) for r in rows]),
        "queue_wait_min_median": statistics.median(waits) if waits else None,
        "queue_wait_min_max": max(waits) if waits else None,
        "span_h": span_h, "episodes": len(eps),
        "episodes_per_hour": len(eps) / span_h if span_h > 0 else None,
        "episode_wall_s_median": statistics.median(e["wall_s"] for e in eps) if eps else None,
        "states": sorted({r["state"] for r in rows}), "nodes": sorted({r["nodes"] for r in rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="lines of '<label> <N> <array job id>' from throughput_submit.sh")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    levels = [level(lb, int(n), j, a.out_root) for lb, n, j in (line.split() for line in open(a.jobs) if line.strip())]
    best = max(levels, key=lambda r: r.get("episodes_per_hour") or 0)
    conc = max(r.get("max_concurrent", 0) for r in levels)
    res = {"levels": levels, "config": "Qwen3-0.6B BF16, one GPU per job, BFCL multi-turn base, sampled decoding",
           "per_user_limit": "QoS gpu3nodes: 2 nodes per user on every GPU partition (P100 nodes 1 GPU, A100 nodes 4)",
           "max_concurrent_observed": conc, "best_episodes_per_hour": best.get("episodes_per_hour"),
           "meets_nfr5": conc >= TARGET_CONCURRENT and (best.get("episodes_per_hour") or 0) >= TARGET_EPH,
           "reduction_ladder_applies": conc < LADDER_BELOW,
           "targets": {"concurrent": TARGET_CONCURRENT, "episodes_per_hour": TARGET_EPH, "ladder_below": LADDER_BELOW}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    for r in levels:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("nodes",)})
    print(f"max concurrent {conc}; best {res['best_episodes_per_hour']} episodes/h; meets NFR-5: {res['meets_nfr5']}")


if __name__ == "__main__":
    main()
