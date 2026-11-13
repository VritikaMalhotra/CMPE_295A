#!/usr/bin/env python
"""Finished-episode counts of this week's runs, for the week plan and the deck (counted, never typed in).

For each run directory: planned specs, finished episodes (``episodes/*.json``), excluded episodes (a ``permanent``
line in ``retries/``, or 4 recorded attempts), and the split by benchmark; plus the Slurm submissions logged in
``$TESTBED_OUT/simuser/submissions.txt``.

    python scripts/run_status.py --runs <exp>/<hash> [...] --out results/m3/run_status.json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os

MAX_INFRA_ATTEMPTS = 4      # testbed/logger.py; kept here so this script runs without the testbed's dependencies


def status(run_dir: str) -> dict:
    specs = [json.loads(l) for l in open(os.path.join(run_dir, "specs.jsonl")) if l.strip()]
    by_id = {s["run_id"]: s for s in specs}
    done = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run_dir, "episodes", "*.json"))]
    # excluded: a ``permanent`` line, or MAX_INFRA_ATTEMPTS recorded attempts (testbed.logger.retry_count, the
    # rule the runner, the report and the inclusion set use)
    excluded = set()
    for p in glob.glob(os.path.join(run_dir, "retries", "*.jsonl")):
        rows = [json.loads(line) for line in open(p) if line.strip()]
        if any(r.get("permanent") for r in rows) or len(rows) >= MAX_INFRA_ATTEMPTS:
            excluded.add(os.path.basename(p)[:-6])
    bench = collections.Counter(by_id[r]["benchmark"] for r in done if r in by_id)
    planned = collections.Counter(s["benchmark"] for s in specs)
    return {"dir": run_dir, "experiment": specs[0]["experiment"] if specs else None,
            "model": specs[0]["model"] if specs else None, "variants": sorted({s["variant"] for s in specs}),
            "specs": len(specs), "finished": len(done), "excluded": len(excluded),
            "finished_share": round(len(done) / len(specs), 4) if specs else 0.0,
            "by_benchmark": {b: [bench.get(b, 0), n] for b, n in sorted(planned.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="<experiment>/<hash> under $TESTBED_OUT, or full paths")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.environ.get("TESTBED_OUT", "runs")
    out = {"collected": datetime.datetime.now().astimezone().isoformat(timespec="minutes"), "runs": {}}
    for r in a.runs:
        d = r if os.path.isabs(r) else os.path.join(root, r)
        s = status(d)
        out["runs"][s["experiment"] or r] = s
        print(f"{s['experiment']}: {s['finished']}/{s['specs']} finished, {s['excluded']} excluded")
    sub = os.path.join(root, "simuser", "submissions.txt")
    if os.path.exists(sub):
        out["submissions"] = [l.rstrip("\n") for l in open(sub) if l.strip()]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
