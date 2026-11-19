"""Decision 18 (Ekant 2026-11-19; advisor to confirm): exclude tau2 episodes that crash the simulated user.

The final H1 pass left 19 tau2 episodes that segfault gpt-oss's llama-server (d812350) every time they reach the
same simulator request: seeded agents replay the same conversation, the simulator emits the same malformed
user-tool call (HTTP 500, peg-native parse failure) and the next request crashes the server. It reproduced with a
single slot and no prompt cache (simulator 154012), so it is not load or slot reuse. Outages never count as attempts,
so these episodes would stay pending forever; this script excludes them with a permanent ``SimulatorCrash`` retry
record (testbed.logger.record_retry), reported as its own group.

Evidence is per episode: only a pending tau2 episode that the agent logs (slurm_logs/*.out) show losing the
simulator at least MIN_OUTAGES times is excluded; anything else is listed and left pending.

    python scripts/exclude_sim_crash.py <run_dir>... [--apply]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.logger import MAX_INFRA_ATTEMPTS, record_retry, retry_count  # noqa: E402

REASON = ("simulated user (gpt-oss-20b, llama.cpp d812350) segfaults on this episode's request every time: "
          "reproduced 16x in job 153714 and with 1 slot, no prompt cache in job 154012 (decision 18)")


MIN_OUTAGES = 3


class SimulatorCrash(RuntimeError):
    pass


def outages(run_dir: str) -> dict[tuple[str, int], int]:
    """(task_id, seed) -> number of 'simulated user unavailable' lines in the experiment's agent logs."""
    pat = re.compile(r"^\s*(.+?) seed=(\d+): simulated user unavailable")
    n: dict[tuple[str, int], int] = {}
    for p in glob.glob(os.path.join(run_dir, "slurm_logs", "*.out")):
        for line in open(p, errors="replace"):
            m = pat.match(line)
            if m:
                k = (m.group(1), int(m.group(2)))
                n[k] = n.get(k, 0) + 1
    return n


def pending_tau2(run_dir: str) -> list[dict]:
    specs = [json.loads(line) for line in open(os.path.join(run_dir, "specs.jsonl")) if line.strip()]
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run_dir, "episodes", "*.json"))}
    return [s for s in specs if s["benchmark"].startswith("tau2") and s["run_id"] not in done
            and retry_count(run_dir, s["run_id"]) < MAX_INFRA_ATTEMPTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--apply", action="store_true", help="write the records (default: list only)")
    a = ap.parse_args()
    for d in a.run_dirs:
        out = outages(d)
        for s in pending_tau2(d):
            n = out.get((str(s["task_id"]), int(s["seed"])), 0)
            ok = n >= MIN_OUTAGES
            print(f"{d.rstrip('/').split('/')[-2]}\t{s['run_id']}\tattempts={retry_count(d, s['run_id'])}"
                  f"\toutages={n}\t{'exclude' if ok else 'skip'}")
            if a.apply and ok:
                record_retry(d, s["run_id"], SimulatorCrash(REASON), permanent=True)


if __name__ == "__main__":
    main()
