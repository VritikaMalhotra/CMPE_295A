"""Decision 18 (Ekant 2026-11-19; advisor to confirm): exclude tau2 episodes that crash the simulated user.

The final H1 pass left 19 tau2 episodes that segfault gpt-oss's llama-server (d812350) every time they reach the
same simulator request: seeded agents replay the same conversation, the simulator emits the same malformed
user-tool call (HTTP 500, peg-native parse failure) and the next request crashes the server. It reproduced with a
single slot and no prompt cache (simulator 154012), so it is not load or slot reuse. Outages never count as attempts,
so these episodes would stay pending forever; this script excludes them with a permanent ``SimulatorCrash`` retry
record (testbed.logger.record_retry), reported as its own group.

Only pending tau2 episodes that already have at least one recorded attempt are excluded (each of the 19 had 1-2
HTTP-500 / decode attempts before the crashes); anything else is listed and left pending.

    python scripts/exclude_sim_crash.py <run_dir>... [--apply]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.logger import MAX_INFRA_ATTEMPTS, record_retry, retry_count  # noqa: E402

REASON = ("simulated user (gpt-oss-20b, llama.cpp d812350) segfaults on this episode's request every time: "
          "reproduced 16x in job 153714 and with 1 slot, no prompt cache in job 154012 (decision 18)")


class SimulatorCrash(RuntimeError):
    pass


def pending_tau2(run_dir: str) -> list[str]:
    specs = [json.loads(line) for line in open(os.path.join(run_dir, "specs.jsonl")) if line.strip()]
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run_dir, "episodes", "*.json"))}
    return [s["run_id"] for s in specs if s["benchmark"].startswith("tau2") and s["run_id"] not in done
            and retry_count(run_dir, s["run_id"]) < MAX_INFRA_ATTEMPTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--apply", action="store_true", help="write the records (default: list only)")
    a = ap.parse_args()
    for d in a.run_dirs:
        for rid in pending_tau2(d):
            n = retry_count(d, rid)
            act = "exclude" if n >= 1 else "skip (no recorded attempt)"
            print(f"{d.rstrip('/').split('/')[-2]}\t{rid}\tattempts={n}\t{act}")
            if a.apply and n >= 1:
                record_retry(d, rid, SimulatorCrash(REASON), permanent=True)


if __name__ == "__main__":
    main()
