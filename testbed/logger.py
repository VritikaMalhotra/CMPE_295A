"""C5 Step Logger: one JSONL record per step, converted to Parquet after each run.

Owner: Ekant Kapgate (draft by Yashashav for review).

Layout under the run directory:  steps/<run_id>.jsonl, episodes/<run_id>.json  ->  steps.parquet, episodes.parquet
Each JSONL line is ``StepRow.to_dict()`` with ``failure_class`` filled by C6 before writing.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from .classifier import classify_step
from .spec import EpisodeResult, StepRow


class StepLogger:
    def __init__(self, run_dir: str, run_id: str):
        os.makedirs(os.path.join(run_dir, "steps"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "episodes"), exist_ok=True)
        self.path = os.path.join(run_dir, "steps", f"{run_id}.jsonl")
        self.ep_path = os.path.join(run_dir, "episodes", f"{run_id}.json")
        self._tmp = self.path + ".part"
        self._f = open(self._tmp, "w")

    def __call__(self, row: StepRow) -> None:
        row.failure_class = classify_step(row)
        self._f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()

    def finish(self, ep: EpisodeResult) -> None:
        self._f.close()
        os.replace(self._tmp, self.path)          # a step file exists only for a finished episode
        with open(self.ep_path + ".part", "w") as f:
            json.dump(ep.to_dict(), f)
        os.replace(self.ep_path + ".part", self.ep_path)

    def abort(self) -> None:
        self._f.close()
        if os.path.exists(self._tmp):
            os.remove(self._tmp)


def to_parquet(run_dir: str) -> tuple[str, str]:
    eps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.json")))]
    done = {e["run_id"] for e in eps}
    # only steps of finished episodes: a kill between the two renames in finish() leaves an orphan step file
    steps = [json.loads(line) for p in sorted(glob.glob(os.path.join(run_dir, "steps", "*.jsonl")))
             if os.path.basename(p)[:-6] in done for line in open(p) if line.strip()]
    sp, ep = os.path.join(run_dir, "steps.parquet"), os.path.join(run_dir, "episodes.parquet")
    tmp_sp, tmp_ep = sp + f".{os.getpid()}.part", ep + f".{os.getpid()}.part"
    sdf = pd.DataFrame(steps)
    for col in ("decoded_calls", "execution_results", "schema_errors"):   # lists -> JSON text for Parquet
        if col in sdf:
            sdf[col] = sdf[col].map(json.dumps)
    sdf.to_parquet(tmp_sp, index=False)
    pd.DataFrame(eps).to_parquet(tmp_ep, index=False)
    os.replace(tmp_sp, sp)
    os.replace(tmp_ep, ep)
    return sp, ep
