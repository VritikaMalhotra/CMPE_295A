"""C5 step logger: atomic episode files, orphan step files, lossless Parquet (design spec 7.2)."""
import json
import os

import pandas as pd

from testbed.logger import StepLogger, to_parquet
from testbed.spec import EpisodeResult, StepRow


def _row(run_id, step, calls):
    return StepRow(run_id=run_id, config_key="k", task_id="multi_turn_base_0", seed=1, turn=0, step=step,
                   variant="q4_0", backend="cuda", repair="none", constraint="none", prompt_tokens=10,
                   completion_tokens=5, ttft_ms=1.0, decode_ms=2.0, latency_ms=3.0, stop_reason="eos",
                   raw_text=f"text {step}", n_tool_call_blocks=len(calls), decoded_calls=calls,
                   execution_results=[f"ok {c}" for c in calls], known_tools=True, schema_errors=[],
                   turn_has_ground_truth=True, is_last_step_of_turn=not calls, forced_quit=False,
                   prompt_sha256="p" * 64, prompt_tokens_sha256="t" * 64)


def _ep(run_id):
    return EpisodeResult(run_id=run_id, config_key="k", task_id="multi_turn_base_0", seed=1, variant="q4_0",
                         backend="cuda", repair="none", constraint="none", valid=False, error_type="x", n_turns=1,
                         n_steps=3, forced_quit=False, wall_s=1.0)


def test_abort_leaves_no_files(tmp_path):
    d = str(tmp_path)
    log = StepLogger(d, "r1")
    log(_row("r1", 0, ["ls()"]))
    log.abort()
    assert os.listdir(os.path.join(d, "steps")) == []
    assert os.listdir(os.path.join(d, "episodes")) == []


def test_finish_writes_both_files(tmp_path):
    d = str(tmp_path)
    log = StepLogger(d, "r1")
    log(_row("r1", 0, ["ls()"]))
    log.finish(_ep("r1"))
    assert sorted(os.listdir(os.path.join(d, "steps"))) == ["r1.jsonl"]
    assert sorted(os.listdir(os.path.join(d, "episodes"))) == ["r1.json"]
    row = json.loads(open(os.path.join(d, "steps", "r1.jsonl")).readline())
    assert row["failure_class"]                                   # C6 labels every row before writing


def test_orphan_step_file_is_ignored(tmp_path):
    d = str(tmp_path)
    log = StepLogger(d, "done")
    log(_row("done", 0, []))
    log.finish(_ep("done"))
    with open(os.path.join(d, "steps", "orphan.jsonl"), "w") as f:  # killed between the two renames
        f.write(json.dumps(_row("orphan", 0, []).to_dict()) + "\n")
    sp, _ = to_parquet(d)
    assert set(pd.read_parquet(sp)["run_id"]) == {"done"}


def test_parquet_roundtrip_is_lossless(tmp_path):
    d = str(tmp_path)
    rows = [_row("r1", 0, ["cd(folder='a')", "ls()"]), _row("r1", 1, ["ls(a=True)"]), _row("r1", 2, [])]
    log = StepLogger(d, "r1")
    for r in rows:
        log(r)
    log.finish(_ep("r1"))
    sp, ep = to_parquet(d)
    df = pd.read_parquet(sp)
    for col in ("decoded_calls", "execution_results", "schema_errors"):
        df[col] = df[col].map(json.loads)
    got = df.drop(columns=["failure_class"]).to_dict("records")
    want = [{k: v for k, v in r.to_dict().items() if k != "failure_class"} for r in rows]
    assert got == want
    assert pd.read_parquet(ep)["run_id"].tolist() == ["r1"]
    assert not [p for p in os.listdir(d) if p.endswith(".part")]  # temp files renamed away
