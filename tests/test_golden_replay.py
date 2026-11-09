"""Golden-episode regression (design spec 7.2): recorded model output replayed through C2 + C5 + C6.

Each fixture in tests/golden/ is a finished cluster episode from the Oct 28 damage runs (spec, step log,
episode result). A fake backend returns the recorded completions in order; the BFCL runner executes the
calls with bfcl's executor and scores with bfcl's checker. Every step row must equal the recorded one
except wall-clock fields and fields added after the recording; the verdict must be the same.

A deliberate change to the runner, classifier or checker that alters these rows needs new fixtures
(copy a fresh episode from a cluster run), and the reason belongs in the commit message.
"""
import glob
import json
import os

import pytest

pytest.importorskip("bfcl_eval")

from testbed.adapter import load_bfcl_multi_turn  # noqa: E402
from testbed.backends import parse_completion  # noqa: E402
from testbed.classifier import classify_step  # noqa: E402
from testbed.runner import run_episode  # noqa: E402
from testbed.spec import RunSpec  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
WALL_CLOCK = {"ttft_ms", "decode_ms", "latency_ms"}
FIXTURES = sorted(os.path.basename(p)[:-len(".spec.json")] for p in glob.glob(os.path.join(GOLDEN, "*.spec.json")))


class ReplayBackend:
    def __init__(self, rows):
        self.rows = list(rows)

    def generate(self, messages, tools, constraint=None, seed=0):
        assert self.rows, "the runner asked for more steps than were recorded"
        r = self.rows.pop(0)
        j = {"content": r["raw_text"], "stop_type": r["stop_reason"], "tokens_evaluated": r["prompt_tokens"],
             "tokens_predicted": r["completion_tokens"],
             "timings": {"prompt_ms": r["ttft_ms"], "predicted_ms": r["decode_ms"]}}
        return parse_completion(j, n_prompt=r["prompt_tokens"], budget=4096, constraint=constraint,
                                latency_ms=r["latency_ms"])


def _strip(row, keys):
    return {k: row[k] for k in keys if k not in WALL_CLOCK}


@pytest.mark.parametrize("run_id", FIXTURES)
def test_golden_episode_replay(run_id):
    spec = RunSpec.from_dict(json.load(open(os.path.join(GOLDEN, f"{run_id}.spec.json"))))
    assert spec.run_id == run_id                                  # the spec hashes to the recorded id
    want_rows = [json.loads(line) for line in open(os.path.join(GOLDEN, f"{run_id}.jsonl"))]
    want_ep = json.load(open(os.path.join(GOLDEN, f"{run_id}.json")))
    task = load_bfcl_multi_turn(category=spec.benchmark.removeprefix("bfcl_"), ids=[spec.task_id])[0]

    got_rows = []

    def log_step(row):
        row.failure_class = classify_step(row)                    # as StepLogger does before writing
        got_rows.append(row.to_dict())

    backend = ReplayBackend(want_rows)
    ep = run_episode(spec, task, backend, None, log_step)

    assert not backend.rows, "recorded steps left over"
    keys = sorted(want_rows[0])                                   # fields that existed when recorded
    assert len(got_rows) == len(want_rows)
    for i, (g, w) in enumerate(zip(got_rows, want_rows)):
        assert json.dumps(_strip(g, keys), sort_keys=True) == json.dumps(_strip(w, keys), sort_keys=True), \
            f"step {i} differs"
    for k in ("valid", "error_type", "n_turns", "n_steps", "forced_quit", "context_overflow"):
        assert getattr(ep, k) == want_ep[k], k
