"""FR-E17 step 1: the fixed prompts both backends get.

Free-running episodes on CUDA and CPU stop seeing the same conversation after their first differing step,
so the parity check instead sends the *same* prompts to both. They are rebuilt from finished episodes of a
recorded run (development data): each episode is replayed through the BFCL runner with its recorded model
output, and the exact prompt text before every step is captured (the same ``qwen_format.format_prompt`` the
backend uses). At most ``--per-episode`` steps per episode, so many tasks and turn depths are covered.

  python scripts/parity_prompts.py --run-dir $TESTBED_OUT/damage_0p6b/61417027a6 --variant bf16 --seed 1 \
      --n 200 --out $TESTBED_OUT/parity/prompts_qwen3-0.6b.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from testbed import qwen_format  # noqa: E402
from testbed.adapter import load_bfcl_multi_turn  # noqa: E402
from testbed.backends import parse_completion  # noqa: E402
from testbed.runner import run_episode  # noqa: E402
from testbed.spec import RunSpec, sha256_text  # noqa: E402


class CaptureBackend:
    """Returns the recorded completions in order and keeps the prompt text the real backend would send."""

    def __init__(self, rows, thinking: bool):
        self.rows, self.thinking, self.prompts = list(rows), thinking, []

    def generate(self, messages, tools, constraint=None, seed=0):
        if not self.rows:
            raise RuntimeError("replay asked for more steps than were recorded")
        self.prompts.append(qwen_format.format_prompt(messages, tools, thinking=self.thinking))
        r = self.rows.pop(0)
        j = {"content": r["raw_text"], "stop_type": r["stop_reason"], "tokens_evaluated": r["prompt_tokens"],
             "tokens_predicted": r["completion_tokens"], "timings": {}}
        return parse_completion(j, n_prompt=r["prompt_tokens"], budget=4096, constraint=constraint, latency_ms=0.0)


def episode_prompts(spec: RunSpec, rows: list[dict], task) -> list[dict]:
    """Prompts before each recorded step of one episode, or [] when the replay does not match the record."""
    be = CaptureBackend(rows, spec.thinking)
    try:
        run_episode(spec, task, be, None, lambda row: None)
    except Exception:
        return []
    if be.rows or len(be.prompts) != len(rows):
        return []
    return [{"source_run_id": spec.run_id, "task_id": spec.task_id, "turn": r["turn"], "step": r["step"],
             "prompt": p, "prompt_sha256": sha256_text(p)} for r, p in zip(rows, be.prompts)]


def collect(run_dir: str, variant: str, seed: int, n: int, per_episode: int, sample_seed: int) -> list[dict]:
    specs = {}
    for line in open(os.path.join(run_dir, "specs.jsonl")):
        d = json.loads(line)
        specs[d["run_id"]] = d
    eps = []
    for p in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.json"))):
        e = json.load(open(p))
        if e["variant"] == variant and e["repair"] == "none" and e["seed"] == seed and e["run_id"] in specs:
            eps.append(e["run_id"])
    rng = random.Random(sample_seed)
    rng.shuffle(eps)
    out, seen = [], set()
    for rid in eps:
        spec = RunSpec.from_dict(specs[rid])
        if not spec.benchmark.startswith("bfcl"):
            continue
        rows = [json.loads(line) for line in open(os.path.join(run_dir, "steps", f"{rid}.jsonl"))]
        task = load_bfcl_multi_turn(category=spec.benchmark.removeprefix("bfcl_"), ids=[spec.task_id])[0]
        got = episode_prompts(spec, rows, task)
        rng.shuffle(got)
        for g in got[:per_episode]:
            if g["prompt_sha256"] in seen:
                continue
            seen.add(g["prompt_sha256"])
            out.append(g)
            if len(out) == n:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    got = collect(a.run_dir, a.variant, a.seed, a.n, a.per_episode, a.sample_seed)
    if len(got) < a.n:
        sys.exit(f"only {len(got)} prompts found (wanted {a.n})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for i, g in enumerate(got):
            f.write(json.dumps({"id": i, **g}, ensure_ascii=False) + "\n")
    print(f"wrote {len(got)} prompts from {len({g['source_run_id'] for g in got})} episodes to {a.out}")


if __name__ == "__main__":
    main()
