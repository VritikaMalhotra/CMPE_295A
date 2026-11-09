"""FR-E17 step 3: compare the CUDA and CPU outputs of each model file on the same prompts.

  python scripts/parity_report.py --model qwen3-0.6b --pair bf16 cuda.jsonl cpu.jsonl \
      --pair q4_0 cuda_q4.jsonl cpu_q4.jsonl --out results/m2/parity_qwen3-0.6b.json

Per variant: prompts compared, share with identical prompt token IDs (must be 100%: both servers saw the
same tokens), share of token-identical outputs, first-divergence position, and agreement on the tool-call
decision and on the decoded calls. Workbook FR-E17: results are pooled across backends only when the
token-identical share is at least 99%; otherwise every RQ1 ratio is reported per backend.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from testbed import qwen_format  # noqa: E402

POOL_THRESHOLD = 0.99


def _load(path: str) -> dict:
    return {r["id"]: r for r in (json.loads(line) for line in open(path) if line.strip())}


def _calls(text: str):
    calls, n = qwen_format.extract_tool_calls(qwen_format.split_think(text)[1])
    return n, json.dumps(calls, sort_keys=True)


def _first_diff(a, b) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    use_tokens = all(a[i].get("tokens") is not None and b[i].get("tokens") is not None for i in ids)
    same_prompt = same_out = same_decision = same_calls = 0
    diverge = []
    for i in ids:
        x, y = a[i], b[i]
        same_prompt += x["prompt_tokens_sha256"] == y["prompt_tokens_sha256"]
        ox, oy = (x["tokens"], y["tokens"]) if use_tokens else (x["content"], y["content"])
        d = _first_diff(ox, oy)
        if d is None:
            same_out += 1
        else:
            diverge.append(d)
        (nx, cx), (ny, cy) = _calls(x["content"]), _calls(y["content"])
        same_decision += (nx > 0) == (ny > 0)
        same_calls += cx == cy
    n = len(ids)
    share = same_out / n
    return {
        "n": n, "compared_on": "token_ids" if use_tokens else "text",
        "prompt_tokens_identical": same_prompt / n,
        "output_identical": share,
        "first_divergence_median": statistics.median(diverge) if diverge else None,
        "first_divergence_min": min(diverge) if diverge else None,
        "tool_call_decision_agree": same_decision / n,
        "decoded_calls_agree": same_calls / n,
        "pool_backends": share >= POOL_THRESHOLD,
        "files": sorted({a[ids[0]]["model_file"], b[ids[0]]["model_file"]}),
        "server_build": sorted({a[ids[0]].get("server_build", ""), b[ids[0]].get("server_build", "")}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pair", nargs=3, action="append", metavar=("VARIANT", "CUDA_JSONL", "CPU_JSONL"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"model": a.model, "rule": f"pool backends only if output_identical >= {POOL_THRESHOLD} (FR-E17)",
           "decoding": "greedy (temperature 0, top_k 1), cache_prompt false, ctx 40960, one slot",
           "variants": {v: compare(_load(c), _load(p)) for v, c, p in a.pair}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for v, r in res["variants"].items():
        if r["n"]:
            print(f"{a.model} {v}: n={r['n']} prompt tokens identical {r['prompt_tokens_identical']:.1%}, "
                  f"outputs identical {r['output_identical']:.1%}, tool-call decision agree "
                  f"{r['tool_call_decision_agree']:.1%}, pool={r['pool_backends']}")


if __name__ == "__main__":
    main()
