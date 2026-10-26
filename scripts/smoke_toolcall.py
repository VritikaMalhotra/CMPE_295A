"""Spike S1 on any GPU node (Workbook Table 37b, step 4): serve one GGUF with the testbed's own backend and
send the pilot's single tool-call request. Pass = the reply parses as one get_weather call.

    .venv/bin/python scripts/smoke_toolcall.py --model $TESTBED_MODELS/Qwen3-0.6B-Q4_K_M.gguf \
        --binary $LLAMA_CUDA_BIN --out $TESTBED_OUT/smoke/s1_$SLURM_JOB_ID.json [--backend cuda] [--reps 3]

Writes one JSON record (node, GPU, CUDA_VISIBLE_DEVICES, binary version, model SHA-256, timings, GPU memory)
so cluster results can be compared with the Oct 8 Colab pilot. Development data.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.backends import Backend, LlamaServer, ServerConfig  # noqa: E402
from testbed.registry import file_sha256  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City and state, e.g. San Jose, CA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
MESSAGES = [{"role": "user", "content": "What is the weather in San Jose, CA right now? Answer in celsius."}]


def sh(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (p.stdout + p.stderr).strip()          # llama-server prints --version to stderr
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {type(e).__name__}"


def gpu_memory_mib() -> str:
    return sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args()
    rec: dict = {
        "spike": "S1", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
        "binary": a.binary, "binary_version": [l for l in sh([a.binary, "--version"]).splitlines() if l.startswith("version")],
        "model": os.path.basename(a.model), "model_sha256": file_sha256(a.model), "backend": a.backend,
    }
    log = os.path.splitext(a.out)[0] + ".server.log"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    threads = int(os.environ["SLURM_CPUS_PER_TASK"]) if os.environ.get("SLURM_CPUS_PER_TASK") else None
    rec["threads"] = threads                          # Slurm does not pin CPUs here (docs/cluster.md)
    cfg = ServerConfig(binary=a.binary, model_path=a.model, backend=a.backend, ctx=a.ctx, threads=threads)
    t0 = time.time()
    with LlamaServer(cfg, log_path=log) as srv:
        rec["server_start_s"] = round(time.time() - t0, 1)
        be = Backend(srv, {"temperature": 0.0, "n_predict": 512})   # greedy: one answer per file and backend
        reps = []
        for i in range(a.reps):
            g = be.generate(MESSAGES, TOOLS, seed=0)
            reps.append({"calls": [{"name": c.name, "arguments": c.arguments} for c in g.tool_calls],
                         "blocks": g.tool_call_blocks, "prompt_tokens": g.prompt_tokens,
                         "completion_tokens": g.completion_tokens, "ttft_ms": g.ttft_ms, "decode_ms": g.decode_ms,
                         "latency_ms": round(g.latency_ms, 1), "stop": g.stop_reason, "text": g.text[:400]})
        rec["gpu_memory_during"] = gpu_memory_mib()
    rec["reps"] = reps
    first = reps[0]["calls"]
    rec["pass"] = (len(first) == 1 and first[0]["name"] == "get_weather"
                   and "san jose" in str(first[0]["arguments"].get("city", "")).lower())
    rec["identical_across_reps"] = len({json.dumps(r["calls"], sort_keys=True) for r in reps}) == 1
    toks = [r["completion_tokens"] / (r["decode_ms"] / 1000) for r in reps if r["decode_ms"]]
    rec["decode_tok_per_s"] = round(sum(toks) / len(toks), 1) if toks else None
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))
    print("PASS" if rec["pass"] else "FAIL")
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
