"""FR-E17 step 2: one model file on one backend, greedy, over the fixed prompts. Resumable (prompts already
in --out are skipped), so a requeued or restarted job continues where it stopped.

  python scripts/parity_run.py --prompts prompts.jsonl --model Qwen3-0.6B-BF16.gguf --backend cpu --out cpu.jsonl

Settings as the study's correctness runs: one slot, prompt cache off, ctx 40960, greedy (temperature 0),
the same llama.cpp commit on both backends. Each line keeps the prompt's token-ID hash as this server
tokenized it, the output text and, when the server returns them, the output token IDs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from testbed.backends import LlamaServer, ServerConfig  # noqa: E402
from testbed.spec import canonical_json, sha256_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TESTBED_CPU_THREADS") or 0) or None)
    a = ap.parse_args()

    binary = os.environ["LLAMA_CUDA_BIN" if a.backend == "cuda" else "LLAMA_CPU_BIN"]
    prompts = [json.loads(line) for line in open(a.prompts)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["id"] for line in open(a.out) if line.strip()}
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{a.backend} {os.path.basename(a.model)}: {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    cfg = ServerConfig(binary=binary, model_path=a.model, backend=a.backend, ctx=a.ctx,
                       threads=a.threads if a.backend == "cpu" else None)
    log = a.out + ".server.log"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with LlamaServer(cfg, log_path=log) as srv, open(a.out, "a") as f:
        version = srv.props().get("build_info", "")
        for p in todo:
            tok = requests.post(srv.base + "/tokenize", json={"content": p["prompt"], "add_special": False},
                                timeout=600).json()["tokens"]
            body = {"prompt": p["prompt"], "n_predict": a.n_predict, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                    "min_p": 0.0, "seed": 0, "cache_prompt": False, "stream": False, "return_tokens": True}
            t0 = time.time()
            r = requests.post(srv.base + "/completion", json=body, timeout=7200)
            r.raise_for_status()
            j = r.json()
            f.write(json.dumps({
                "id": p["id"], "backend": a.backend, "model_file": os.path.basename(a.model),
                "prompt_sha256": sha256_text(p["prompt"]), "prompt_tokens_sha256": sha256_text(canonical_json(tok)),
                "n_prompt_tokens": len(tok), "content": j.get("content", ""), "tokens": j.get("tokens"),
                "stop_type": j.get("stop_type"), "timings": j.get("timings"), "wall_s": time.time() - t0,
                "server_build": version}, ensure_ascii=False) + "\n")
            f.flush()
    print("finished", flush=True)


if __name__ == "__main__":
    main()
