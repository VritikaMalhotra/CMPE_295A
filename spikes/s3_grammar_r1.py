"""Spike S3 / repair R1: per-task GBNF grammar from the task's tool schemas, sent in the
`grammar` field of llama-server's raw /completion endpoint; plus the server-log check
that /v1/chat/completions with tools attaches a lazy grammar while raw /completion does not.

Pilot run: Google Colab A100, llama.cpp v0.6.0, bfcl-eval 2026.3.23, 2026-10-08 PT.
Pilot / development data, not study results.

The JSON-schema-to-GBNF converter is llama.cpp's examples/json_schema_to_grammar.py,
taken from commit ae9afff8d2c012ca760eb9c2adf41961cf6f6232 (the Python copy was removed
upstream on 2026-09-12; v0.6.0 ships only the C++ version).

Grammar shape (Qwen3 native format), pilot variant:
    root  ::= block ("\\n" block)*
    block ::= "<tool_call>\\n" call "\\n</tool_call>"
    call  ::= {"name": <one of the task's tools>, "arguments": <schema-valid object>}
It forces at least one call. The study's R1 also allows a free-text branch.

    python s3_grammar_r1.py run --data .../BFCL_v4_simple_python.json --prompts <bfcl result file with input log>
    python s3_grammar_r1.py lazy-check --gguf models/Qwen3-0.6B-Q4_K_M.gguf
"""
import argparse
import json
import os
import re
import signal
import subprocess
import time

import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import json_schema_to_grammar as j2g  # noqa: E402

LLAMA = "/content/work/llama.cpp"
PORT = 18090
TYPE_MAP = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer",
            "int": "integer", "string": "string", "boolean": "boolean", "bool": "boolean", "array": "array",
            "tuple": "array", "list": "array"}


def to_json_schema(p):
    out = {}
    if p.get("type") in TYPE_MAP:
        out["type"] = TYPE_MAP[p["type"]]
    if "enum" in p:
        out["enum"] = p["enum"]
    if "properties" in p:
        out["properties"] = {k: to_json_schema(v) for k, v in p["properties"].items()}
        out["required"] = [r for r in p.get("required", []) if r in p["properties"]]
        out["additionalProperties"] = False
    if "items" in p:
        out["items"] = to_json_schema(p["items"])
    return out


def call_schema(fn):
    return {"type": "object",
            "properties": {"name": {"const": fn["name"]}, "arguments": to_json_schema(fn["parameters"])},
            "required": ["name", "arguments"], "additionalProperties": False}


def task_grammar(functions):
    conv = j2g.SchemaConverter(prop_order={"name": 0, "arguments": 1}, allow_fetch=False, dotall=False,
                               raw_pattern=False)
    schema = call_schema(functions[0]) if len(functions) == 1 else {"anyOf": [call_schema(f) for f in functions]}
    conv.visit(schema, "call")
    return 'root ::= block ("\\n" block)*\nblock ::= "<tool_call>\\n" call "\\n</tool_call>"\n' + conv.format_grammar()


def serve(gguf, port=PORT, extra="", log="/dev/null"):
    p = subprocess.Popen(f"{LLAMA}/build-cuda/bin/llama-server -m {gguf} -ngl 99 -c 8192 -np 1 --host 127.0.0.1 "
                         f"--port {port} --seed 0 --no-webui {extra} > {log} 2>&1", shell=True, preexec_fn=os.setsid)
    for _ in range(240):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return p
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not start")


def stop(p):
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    p.wait()


def run_condition(tasks, prompts, ids, out_path, use_grammar, temperature=0.0):
    """Writes a bfcl-format result file so `bfcl evaluate` can score it."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for i, tid in enumerate(ids):
            body = {"prompt": prompts[tid], "n_predict": 4096, "temperature": temperature,
                    "cache_prompt": False, "seed": i}
            if use_grammar:
                body["grammar"] = task_grammar(tasks[tid]["function"])
            r = requests.post(f"http://127.0.0.1:{PORT}/completion", json=body, timeout=600).json()
            out.write(json.dumps({"id": tid, "result": r.get("content", ""),
                                  "output_token_count": r.get("timings", {}).get("predicted_n"),
                                  "latency": r.get("timings", {}).get("predicted_ms")}) + "\n")


def lazy_check(gguf, tools, messages, prompt_ids):
    """One fresh verbose server per request; returns the grammar-related log lines for each."""
    found = {}
    for kind in ["chat", "raw"]:
        log = f"/tmp/lazy_{kind}.log"
        p = serve(gguf, port=18095, extra="--jinja --verbose", log=log)
        if kind == "chat":
            requests.post("http://127.0.0.1:18095/v1/chat/completions",
                          json={"messages": messages, "tools": tools, "temperature": 0,
                                "chat_template_kwargs": {"enable_thinking": False}}).json()
        else:
            requests.post("http://127.0.0.1:18095/completion",
                          json={"prompt": prompt_ids, "n_predict": 128, "temperature": 0}).json()
        time.sleep(1.5)
        stop(p)
        found[kind] = [l for l in open(log).read().splitlines() if re.search(r"Grammar", l)]
    return found


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True, help="BFCL_v4_simple_python.json")
    r.add_argument("--prompts", required=True, help="bfcl result file written by generate --include-input-log")
    r.add_argument("--gguf", required=True)
    r.add_argument("--out-root", required=True)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "run":
        tasks = {json.loads(l)["id"]: json.loads(l) for l in open(a.data)}
        prompts = {}
        for l in open(a.prompts):
            rec = json.loads(l)
            prompts[rec["id"]] = rec["inference_log"][0]["content"]["formatted_prompt"]
        ids = sorted(tasks, key=lambda s: int(s.rsplit("_", 1)[1]))
        ids = ids[:a.limit] if a.limit else ids
        p = serve(a.gguf)
        for cond in ["none", "gbnf"]:
            out = f"{a.out_root}_{cond}/result/Qwen_Qwen3-0.6B-FC/non_live/BFCL_v4_simple_python_result.json"
            run_condition(tasks, prompts, ids, out, cond == "gbnf", a.temperature)
        stop(p)


if __name__ == "__main__":
    main()
