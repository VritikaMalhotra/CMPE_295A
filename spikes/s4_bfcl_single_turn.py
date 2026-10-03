"""Spike S4 + FR-E2 pilot: BFCL single-turn non-live through the official bfcl-eval
against a local llama-server, then a per-task table (parsed, tool exists, args
schema-valid, correct).

Pilot run: Google Colab A100, bfcl-eval 2026.3.23, llama.cpp v0.6.0, 2026-10-08 PT.
Pilot / development data, not study results.

Setup notes
- bfcl-eval lives in its own venv (Python 3.12). It also needs `soundfile`
  (imported by qwen_agent) which the package does not declare.
- bfcl's QwenFCHandler builds a thinking-mode prompt. For non-thinking runs we
  append Qwen3's enable_thinking=false suffix in `_format_prompt`:
      formatted_prompt += "<|im_start|>assistant\\n"
    + formatted_prompt += "<think>\\n\\n</think>\\n\\n"
- Generation uses /v1/completions on llama-server (port 18080; 8080 is taken on Colab).

    python s4_bfcl_single_turn.py run --fmt Q4_K_M --gguf models/Qwen3-0.6B-Q4_K_M.gguf --cats simple_python
    python s4_bfcl_single_turn.py table --fmts BF16,Q4_K_M,Q4_0 --cat simple_python --out bfcl_pilot.csv
"""
import argparse
import json
import os
import re
import subprocess
import time

import jsonschema
import pandas as pd
import requests

LLAMA = "/content/work/llama.cpp"
BFCL_BIN = "/content/bfcl-env/bin"
MODEL_DIR = "/content/work/Qwen3-0.6B"
MODEL = "Qwen/Qwen3-0.6B-FC"
PORT = 18080
TOOL_CALL_RE = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.S)  # same pattern as bfcl's handler
TYPE_MAP = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer",
            "int": "integer", "string": "string", "boolean": "boolean", "bool": "boolean", "array": "array",
            "tuple": "array", "list": "array"}


def run(fmt, gguf, cats, root_base="/content/work/bfcl"):
    root = f"{root_base}/{fmt}"
    os.makedirs(root, exist_ok=True)
    server = subprocess.Popen(
        f"{LLAMA}/build-cuda/bin/llama-server -m {gguf} -ngl 99 -c 8192 -np 1 --host 127.0.0.1 "
        f"--port {PORT} --seed 0 --no-webui > {root}/server.log 2>&1", shell=True)
    for _ in range(300):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(1)
    env = dict(os.environ, BFCL_PROJECT_ROOT=root, LOCAL_SERVER_ENDPOINT="127.0.0.1", LOCAL_SERVER_PORT=str(PORT))
    subprocess.run(f"{BFCL_BIN}/bfcl generate --model {MODEL} --test-category {cats} --temperature 0 "
                   f"--skip-server-setup --local-model-path {MODEL_DIR} --num-threads 1 --include-input-log",
                   shell=True, env=env, check=True)
    server.terminate()
    server.wait()
    subprocess.run(f"{BFCL_BIN}/bfcl evaluate --model {MODEL} --test-category {cats}", shell=True, env=env, check=True)


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


def schema_valid(call, functions):
    fn = next((f for f in functions if f["name"] == call.get("name")), None)
    if fn is None or not isinstance(call.get("arguments"), dict):
        return False
    try:
        jsonschema.validate(call["arguments"], to_json_schema(fn["parameters"]))
        return True
    except jsonschema.ValidationError:
        return False


def classify(raw, functions):
    blocks = TOOL_CALL_RE.findall(raw)
    calls, bad = [], 0
    for b in blocks:
        try:
            c = json.loads(b)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict):
            calls.append(c)
        else:
            bad += 1
    names = {f["name"] for f in functions}
    parsed = bool(calls) and bad == 0
    tool_exists = parsed and all(c["name"] in names for c in calls)
    valid = tool_exists and all(schema_valid(c, functions) for c in calls)
    return {"n_blocks": len(blocks), "n_calls": len(calls), "parsed": parsed, "tool_exists": tool_exists,
            "schema_valid": valid, "called_names": "|".join(c["name"] for c in calls)}


def failure_class(row):
    if row["correct"]:
        return "correct"
    if not row["parsed"]:
        return "no_call" if row["n_blocks"] == 0 else "malformed"
    if not row["tool_exists"]:
        return "hallucinated_tool"
    if not row["schema_valid"]:
        return "schema_invalid"
    return "wrong_value_or_count"


def table(fmts, cat, data_dir, root_base="/content/work/bfcl"):
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open(f"{data_dir}/BFCL_v4_{cat}.json")}
    rows = []
    for fmt in fmts:
        base = f"{root_base}/{fmt}"
        score = [json.loads(l) for l in open(f"{base}/score/Qwen_Qwen3-0.6B-FC/non_live/BFCL_v4_{cat}_score.json")]
        fails = {d["id"]: d for d in score[1:]}
        for l in open(f"{base}/result/Qwen_Qwen3-0.6B-FC/non_live/BFCL_v4_{cat}_result.json"):
            r = json.loads(l)
            raw = r["result"] if isinstance(r["result"], str) else json.dumps(r["result"])
            row = {"task_id": r["id"], "format": fmt, "category": cat, **classify(raw, tasks[r["id"]]["function"]),
                   "correct": r["id"] not in fails, "bfcl_error_type": fails.get(r["id"], {}).get("error_type", ""),
                   "output_tokens": r.get("output_token_count")}
            row["failure_class"] = failure_class(row)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fmt", required=True)
    r.add_argument("--gguf", required=True)
    r.add_argument("--cats", default="simple_python")
    t = sub.add_parser("table")
    t.add_argument("--fmts", default="BF16,Q4_K_M,Q4_0")
    t.add_argument("--cat", default="simple_python")
    t.add_argument("--data-dir", default="/content/bfcl-env/lib/python3.12/site-packages/bfcl_eval/data")
    t.add_argument("--out", default="bfcl_pilot.csv")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.fmt, a.gguf, a.cats)
    else:
        df = table(a.fmts.split(","), a.cat, a.data_dir)
        df.to_csv(a.out, index=False)
        print(df.groupby("format", sort=False)[["parsed", "tool_exists", "schema_valid", "correct"]].mean())


if __name__ == "__main__":
    main()
