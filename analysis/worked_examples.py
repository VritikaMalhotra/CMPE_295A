#!/usr/bin/env python
"""Worked examples from our own logs (feedback points 5-7, 2026-11-09): one episode per domain and one step per
powered failure class, BF16 next to 4-bit on the same task and seed wherever both exist.

Everything printed is copied from the logged steps (``steps/*.jsonl``) and episode records; nothing is written by
hand. Long texts are cut mechanically at ``--max-chars`` with a visible ``[… N more characters]`` marker. Picks
are deterministic: for each domain, the first task (sorted id) where the reference and the 4-bit cell both have
seed 1 and their verdicts differ; else the first task both have. Development data, preview.

    python analysis/worked_examples.py --model qwen3-0.6b --ref BF16=<ref run dir> --cell Q4_0-pure=<cell run dir> \\
        --out docs/worked_examples_qwen3-0.6b.md [--json results/m3/worked_examples_qwen3-0.6b.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testbed.classifier import classify_step  # noqa: E402
from testbed.spec import StepRow  # noqa: E402

DOMAINS = ["bfcl_multi_turn_base", "bfcl_multi_turn_miss_func", "bfcl_multi_turn_miss_param",
           "tau2_verified_airline", "tau2_verified_retail", "tau2_verified_telecom"]
POWERED = ["exec_error", "tool_error_payload", "unknown_tool", "missing_call", "missing_call_direct_answer"]
FIELDS = set(StepRow.__dataclass_fields__)


def load(run_dir: str, seed: int = 1) -> dict:
    """(benchmark, task) -> {"episode", "steps"} for finished episodes of one seed."""
    specs = {}
    for line in open(os.path.join(run_dir, "specs.jsonl")):
        if line.strip():
            d = json.loads(line)
            specs[d["run_id"]] = d
    out = {}
    for p in glob.glob(os.path.join(run_dir, "episodes", "*.json")):
        e = json.load(open(p))
        sp = specs.get(e["run_id"])
        if not sp or e.get("seed") != seed:
            continue
        sp_path = os.path.join(run_dir, "steps", e["run_id"] + ".jsonl")
        rows = [json.loads(l) for l in open(sp_path) if l.strip()] if os.path.exists(sp_path) else []
        for r in rows:
            r.setdefault("benchmark", sp["benchmark"])
            r["label"] = classify_step(StepRow(**{k: r.get(k) for k in FIELDS if k in r}))
        out[(sp["benchmark"], str(e["task_id"]))] = {"episode": e, "steps": rows, "run_dir": run_dir}
    return out


def cut(s, n: int) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f" [… {len(s) - n} more characters]"


def step_block(r: dict, n: int) -> str:
    lines = [f"- turn {r['turn']}, step {r['step']}: **`{r['label']}`** (stop: {r.get('stop_reason')})",
             f"  - output: `{cut(r.get('raw_text', ''), n)}`"]
    if r.get("decoded_calls"):
        lines.append(f"  - parsed calls: `{cut(r['decoded_calls'], n)}`")
    if r.get("schema_errors"):
        lines.append(f"  - schema errors: `{cut(r['schema_errors'], n)}`")
    if r.get("execution_results"):
        lines.append(f"  - results: `{cut(r['execution_results'], n)}`")
    return "\n".join(lines)


def step_record(r: dict, n: int) -> dict:
    return {"turn": r["turn"], "step": r["step"], "label": r["label"], "output": cut(r.get("raw_text", ""), n),
            "calls": r.get("decoded_calls") or [], "results": [cut(x, n) for x in (r.get("execution_results") or [])]}


def episode_record(ep: dict, n: int, max_steps: int) -> dict:
    e = ep["episode"]
    return {"run_id": e["run_id"], "valid": bool(e.get("valid")), "error_type": e.get("error_type") or "",
            "n_steps": e.get("n_steps"), "steps": [step_record(r, n) for r in ep["steps"][:max_steps]]}


def episode_block(label: str, ep: dict, n: int, max_steps: int) -> str:
    e = ep["episode"]
    head = (f"**{label}**, run `{e['run_id']}`: verdict **{'pass' if e.get('valid') else 'fail'}** "
            f"(`{e.get('error_type') or '-'}`), {e.get('n_steps')} steps")
    body = [step_block(r, n) for r in ep["steps"][:max_steps]]
    if len(ep["steps"]) > max_steps:
        body.append(f"- [… {len(ep['steps']) - max_steps} more steps in `steps/{e['run_id']}.jsonl`]")
    return head + "\n\n" + "\n".join(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=6)
    a = ap.parse_args()
    rl, rd = a.ref.split("=", 1)
    cl, cd = a.cell.split("=", 1)
    ref, cell = load(rd), load(cd)
    md = [f"# Worked examples from the logs: {a.model}, {rl} vs {cl}\n",
          "Copied from the step logs (seed 1, sampled decoding); long texts cut mechanically. Labels are the "
          "classifier's (codebook v0). Development data, preview.\n",
          f"Reference: `{rd}`  \n4-bit cell: `{cd}`\n", "## One episode per domain\n"]
    picks = {"model": a.model, "ref_label": rl, "cell_label": cl, "ref_dir": rd, "cell_dir": cd,
             "domains": {}, "classes": {}}
    for dom in DOMAINS:
        both = sorted(k for k in ref if k[0] == dom and k in cell)
        if not both:
            only = sorted(k for k in ref if k[0] == dom)
            if not only:
                continue
            k = only[0]
            md.append(f"### {dom}, task `{k[1]}` ({cl} has not finished this domain yet)\n")
            md.append(episode_block(rl, ref[k], a.max_chars, a.max_steps) + "\n")
            picks["domains"][dom] = {"task": k[1], "ref_run": ref[k]["episode"]["run_id"], "cell_run": None,
                                     "ref": episode_record(ref[k], a.max_chars, a.max_steps)}
            continue
        diff = [k for k in both if bool(ref[k]["episode"].get("valid")) != bool(cell[k]["episode"].get("valid"))]
        k = (diff or both)[0]
        md.append(f"### {dom}, task `{k[1]}`" + (" (verdicts differ)" if diff else "") + "\n")
        md.append(episode_block(rl, ref[k], a.max_chars, a.max_steps) + "\n")
        md.append(episode_block(cl, cell[k], a.max_chars, a.max_steps) + "\n")
        picks["domains"][dom] = {"task": k[1], "ref_run": ref[k]["episode"]["run_id"],
                                 "cell_run": cell[k]["episode"]["run_id"], "verdicts_differ": bool(diff),
                                 "ref": episode_record(ref[k], a.max_chars, a.max_steps),
                                 "cell": episode_record(cell[k], a.max_chars, a.max_steps)}
    md.append("## One step per powered failure class\n")
    md.append("Classes with enough events for a confirmatory test at 2,000 episodes, or close to it "
              "(`docs/cluster.md`, power table). For each: the first 4-bit step of that class (sorted by task), and "
              "the reference's step at the same task, turn and step index.\n")
    for c in POWERED:
        found = None
        for k in sorted(cell):
            for r in cell[k]["steps"]:
                if r["label"] == c:
                    found = (k, r)
                    break
            if found:
                break
        if not found:
            md.append(f"### `{c}`: no {cl} step of this class yet\n")
            continue
        k, r = found
        md.append(f"### `{c}`: {k[0]}, task `{k[1]}`\n")
        md.append(f"**{cl}**\n\n" + step_block(r, a.max_chars) + "\n")
        twin = [x for x in ref.get(k, {}).get("steps", []) if x["turn"] == r["turn"] and x["step"] == r["step"]]
        md.append(f"**{rl}, same task, turn and step**\n\n" + (step_block(twin[0], a.max_chars) if twin else
                  "- (the reference episode has no step at this position)") + "\n")
        picks["classes"][c] = {"benchmark": k[0], "task": k[1], "turn": r["turn"], "step": r["step"],
                               "cell_run": r["run_id"], "ref_label": twin[0]["label"] if twin else None,
                               "cell_step": step_record(r, a.max_chars),
                               "ref_step": step_record(twin[0], a.max_chars) if twin else None}
    with open(a.out, "w") as f:
        f.write("\n".join(md))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(picks, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
