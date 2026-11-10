#!/usr/bin/env python
"""Provenance check for paired runs: the first prompt of every episode (turn 0, step 0) must be byte-identical,
and tokenise identically, between a BF16 reference and a 4-bit cell of the same model, whatever testbed commit
each episode ran under. The first prompt depends only on the task, its tools and the prompt format, so a
mismatch would mean the prompt path changed between the runs.

    python analysis/prompt_identity_check.py --ref <ref run dir> --cell <cell run dir> [...] --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def first_prompts(run_dir: str) -> dict:
    specs = {}
    for line in open(os.path.join(run_dir, "specs.jsonl")):
        if line.strip():
            d = json.loads(line)
            specs[d["run_id"]] = d
    out = {}
    for p in glob.glob(os.path.join(run_dir, "steps", "*.jsonl")):
        rid = os.path.basename(p)[:-6]
        if rid not in specs:
            continue
        with open(p) as f:
            line = f.readline()
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("turn") != 0 or r.get("step") != 0:
            continue
        s = specs[rid]
        out[(s["benchmark"], str(s["task_id"]), int(s["seed"]))] = (r.get("prompt_sha256"), r.get("prompt_tokens_sha256"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = first_prompts(a.ref)
    res = {"reference": a.ref, "cells": {}}
    for c in a.cell:
        cell = first_prompts(c)
        both = sorted(set(ref) & set(cell))
        text_mis = [k for k in both if ref[k][0] != cell[k][0]]
        tok_mis = [k for k in both if ref[k][1] != cell[k][1]]
        commits = sorted({json.load(open(m)).get("testbed_git", {}).get("commit", "")[:7]
                          for m in glob.glob(os.path.join(c, "manifests", "*.json"))})
        res["cells"][c] = {"compared": len(both), "text_mismatch": len(text_mis), "token_mismatch": len(tok_mis),
                           "examples": [list(k) for k in (text_mis or tok_mis)[:5]], "manifest_commits": commits,
                           "pass": not text_mis and not tok_mis and len(both) > 0}
        print(f"{c}: {len(both)} first prompts compared, {len(text_mis)} text / {len(tok_mis)} token mismatches; "
              f"commits {commits}")
    res["pass"] = all(v["pass"] for v in res["cells"].values())
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
