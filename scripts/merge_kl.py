#!/usr/bin/env python3
"""Merge kl_<model>.json files from several scripts/kl_check.sh runs of one model on one backend into one file
(M3 exit: every variant KL-checked against BF16 on both backends; docs/m3_results.md).

    python scripts/merge_kl.py --out results/m3/kl_cuda_Qwen3-0.6B.json \
        results/m2/kl_Qwen3-0.6B.json $TESTBED_OUT/kl_m3/Qwen3-0.6B/kl_Qwen3-0.6B.json

The parts must share model, eval text (SHA-256), context and chunk count, and their BF16 perplexity must agree
(the base pass is recomputed in each run, so a mismatch means the runs were not comparable). A variant measured
in two parts is an error.
"""
from __future__ import annotations

import argparse
import json
import sys

SAME = ("model", "eval_text_sha256", "ctx", "chunks")


def merge(parts: list[dict], names: list[str], ppl_tol: float = 1e-3) -> dict:
    first = parts[0]
    for p, n in zip(parts[1:], names[1:]):
        for k in SAME:
            if p.get(k) != first.get(k):
                raise ValueError(f"{n}: {k} = {p.get(k)!r}, expected {first.get(k)!r}")
    out = {k: first.get(k) for k in first if k != "variants"}
    out["variants"], out["merged_from"], bases = {}, [], []
    for p, n in zip(parts, names):
        if not p.get("complete", True):
            raise ValueError(f"{n}: run not complete")
        for v, r in p["variants"].items():
            if v in out["variants"]:
                raise ValueError(f"{v} appears in more than one part ({n})")
            out["variants"][v] = r
            bases.append((n, r.get("ppl_base")))
        out["merged_from"].append({"file": n, "created": p.get("created"), "variants": list(p["variants"])})
    vals = [b for _, b in bases if b is not None]
    if vals and max(vals) - min(vals) > ppl_tol * max(vals):
        raise ValueError(f"BF16 perplexity differs between parts: {bases}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args(argv)
    try:
        m = merge([json.load(open(p)) for p in a.parts], a.parts)
    except ValueError as e:
        print(f"merge_kl: {e}", file=sys.stderr)
        return 1
    with open(a.out, "w") as fh:
        json.dump(m, fh, indent=1)
    print(f"{a.out}: {', '.join(m['variants'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
