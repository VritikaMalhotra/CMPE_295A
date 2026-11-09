#!/usr/bin/env python3
"""FR-E5 check: calibration text must share no 13-gram with any benchmark task (Workbook 2, Table 19 / FR-E5).

Two steps, both on the login node (they need the testbed venv with bfcl-eval and tau2, no GPU):

    # 1. every string in the BFCL multi-turn data and the tau2 verified tasks -> one text file
    .venv/bin/python scripts/benchmark_ngrams.py dump --out $CMPE295_GROUP/data/calib/benchmark_text.txt
    # 2. drop calibration lines that share a 13-gram with it; write the clean text and a JSON report
    .venv/bin/python scripts/benchmark_ngrams.py clean --bench $CMPE295_GROUP/data/calib/benchmark_text.txt \
        --calib $CMPE295_GROUP/data/eval/wikitext-2-raw/wiki.train.raw \
        --out $CMPE295_GROUP/data/calib/calib.txt --report results/m3/calib_ngram_check.json

Tokens are lower-cased runs of letters and digits, so punctuation and spacing differences cannot hide an
overlap. A line is dropped if any 13-gram inside it, or across its boundary with the next line, is a benchmark
13-gram. The report records both files' SHA-256, what was dropped, and n_overlapping_after (must be 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

N = 13
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def ngrams(toks: list[str], n: int = N):
    for i in range(len(toks) - n + 1):
        yield hash(tuple(toks[i:i + n]))


def ngram_set(text: str, n: int = N) -> set[int]:
    return set(ngrams(tokens(text), n))


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def dump(out: str) -> dict:
    """Write every string of the benchmark data we evaluate on (generously: all BFCL multi-turn files and
    their function docs, all tau2 tasks and policies of the three verified domains)."""
    parts, sources = [], []
    from testbed.adapter import bfcl_data_dir
    d = bfcl_data_dir()
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            if "multi_turn" not in p or not f.endswith((".json", ".jsonl")):
                continue
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parts.extend(_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        parts.append(line)
            sources.append(os.path.relpath(p, d))
    try:
        from tau2.run import load_tasks
        for dom in ("airline", "retail", "telecom"):
            for split in ("base",):
                for t in load_tasks(dom, split):
                    parts.extend(_strings(json.loads(t.model_dump_json())))
            sources.append(f"tau2:{dom}:base tasks")
            try:   # domain policy text, if this tau2 version exposes it
                from tau2.registry import registry
                env = registry.get_env_constructor(dom)()
                parts.append(getattr(env, "policy", "") or "")
                sources.append(f"tau2:{dom}:policy")
            except Exception as e:  # noqa: BLE001 - recorded, not fatal
                sources.append(f"tau2:{dom}:policy unavailable ({type(e).__name__})")
    except ImportError:
        sys.exit("tau2 is not installed in this venv: bash scripts/setup_env.sh --tau2 (login node)")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(parts))
    info = {"out": out, "n_strings": len(parts), "sources": sources}
    with open(out + ".sources.json", "w") as fh:
        json.dump(info, fh, indent=1)
    return info


def clean(bench_text: str, calib_lines: list[str], n: int = N) -> tuple[list[str], list[int]]:
    """Return (kept lines, indices of dropped lines). Overlaps that straddle two lines drop both."""
    bench = ngram_set(bench_text, n)
    toks = [tokens(l) for l in calib_lines]
    flat, owner = [], []
    for i, t in enumerate(toks):
        flat.extend(t)
        owner.extend([i] * len(t))
    drop = set()
    for j in range(len(flat) - n + 1):
        if hash(tuple(flat[j:j + n])) in bench:
            drop.update(range(owner[j], owner[j + n - 1] + 1))
    kept = [l for i, l in enumerate(calib_lines) if i not in drop]
    return kept, sorted(drop)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--out", required=True)
    b = sub.add_parser("clean")
    b.add_argument("--bench", required=True)
    b.add_argument("--calib", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--report", required=True)
    b.add_argument("-n", type=int, default=N)
    args = ap.parse_args(argv)
    if args.cmd == "dump":
        print(json.dumps(dump(args.out))[:400])
        return 0
    with open(args.bench) as fh:
        bench_text = fh.read()
    with open(args.calib) as fh:
        lines = fh.read().split("\n")
    kept, dropped = clean(bench_text, lines, args.n)
    with open(args.out, "w") as fh:
        fh.write("\n".join(kept))
    # verify the written file independently
    after = len(ngram_set(bench_text, args.n) & ngram_set("\n".join(kept), args.n))
    rep = {
        "n": args.n, "calib_in": args.calib, "calib_in_sha256": _sha(args.calib),
        "bench_text": args.bench, "bench_text_sha256": _sha(args.bench),
        "lines_in": len(lines), "lines_dropped": len(dropped), "dropped_line_indices": dropped[:200],
        "calib_out": args.out, "calib_out_sha256": _sha(args.out),
        "tokens_out": sum(len(tokens(l)) for l in kept), "n_overlapping_after": after, "pass": after == 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"dropped {len(dropped)} of {len(lines)} lines; overlapping 13-grams after: {after}; -> {args.out}")
    return 0 if after == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
