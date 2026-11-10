#!/usr/bin/env python3
"""Extend the model manifest with the M3/R2 files (same fields as results/m2/models_manifest.json).

    .venv/bin/python scripts/models_manifest.py --models-dir $TESTBED_MODELS \
        --base results/m2/models_manifest.json --out results/m3/models_manifest.json

For every <Model>-<variant>.gguf of a model already in the base manifest it records SHA-256 (from the .sha256
sidecar, or computed), file bytes, weight count from the GGUF header (llama.cpp's gguf-py at $LLAMA_DIR) and
whole-file bits per weight. Existing entries are re-checked: a SHA-256 that changed is an error, not an update.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys


def sha256(path: str) -> str:
    side = path + ".sha256"
    if os.path.exists(side):
        return open(side).read().split()[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.environ.get("LLAMA_DIR", "llama.cpp"), "gguf-py"))
    from gguf import GGUFReader
    man = json.load(open(a.base))
    problems = []
    for name, entry in man["models"].items():
        for path in sorted(glob.glob(os.path.join(a.models_dir, f"{name}-*.gguf"))):
            variant = os.path.basename(path)[len(name) + 1:-len(".gguf")]
            if variant.endswith("Q4_0only"):     # temporary files of the R2b ranking
                continue
            digest = sha256(path)
            old = entry["files"].get(variant)
            if old and old["sha256"] != digest:
                problems.append(f"{name} {variant}: SHA-256 changed ({old['sha256'][:12]} -> {digest[:12]})")
                continue
            if old:
                continue
            n = sum(int(t.n_elements) for t in GGUFReader(path).tensors)
            size = os.path.getsize(path)
            entry["files"][variant] = {"sha256": digest, "file_bytes": size, "n_weights": n,
                                       "bpw_file": round(size * 8 / n, 4)}
            print(f"added {name} {variant}: {size * 8 / n:.3f} bpw")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(man, open(a.out, "w"), indent=1)
    for p in problems:
        print("ERROR", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
