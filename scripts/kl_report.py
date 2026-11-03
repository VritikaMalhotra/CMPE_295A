#!/usr/bin/env python3
"""Turn llama-perplexity logs from scripts/kl_check.sh into one JSON report per model.

For each variant log (<variant>.log, written with --kl-divergence against the BF16 base) the report keeps:
perplexity of the variant and of BF16, mean KL divergence with its uncertainty, the KLD percentiles, RMS and
mean change in the top token's probability, the share of positions where both models pick the same top token,
the file size and the measured whole-file bits per weight from the model loader, and the file's SHA-256
(from the .sha256 sidecar that make_probe_models.sh writes). No pass/fail thresholds: Workbook 2 fixes those
after the pilot. Two ordering checks are recorded: Q8_0 should be closer to BF16 than both 4-bit variants.

    python scripts/kl_report.py --model Qwen3-0.6B --dir <kl_dir> --models-dir <dir> --text <txt> \
        --ctx 512 --chunks 50 --out <kl_dir>/kl_Qwen3-0.6B.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re

NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
PATTERNS = {
    "ppl_q": rf"Mean PPL\(Q\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "ppl_base": rf"Mean PPL\(base\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "kld_mean": rf"Mean\s+KLD:\s*{NUM}\s*±\s*{NUM}",
    "kld_max": rf"Maximum KLD:\s*{NUM}",
    "kld_p999": rf"99\.9%\s+KLD:\s*{NUM}",
    "kld_p99": rf"99\.0%\s+KLD:\s*{NUM}",
    "kld_median": rf"Median\s+KLD:\s*{NUM}",
    "dp_mean_pct": rf"Mean\s+Δp:\s*{NUM}\s*±\s*{NUM}\s*%",
    "dp_rms_pct": rf"RMS Δp\s*:\s*{NUM}\s*±\s*{NUM}\s*%",
    "same_top_p_pct": rf"Same top p:\s*{NUM}\s*±\s*{NUM}\s*%",
}
SIZE = re.compile(r"file size\s*=\s*" + NUM + r"\s*(MiB|GiB)\s*\(" + NUM + r"\s*BPW\)")
FTYPE = re.compile(r"file type\s*=\s*(.+?)\s*$", re.M)
FINAL = re.compile(r"Final estimate: PPL = " + NUM + r" \+/- " + NUM)
CHUNKS = re.compile(r"computing over (\d+) chunks, n_ctx=(\d+)")


def parse_log(text: str) -> dict:
    out: dict = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2:
            out[key] = float(m.group(1))
            out[key + "_unc"] = float(m.group(2))
        else:
            out[key] = float(m.group(1))
    m = SIZE.search(text)
    if m:
        size, unit, bpw = float(m.group(1)), m.group(2), float(m.group(3))
        out["file_size_mib"] = round(size * 1024 if unit == "GiB" else size, 2)
        out["bpw"] = bpw
    m = FTYPE.search(text)
    if m:
        out["file_type"] = m.group(1)
    m = FINAL.search(text)
    if m:
        out["ppl_final"] = float(m.group(1))
        out["ppl_final_unc"] = float(m.group(2))
    m = CHUNKS.search(text)
    if m:
        out["chunks_scored"] = int(m.group(1))
        out["n_ctx"] = int(m.group(2))
    return out


def sha256_sidecar(path: str) -> str | None:
    side = path + ".sha256"
    if os.path.exists(side):
        return open(side).read().split()[0]
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_report(model: str, kl_dir: str, models_dir: str | None, text: str | None, ctx: int | None,
                 chunks: int | None) -> dict:
    base_log = os.path.join(kl_dir, "BF16.log")
    base = parse_log(open(base_log).read()) if os.path.exists(base_log) else {}
    if models_dir:
        base["sha256"] = sha256_sidecar(os.path.join(models_dir, f"{model}-BF16.gguf"))
    variants = {}
    for log in sorted(glob.glob(os.path.join(kl_dir, "*.log"))):
        v = os.path.basename(log)[:-4]
        if v == "BF16":
            continue
        row = parse_log(open(log).read())
        if models_dir:
            row["sha256"] = sha256_sidecar(os.path.join(models_dir, f"{model}-{v}.gguf"))
        row["complete"] = "kld_mean" in row and "same_top_p_pct" in row
        variants[v] = row

    def kld(v):
        return (variants.get(v) or {}).get("kld_mean")

    checks = {}
    if kld("Q8_0") is not None:
        for other in ("Q4_K_M", "Q4_0-pure"):
            if kld(other) is not None:
                checks[f"q8_0_closer_than_{other.lower().replace('-', '_')}"] = kld("Q8_0") < kld(other)
    return {
        "model": model,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": "llama-perplexity --kl-divergence (BF16 base)",
        "eval_text": text,
        "eval_text_sha256": sha256_file(text) if text and os.path.exists(text) else None,
        "ctx": ctx,
        "chunks": chunks,
        "bf16": base,
        "variants": variants,
        "checks": checks,
        "complete": bool(variants) and all(r["complete"] for r in variants.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models-dir")
    ap.add_argument("--text")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--chunks", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = build_report(a.model, a.dir, a.models_dir, a.text, a.ctx, a.chunks)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, a.out)
    for v, r in rep["variants"].items():
        print(f"{a.model} {v}: KLD {r.get('kld_mean')} ± {r.get('kld_mean_unc')}, same top {r.get('same_top_p_pct')}%,"
              f" PPL {r.get('ppl_q')} vs {r.get('ppl_base')}, {r.get('bpw')} BPW")
    print("checks:", rep["checks"], "complete:", rep["complete"])


if __name__ == "__main__":
    main()
