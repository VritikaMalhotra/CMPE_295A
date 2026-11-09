#!/usr/bin/env python3
"""R2b layer-selective variant (Workbook 2 FR-E10, Table 19; Ch. 3 "Layer sensitivity").

Rank the linear weights of a model by a causal test, then build Q4_0 everywhere except the most sensitive
weights at Q8_0, at a whole-file budget of 5.5 bits per weight (FR-E10 acceptance: within +/-0.1).

The causal test: quantize ONE tensor to Q4_0, keep every other tensor in BF16, and measure the mean KL
divergence from the BF16 model on calibration text that shares no 13-gram with the benchmarks (FR-E5,
scripts/benchmark_ngrams.py). llama-quantize applies --tensor-type overrides only on a quantized base type and
in command-line order, first match wins (src/llama-quant.cpp at d812350), so one tensor at Q4_0 with the rest
in BF16 is:  llama-quantize --pure --tensor-type '^<name>$=q4_0' --tensor-type '.=bf16' BF16.gguf out.gguf Q4_0

    python scripts/rank_tensors.py list   --model Qwen3-0.6B            # candidates -> rank/<model>/tensors.json
    python scripts/rank_tensors.py base   --model Qwen3-0.6B            # BF16 logits on the ranking text (GPU job)
    python scripts/rank_tensors.py run    --model Qwen3-0.6B --task 0 --ntasks 8   # one array task's share
    python scripts/rank_tensors.py select --model Qwen3-0.6B --bpw 5.5  # ranking + tensor-type file for the build
Jobs: slurm/rank_tensors.sbatch (array); build: scripts/make_layer_selective.sh. Owner: Yashashav (R2).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

BITS = {"Q4_0": 4.5, "Q8_0": 8.5, "BF16": 16.0, "F16": 16.0, "F32": 32.0}
KLD = re.compile(r"Mean\s+KLD:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*±\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        sys.exit(f"{name} is not set: source scripts/hpc_env.sh first")
    return v


def paths(model: str) -> dict:
    out = os.environ.get("RANK_OUT") or os.path.join(env("TESTBED_OUT"), "rank", model)
    m = env("TESTBED_MODELS")
    llama = env("LLAMA_DIR")
    build = os.environ.get("RANK_BUILD", "build-cuda")
    return {
        "out": out, "bf16": os.path.join(m, f"{model}-BF16.gguf"), "tensors": os.path.join(out, "tensors.json"),
        "base": os.path.join(out, "BF16.kld"), "scores": os.path.join(out, "scores"),
        "quantize": os.environ.get("LLAMA_QUANTIZE") or os.path.join(llama, build, "bin", "llama-quantize"),
        "perplexity": os.environ.get("LLAMA_PERPLEXITY") or os.path.join(llama, build, "bin", "llama-perplexity"),
        "text": os.environ.get("RANK_TEXT") or os.path.join(env("CMPE295_GROUP"), "data", "calib", "calib.txt"),
        "gguf_py": os.path.join(llama, "gguf-py"),
    }


def ppl_args() -> list[str]:
    return ["-c", os.environ.get("RANK_CTX", "512"), "--chunks", os.environ.get("RANK_CHUNKS", "20"),
            "-ngl", os.environ.get("RANK_NGL", "99"), "-t", os.environ.get("TESTBED_CPU_THREADS", "4")]


# ---------------------------------------------------------------- pure logic (unit-tested) ------------
def is_candidate(name: str, shape: list[int]) -> bool:
    """Linear weights: 2-D tensors inside the blocks. Token embedding and output head stay at the base type
    (Q4_0 with --pure), as in the Q4_0 control; norms and other 1-D tensors are never quantized."""
    return name.startswith("blk.") and name.endswith(".weight") and len([d for d in shape if d > 1]) >= 2


def select(tensors: list[dict], scores: dict[str, float], target_bpw: float = 5.5) -> dict:
    """Promote candidates to Q8_0 in order of decreasing KL damage until the predicted whole-file bits per
    weight reaches target_bpw. `tensors`: [{name, n, candidate, base_bits}] for every tensor in the file.
    Predicted bits ignore the GGUF header (small; the build reports the measured value)."""
    missing = [t["name"] for t in tensors if t["candidate"] and t["name"] not in scores]
    if missing:
        raise ValueError(f"{len(missing)} candidates have no score, e.g. {missing[:3]}")
    total_n = sum(t["n"] for t in tensors)
    bits = sum(t["n"] * t["base_bits"] for t in tensors)
    order = sorted((t for t in tensors if t["candidate"]), key=lambda t: (-scores[t["name"]], t["name"]))
    lin_n = sum(t["n"] for t in order)
    chosen, chosen_n = [], 0
    for t in order:
        if bits / total_n >= target_bpw:
            break
        bits += t["n"] * (BITS["Q8_0"] - t["base_bits"])
        chosen.append(t["name"])
        chosen_n += t["n"]
    return {"target_bpw": target_bpw, "predicted_bpw": bits / total_n, "n_chosen": len(chosen),
            "share_of_linear_weights": chosen_n / lin_n if lin_n else 0.0, "chosen": chosen,
            "ranking": [{"name": t["name"], "kld": scores[t["name"]], "n": t["n"]} for t in order]}


def tensor_type_lines(chosen: list[str], qtype: str = "q8_0") -> list[str]:
    return [f"^{re.escape(n)}$={qtype}" for n in chosen]


def parse_kld(log: str) -> tuple[float, float]:
    m = KLD.search(log)
    if not m:
        raise ValueError("no 'Mean KLD' line in llama-perplexity output")
    return float(m.group(1)), float(m.group(2))


# ---------------------------------------------------------------- commands -----------------------------
def cmd_list(a) -> None:
    p = paths(a.model)
    sys.path.insert(0, p["gguf_py"])
    from gguf import GGUFReader  # llama.cpp's own reader at the pinned commit
    r = GGUFReader(p["bf16"])
    rows = []
    for t in r.tensors:
        shape = [int(x) for x in t.shape]
        cand = is_candidate(t.name, shape)
        tname = t.tensor_type.name
        base = "Q4_0" if (cand or t.name in ("token_embd.weight", "output.weight")) else tname
        rows.append({"name": t.name, "shape": shape, "n": int(t.n_elements), "type_in_bf16_file": tname,
                     "candidate": cand, "base_bits": BITS.get(base, 16.0)})
    os.makedirs(p["out"], exist_ok=True)
    with open(p["tensors"], "w") as f:
        json.dump({"model": a.model, "file": p["bf16"], "tensors": rows}, f, indent=1)
    c = [x for x in rows if x["candidate"]]
    print(f"{a.model}: {len(rows)} tensors, {len(c)} candidates ({sum(x['n'] for x in c) / 1e6:.1f}M weights) "
          f"-> {p['tensors']}")


def cmd_base(a) -> None:
    p = paths(a.model)
    os.makedirs(p["out"], exist_ok=True)
    cmd = [p["perplexity"], "-m", p["bf16"], "-f", p["text"], *ppl_args(), "--kl-divergence-base", p["base"]]
    with open(os.path.join(p["out"], "BF16.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print("base logits ->", p["base"])


def cmd_run(a) -> None:
    p = paths(a.model)
    if not os.path.exists(p["base"]):
        sys.exit(f"no base logits at {p['base']}: run `rank_tensors.py base` first")
    with open(p["tensors"]) as f:
        cands = [t for t in json.load(f)["tensors"] if t["candidate"]]
    mine = [t for i, t in enumerate(cands) if i % a.ntasks == a.task]
    os.makedirs(p["scores"], exist_ok=True)
    scratch = os.environ.get("RANK_SCRATCH") or os.path.join(p["out"], "tmp")
    os.makedirs(scratch, exist_ok=True)
    for t in mine:
        dst = os.path.join(p["scores"], t["name"] + ".json")
        if os.path.exists(dst):
            continue
        gg = os.path.join(scratch, f"{a.model}-{t['name']}-Q4_0only.gguf")
        t0 = time.time()
        q = subprocess.run([p["quantize"], "--pure", "--tensor-type", f"^{re.escape(t['name'])}$=q4_0",
                            "--tensor-type", ".=bf16", p["bf16"], gg, "Q4_0"], capture_output=True, text=True)
        if q.returncode != 0:
            sys.exit(f"llama-quantize failed for {t['name']}:\n{q.stdout[-2000:]}{q.stderr[-2000:]}")
        r = subprocess.run([p["perplexity"], "-m", gg, *ppl_args(), "--kl-divergence-base", p["base"],
                            "--kl-divergence"], capture_output=True, text=True)
        os.remove(gg)
        if r.returncode != 0:
            sys.exit(f"llama-perplexity failed for {t['name']}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        kld, unc = parse_kld(r.stdout + r.stderr)
        rec = {"name": t["name"], "n": t["n"], "kld_mean": kld, "kld_unc": unc, "seconds": round(time.time() - t0, 1),
               "ctx": int(os.environ.get("RANK_CTX", "512")), "chunks": int(os.environ.get("RANK_CHUNKS", "20")),
               "text": p["text"]}
        with open(dst + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(dst + ".tmp", dst)
        print(f"{t['name']}: KLD {kld:.5f} ± {unc:.5f} ({rec['seconds']} s)", flush=True)


def cmd_select(a) -> None:
    p = paths(a.model)
    with open(p["tensors"]) as f:
        tensors = json.load(f)["tensors"]
    scores = {}
    for fn in glob.glob(os.path.join(p["scores"], "*.json")):
        with open(fn) as f:
            r = json.load(f)
        scores[r["name"]] = r["kld_mean"]
    sel = select(tensors, scores, a.bpw)
    sel["model"] = a.model
    with open(os.path.join(p["out"], "selection.json"), "w") as f:
        json.dump(sel, f, indent=1)
    with open(os.path.join(p["out"], "tensor_types.txt"), "w") as f:
        f.write("\n".join(tensor_type_lines(sel["chosen"])) + "\n")
    print(f"{a.model}: {sel['n_chosen']} tensors to Q8_0 ({sel['share_of_linear_weights']:.1%} of linear weights), "
          f"predicted {sel['predicted_bpw']:.3f} bpw -> {p['out']}/tensor_types.txt")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "base", "run", "select"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        if name == "run":
            s.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
            s.add_argument("--ntasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
        if name == "select":
            s.add_argument("--bpw", type=float, default=5.5)
    a = ap.parse_args(argv)
    {"list": cmd_list, "base": cmd_base, "run": cmd_run, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
