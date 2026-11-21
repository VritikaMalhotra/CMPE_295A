#!/usr/bin/env python3
"""Export one R3 arm (train/qad.py ``final/``) to GGUF and check that the deployed Q4_0 file holds exactly the codes
the student trained against.

1. ``convert_hf_to_gguf.py --outtype f32``: the fp32 master weights, unrounded.
2. ``llama-quantize --pure <f32> <q4_0> Q4_0``: the deployed file (same recipe as the baseline Q4_0-pure files).
3. ``--outtype bf16`` as well, for the control arm's 16-bit evaluation (pre-registration Table 12, control (b)).
4. Fidelity: for every Q4_0 tensor in the Q4_0 file, ``q4_0_fake`` (train/qad.py) of the same-named f32 tensor
   must equal its dequantized values; and the number of Q4_0 tensors must equal the number of distinct weights the
   student fake-quantized (``final/r3.json``). Comparing two GGUFs by tensor name needs no HF-to-GGUF name map.

    python train/export_r3.py --final $TESTBED_OUT/r3/Qwen3-0.6B/klce/final --name Qwen3-0.6B-R3-QAD-KLCE \
        --out $TESTBED_MODELS [--bf16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qad import q4_0_fake  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fidelity(f32_path: str, q4_path: str, llama: str) -> dict:
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(llama, "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    ref = {t.name: t for t in GGUFReader(f32_path).tensors}
    worst, n, per = 0.0, 0, {}
    for t in GGUFReader(q4_path).tensors:
        if t.tensor_type.name != "Q4_0":
            continue
        r = ref[t.name]
        w = np.asarray(r.data, dtype=np.float32)
        shape = w.shape if w.ndim == 2 else tuple(int(x) for x in reversed(r.shape))
        w = w.reshape(shape)
        expect = q4_0_fake(torch.from_numpy(w)).numpy()
        got = dequantize(t.data, t.tensor_type).reshape(expect.shape).astype(np.float32)
        step = float(np.abs(expect).max()) / 8 or 1.0          # relative to the largest quantization step
        diff = float(np.abs(got - expect).max()) / step
        per[t.name] = diff
        worst = max(worst, diff)
        n += 1
    return {"q4_0_tensors": n, "max_rel_diff": worst, "pass": worst < 1e-3,
            "worst": sorted(per.items(), key=lambda kv: -kv[1])[:3]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="train/qad.py <out>/final")
    ap.add_argument("--name", required=True, help="file stem, e.g. Qwen3-0.6B-R3-QAD-KLCE")
    ap.add_argument("--out", required=True, help="directory for the GGUF files (usually $TESTBED_MODELS)")
    ap.add_argument("--llama", default=os.environ.get("LLAMA_DIR"))
    ap.add_argument("--bf16", action="store_true", help="also write a BF16 GGUF (control arm, 16-bit evaluation)")
    a = ap.parse_args()
    qbin = os.environ.get("LLAMA_QUANTIZE") or os.path.join(a.llama, "build-cpu", "bin", "llama-quantize")
    conv = os.path.join(a.llama, "convert_hf_to_gguf.py")
    f32 = os.path.join(a.out, f"{a.name}-F32.gguf")
    q4 = os.path.join(a.out, f"{a.name}-Q4_0-pure.gguf")
    subprocess.run([sys.executable, conv, a.final, "--outtype", "f32", "--outfile", f32], check=True)
    subprocess.run([qbin, "--pure", f32, q4, "Q4_0"], check=True)
    rep = {"final": a.final, "f32": f32, "q4_0": q4, "q4_0_sha256": sha256(q4)}
    if a.bf16:
        bf = os.path.join(a.out, f"{a.name}-BF16.gguf")
        subprocess.run([sys.executable, conv, a.final, "--outtype", "bf16", "--outfile", bf], check=True)
        rep.update(bf16=bf, bf16_sha256=sha256(bf))
    rep["fidelity"] = fid = fidelity(f32, q4, a.llama)
    meta = json.load(open(os.path.join(a.final, "r3.json")))
    rep["fake_quant_weights"] = meta["fake_quant_weights"]
    rep["count_match"] = fid["q4_0_tensors"] == meta["fake_quant_weights"]
    for p in (q4, rep.get("bf16")):
        if p:
            open(p + ".sha256", "w").write(f"{sha256(p)}  {os.path.basename(p)}\n")
    json.dump(rep, open(os.path.join(a.out, f"{a.name}.export.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("q4_0", "count_match", "fake_quant_weights")} | {"fidelity": fid}, indent=1))
    if not (fid["pass"] and rep["count_match"]):
        sys.exit("R3 export check failed: the Q4_0 file does not hold the codes the student trained against")


if __name__ == "__main__":
    main()
