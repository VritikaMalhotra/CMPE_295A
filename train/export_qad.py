"""R3 export (C8, the S8 path): S7 toy-QAD checkpoint -> HF model -> BF16 GGUF -> pure Q4_0 GGUF, plus checks.

Owner: Yashashav Devalapalli Kamalraj.

S7 wrapped each decoder Linear in ``FQLinear(lin)``, so checkpoint keys read ``...q_proj.lin.weight``.
This script maps them back to a plain Qwen3 model, saves it, converts with llama.cpp, quantizes with
``--pure Q4_0`` (the same recipe as the pilot's Q4_0 file), and checks the export two ways:

1. **Weight fidelity.** Every decoder Linear in the GGUF, dequantized, must equal ``q4_0_fake`` of the trained
   weight (the quantizer the QAD loss saw, bit-exact with llama.cpp). Reports the max diff relative to the
   tensor's largest quantization step.
2. **KL to BF16.** On held-out sequences, KL(teacher BF16 || student-fake-quant) for the QAD checkpoint
   vs. for plain Q4_0 of the base model. QAD should not be worse; with S7's 50 toy steps the gain is small.

Note: ``--pure Q4_0`` also quantizes the token embedding/lm_head, which S7 kept in BF16. That mismatch is the
same for the R3 file and the pilot Q4_0 file, so the R3-vs-Q4_0 comparison stays like-for-like.

    python train/export_qad.py --base /content/work/Qwen3-0.6B --ckpt runA/ckpt.pt --out out/r3 \
        --llama /content/llama.cpp --data seqs.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


def q4_0_fake(w):
    """llama.cpp quantize_row_q4_0_ref, exactly (same function as train/qad_toy.py): float32,
    d = signed amax / -8, q = min(15, (int8)(x*id + 8.5)), d rounded through fp16 as stored."""
    import torch
    o, i = w.shape
    wb = w.float().reshape(o, i // 32, 32)
    idx = wb.abs().argmax(-1, keepdim=True)
    mx = torch.gather(wb, -1, idx)
    d = mx / -8.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = torch.clamp(torch.trunc(wb * inv + 8.5), 0, 15)
    return ((q - 8) * d.half().float()).reshape(o, i)


def remap(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        out[k.replace(".lin.weight", ".weight").replace(".lin.bias", ".bias")] = v
    return out


def export_hf(base: str, ckpt: str, out_hf: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    sd = remap(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]          # tied to embed_tokens in Qwen3-0.6B
    if missing or unexpected:
        sys.exit(f"checkpoint/model mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.save_pretrained(out_hf, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_hf)
    return model


def to_gguf(out_hf: str, out_dir: str, llama: str, name: str) -> tuple[str, str]:
    bf16 = os.path.join(out_dir, f"{name}-BF16.gguf")
    q4 = os.path.join(out_dir, f"{name}-Q4_0-pure.gguf")
    subprocess.run([sys.executable, os.path.join(llama, "convert_hf_to_gguf.py"), out_hf, "--outtype", "bf16",
                    "--outfile", bf16], check=True)
    qbin = shutil.which("llama-quantize") or next(p for p in (
        os.path.join(llama, "build-cuda", "bin", "llama-quantize"), os.path.join(llama, "build-cpu", "bin", "llama-quantize"),
        os.path.join(llama, "build", "bin", "llama-quantize")) if os.path.exists(p))
    subprocess.run([qbin, "--pure", bf16, q4, "Q4_0"], check=True)
    return bf16, q4


def check_weights(model, q4_path: str, llama: str, max_tensors: int | None = None) -> dict:
    """Dequantize the GGUF's decoder Linear tensors and compare with q4_0_fake of the HF weights."""
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(llama, "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    hf = dict(model.named_parameters())
    names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj"}
    worst, n = 0.0, 0
    per = {}
    for t in GGUFReader(q4_path).tensors:
        parts = t.name.split(".")                  # blk.<i>.<kind>.weight
        if parts[0] != "blk" or parts[2] not in names:
            continue
        w = hf[f"model.layers.{parts[1]}.{names[parts[2]]}.weight"].detach()
        ref = q4_0_fake(w).float().numpy()
        got = dequantize(t.data, t.tensor_type).reshape(ref.shape).astype(np.float32)
        # relative to each block's step d, so the check does not depend on weight scale; with the exact
        # quantizer the only gap is float rounding of d*(q-8)
        dmax = float(np.abs(ref).max()) / 8 or 1.0
        diff = float(np.abs(got - ref).max()) / dmax
        per[t.name] = diff
        worst = max(worst, diff)
        n += 1
        if max_tensors and n >= max_tensors:
            break
    # pass = no weight is a whole quantization level away (that would be 1/8 of the max step, 0.125)
    return {"tensors_checked": n, "max_rel_diff": worst, "pass": worst < 1e-3, "worst": sorted(per.items(),
            key=lambda kv: -kv[1])[:3]}


def kl_check(base: str, model, data: str, n_seq: int = 16) -> dict:
    """Mean per-token KL(teacher || student) with S7's fake quant applied, QAD vs. plain Q4_0 of the base."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = torch.load(data)[-n_seq:]                 # S7 trained on the first 50*4 sequences; use the tail
    teacher = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(dev).eval()

    def fq_copy(m):
        m = m.to(dev).eval()
        with torch.no_grad():
            for name, mod in m.model.named_modules():
                if isinstance(mod, torch.nn.Linear) and mod.in_features % 32 == 0:
                    mod.weight.copy_(q4_0_fake(mod.weight).to(mod.weight.dtype))
        return m

    plain = fq_copy(AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16))
    qad = fq_copy(model)
    out = {"plain_q4_0": 0.0, "qad_q4_0": 0.0}
    with torch.no_grad():
        for i in range(0, len(seqs), 4):
            x = seqs[i:i + 4].to(dev)
            tl = F.log_softmax(teacher(input_ids=x).logits.float(), -1)
            for k, m in (("plain_q4_0", plain), ("qad_q4_0", qad)):
                sl = F.log_softmax(m(input_ids=x).logits.float(), -1)
                out[k] += F.kl_div(sl, tl, log_target=True, reduction="batchmean").item() / x.shape[1]
    nb = (len(seqs) + 3) // 4
    out = {k: v / nb for k, v in out.items()}
    out["qad_not_worse"] = out["qad_q4_0"] <= out["plain_q4_0"] * 1.01
    out["n_seq"] = int(len(seqs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF Qwen3-0.6B directory")
    ap.add_argument("--ckpt", required=True, help="S7 ckpt.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama", required=True, help="llama.cpp checkout (convert script, gguf-py, build)")
    ap.add_argument("--data", help="S7 seqs.pt, for the KL check")
    ap.add_argument("--name", default="Qwen3-0.6B-S7QAD")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hf_dir = os.path.join(a.out, "hf")
    model = export_hf(a.base, a.ckpt, hf_dir)
    bf16, q4 = to_gguf(hf_dir, a.out, a.llama, a.name)
    report = {"ckpt": a.ckpt, "bf16_gguf": bf16, "q4_0_gguf": q4, "weights": check_weights(model, q4, a.llama)}
    if a.data:
        report["kl"] = kl_check(a.base, model, a.data)
    with open(os.path.join(a.out, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["weights"]["pass"]:
        sys.exit("weight fidelity check failed")


if __name__ == "__main__":
    main()
