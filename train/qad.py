#!/usr/bin/env python3
"""R3 quantization-aware distillation (Workbook 2 FR-E11; pre-registration §2): the three arms on one model.

* ``kl``      QAD-KL: the student sees Q4_0 fake-quantized weights and minimises KL(teacher || student) to its
              own frozen BF16 copy.
* ``klce``    QAD-KL+CE: the same plus ``ce_weight`` x next-token cross-entropy on the training mix.
* ``control`` train-longer: QAD-KL+CE with fake quantization **off** (quantized only at export). Same data, order,
              tokens, steps, optimizer and seed as ``klce``, so the two differ only in training under quantization.

Fake quantization reproduces llama.cpp's ``quantize_row_q4_0_ref`` bit for bit (``q4_0_fake``, as in
train/qad_toy.py) with a straight-through estimator, on every weight that ``llama-quantize --pure Q4_0`` quantizes:
each ``nn.Linear`` and ``nn.Embedding`` whose row length is a multiple of 32 (a tied output head is the embedding
matrix, so it is covered once). Master weights are fp32 and compute is BF16 autocast, so small updates are not
lost to BF16 rounding; the export goes through an f32 GGUF, so the Q4_0 codes in the deployed file are exactly the
codes the student trained against (train/export_r3.py checks this).

Subcommands:

    # once per model (login node or CPU job): tokenize and pack the QAD mix and the eval conversations
    python train/qad.py pack  --model-dir $TESTBED_MODELS/hf/Qwen3-0.6B --out $CMPE295_GROUP/data/qad_pack/Qwen3-0.6B
    # one arm (GPU job, slurm/qad_train.sbatch; resumes from <out>/ckpt.pt)
    python train/qad.py train --model-dir ... --pack ... --arm klce --out $TESTBED_OUT/r3/Qwen3-0.6B/klce
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time

MIX = "/scratch/cmpe295-bertin/data/qad_mix/train.jsonl"
EVAL = "/scratch/cmpe295-bertin/data/qad_mix/rank_conversations.jsonl"   # disjoint from train.jsonl (manifest)
ARMS = ("kl", "klce", "control")


# ---------------------------------------------------------------- quantizer
def q4_0_fake(w):
    """llama.cpp quantize_row_q4_0_ref, exactly: float32, d = signed amax / -8, q = min(15, (int8)(x*id + 8.5)),
    d rounded through fp16 as stored. Rows are split into blocks of 32 along the last dimension, as in GGUF."""
    import torch
    o, i = w.shape
    wb = w.float().reshape(o, i // 32, 32)
    idx = wb.abs().argmax(-1, keepdim=True)
    mx = torch.gather(wb, -1, idx)
    d = mx / -8.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = torch.clamp(torch.trunc(wb * inv + 8.5), 0, 15)
    return ((q - 8) * d.half().float()).reshape(o, i)


class FakeQuant:
    """Parametrization factory; ``FakeQuant.on`` switches every instance (the control arm trains with it off and
    turns it on only to evaluate 'after plain Q4_0')."""
    on = True

    @staticmethod
    def module():
        import torch.nn as nn

        class _FQ(nn.Module):
            def forward(self, w):
                if not FakeQuant.on:
                    return w
                return w + (q4_0_fake(w).to(w.dtype) - w).detach()     # straight-through estimator
        return _FQ()


def add_fake_quant(model) -> tuple[list[str], int]:
    """Register the Q4_0 parametrization on every weight --pure Q4_0 quantizes. Returns the module names and the
    number of distinct weight tensors: a tied output head shares the embedding's Parameter, so it gets its own
    parametrization (otherwise it would compute with the unquantized tensor) but counts once, as in the GGUF,
    which stores only token_embd."""
    import torch.nn as nn
    from torch.nn.utils import parametrize
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.Embedding)) and m.weight.dim() == 2 and m.weight.shape[1] % 32 == 0]
    distinct = len({id(m.weight) for _, m in mods})          # before parametrizing changes .weight
    for _, m in mods:
        parametrize.register_parametrization(m, "weight", FakeQuant.module(), unsafe=True)
    return [n for n, _ in mods], distinct


# ---------------------------------------------------------------- packing
def _normalize(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [tc if "function" in tc else
                               {"type": "function", "function": {"name": tc.get("name"), "arguments": tc.get("arguments", {})}}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


def render(tok, rec: dict) -> str | None:
    if rec["kind"] == "text":
        return rec["text"] + (tok.eos_token or "")
    try:
        return tok.apply_chat_template(_normalize(rec["messages"]), tools=rec.get("tools") or None, tokenize=False)
    except Exception:                    # a template that cannot render this record: counted, skipped
        return None


def cmd_pack(a) -> None:
    import numpy as np
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    os.makedirs(a.out, exist_ok=True)
    meta = {"model_dir": a.model_dir, "seq_len": a.seq_len, "tokenizer_vocab": len(tok)}
    for split, path, limit in (("train", a.mix, None), ("eval", a.eval, a.eval_seqs)):
        ids, n_rec, skipped = [], 0, 0
        for line in open(path):
            rec = json.loads(line)
            text = render(tok, rec)
            if text is None:
                skipped += 1
                continue
            ids.extend(tok(text, add_special_tokens=False)["input_ids"])
            n_rec += 1
            if limit and len(ids) >= limit * a.seq_len:
                break
        n = len(ids) // a.seq_len
        if limit:
            n = min(n, limit)
        arr = np.asarray(ids[: n * a.seq_len], dtype=np.uint32).reshape(n, a.seq_len)
        f = os.path.join(a.out, f"{split}.npy")
        np.save(f, arr)
        meta[split] = {"source": path, "source_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                       "records": n_rec, "skipped": skipped, "sequences": n, "tokens": int(n * a.seq_len),
                       "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()}
        print(f"{split}: {n_rec} records ({skipped} skipped) -> {n} x {a.seq_len} tokens", flush=True)
    json.dump(meta, open(os.path.join(a.out, "pack.json"), "w"), indent=1)


# ---------------------------------------------------------------- training
def kl_ce(student_logits, teacher_logits, labels, ce_weight: float):
    import torch.nn.functional as F
    s = F.log_softmax(student_logits.float(), -1)
    t = F.log_softmax(teacher_logits.float(), -1)
    kl = (t.exp() * (t - s)).sum(-1).mean()                     # per-token KL(teacher || student)
    if ce_weight == 0:
        return kl, kl, kl.new_zeros(())
    ce = F.nll_loss(s[:, :-1].reshape(-1, s.shape[-1]), labels[:, 1:].reshape(-1))
    return kl + ce_weight * ce, kl, ce


def copy_tokenizer(src: str, dst: str) -> None:
    """The tokenizer and chat-template files of the base model, which convert_hf_to_gguf.py needs next to the
    trained weights (save_pretrained writes only config and weights)."""
    import shutil
    for f in os.listdir(src):
        if (f.endswith((".safetensors", ".safetensors.index.json")) or f in ("config.json", "generation_config.json")
                or os.path.isdir(os.path.join(src, f))):        # a shard index would point the converter at missing shards
            continue
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def evaluate(student, teacher, ev, dev, batch: int = 2) -> dict:
    """Mean per-token KL to the teacher on the eval sequences, with fake quantization on (what the Q4_0 file
    will do) and off (the 16-bit student)."""
    import torch
    out = {}
    was = FakeQuant.on
    student.eval()
    for key, flag in (("kl_q4_0", True), ("kl_16bit", False)):
        FakeQuant.on = flag
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(ev), batch):
                x = ev[i:i + batch].to(dev)
                _, kl, _ = kl_ce(student(input_ids=x).logits, teacher(input_ids=x).logits, x, 0.0)
                tot += kl.item() * len(x)
                n += len(x)
        out[key] = tot / n
    FakeQuant.on = was
    student.train()
    return out


def cmd_train(a) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    ck, log = os.path.join(a.out, "ckpt.pt"), os.path.join(a.out, "train.jsonl")
    ce_w = 0.0 if a.arm == "kl" else a.ce_weight
    train = torch.from_numpy(np.load(os.path.join(a.pack, "train.npy")).astype(np.int64))
    ev = torch.from_numpy(np.load(os.path.join(a.pack, "eval.npy")).astype(np.int64))[: a.eval_seqs]
    seqs_per_step = a.micro_batch * a.grad_accum
    total = min(a.max_steps or 10**9, len(train) // seqs_per_step)
    cfg = {k: v for k, v in vars(a).items() if k != "func"}
    cfg.update(total_steps=total, tokens_per_step=seqs_per_step * train.shape[1], ce_weight_used=ce_w,
               pack=json.load(open(os.path.join(a.pack, "pack.json"))), torch=torch.__version__)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    teacher = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).to(dev)
    if a.grad_ckpt:
        student.gradient_checkpointing_enable()
    fq, n_distinct = add_fake_quant(student)
    FakeQuant.on = a.arm != "control"
    cfg["fake_quant_modules"], cfg["fake_quant_weights"] = len(fq), n_distinct
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step = 0
    if os.path.exists(ck):
        state = torch.load(ck, map_location="cpu", weights_only=False)
        if state["cfg"]["arm"] != a.arm or state["cfg"]["pack"]["train"]["sha256"] != cfg["pack"]["train"]["sha256"]:
            sys.exit(f"refusing to resume: {ck} is from another arm or data")
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        if os.path.exists(log):                  # steps after the checkpoint are recomputed
            keep = [l for l in open(log) if json.loads(l).get("step", 0) <= step]
            open(log, "w").writelines(keep)
        print(json.dumps({"event": "resumed", "step": step}), flush=True)
    else:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
        with open(log, "a") as f:
            f.write(json.dumps({"step": 0, **evaluate(student, teacher, ev, dev)}) + "\n")

    stop = {"flag": False}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("flag", True))

    def save(final: bool = False) -> None:
        state = {"model": student.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}
        torch.save(state, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        print(json.dumps({"event": "checkpoint", "step": step, "final": final}), flush=True)

    student.train()
    torch.cuda.reset_peak_memory_stats()
    last_save, t0 = time.time(), time.time()
    while step < total and not stop["flag"] and step != a.stop_at:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.lr, a.warmup)
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
        base = step * seqs_per_step
        for j in range(a.grad_accum):
            x = train[base + j * a.micro_batch: base + (j + 1) * a.micro_batch].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    tl = teacher(input_ids=x).logits
                loss, kl, ce = kl_ce(student(input_ids=x).logits, tl, x, ce_w)
            (loss / a.grad_accum).backward()
            for k, v in (("loss", loss), ("kl", kl), ("ce", ce)):
                acc[k] += v.item() / a.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        step += 1
        el = time.time() - t0
        rec = {"step": step, **acc, "grad_norm": gn, "lr": opt.param_groups[0]["lr"],
               "tokens": step * cfg["tokens_per_step"], "tok_per_s": cfg["tokens_per_step"] / max(el, 1e-9),
               "peak_mib": torch.cuda.max_memory_allocated() // 2**20, "t": time.time()}
        t0 = time.time()
        if step % a.eval_every == 0 or step == total:
            rec.update(evaluate(student, teacher, ev, dev))
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step == total:
            print(f"step {step}/{total} loss {acc['loss']:.5f} kl {acc['kl']:.5f} ce {acc['ce']:.4f} "
                  f"{rec['tok_per_s']:.0f} tok/s peak {rec['peak_mib']} MiB", flush=True)
        if time.time() - last_save > a.ckpt_minutes * 60:
            save()
            last_save = time.time()
    save(final=step >= total)
    if step >= total:
        from torch.nn.utils import parametrize
        for name in fq:                          # keep the trained (unquantized) master weights
            parametrize.remove_parametrizations(student.get_submodule(name), "weight", leave_parametrized=False)
        final = os.path.join(a.out, "final")
        student.save_pretrained(final, safe_serialization=True)
        copy_tokenizer(a.model_dir, final)
        for f in os.listdir(final):              # save_pretrained writes weights 0600; the group needs them
            os.chmod(os.path.join(final, f), 0o660)
        json.dump({"steps": step, "arm": a.arm, "fake_quant_weights": n_distinct, "fake_quant_modules": fq},
                  open(os.path.join(a.out, "final", "r3.json"), "w"), indent=1)
        print(json.dumps({"event": "done", "steps": step}), flush=True)
    else:
        print(json.dumps({"event": "stopped", "step": step}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("pack")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mix", default=MIX)
    p.add_argument("--eval", default=EVAL)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-seqs", type=int, default=64)
    p.set_defaults(func=cmd_pack)
    t = sub.add_parser("train")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--pack", required=True)
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--micro-batch", type=int, default=2)
    t.add_argument("--grad-accum", type=int, default=8)          # 2 x 8 x 2,048 = 32,768 tokens per step
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--warmup", type=int, default=50)
    t.add_argument("--ce-weight", type=float, default=0.1)       # 2026-11-21 pilot: 0.5 drifts from the teacher
    t.add_argument("--max-steps", type=int, default=0, help="0 = one pass over the packed mix")
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--eval-seqs", type=int, default=64)
    t.add_argument("--ckpt-minutes", type=float, default=30)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--stop-at", type=int, default=-1, help="checkpoint and stop at this step (tests a walltime cut)")
    t.add_argument("--seed", type=int, default=295)
    t.set_defaults(func=cmd_train)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
