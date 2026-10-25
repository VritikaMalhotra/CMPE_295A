"""R3 toy QAD for M1: the S7 spike (spikes/s7_toy_qad_resume.py) with two changes only:
the fake quantizer now matches llama.cpp's Q4_0 bit for bit, and the model path is an argument.

Original S7 docstring follows.

Spike S7: 50-step toy QAD run (Qwen3-0.6B, Q4_0 fake quantization with a straight-through
estimator, KL to the model's own BF16 outputs), checkpointed every 5 steps so a killed run
can resume and reproduce the uninterrupted loss curve.

Pilot run: Google Colab A100-40GB, torch 2.11.0+cu130, transformers 5.18.0, 2026-10-08 PT.
Pilot / development data, not study results. Result: all 50 per-step losses identical
between the uninterrupted run and the run killed after step 27 and resumed from step 25.

    python s7_toy_qad_resume.py --out runA --data seqs.pt
    bash s7_kill_and_resume.sh
"""
import os, sys, json, time, math, argparse, random
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--ckpt-every", type=int, default=5); ap.add_argument("--data", required=True); ap.add_argument("--model", default="/content/work/Qwen3-0.6B"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); CK = f"{a.out}/ckpt.pt"; LOG = f"{a.out}/loss.jsonl"
torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
MP = a.model; dev = "cuda"
def q4_0_fake(w):
    """llama.cpp quantize_row_q4_0_ref, exactly: float32, d = signed amax / -8, q = min(15, (int8)(x*id + 8.5)),
    d stored as fp16. (The S7 spike used torch.round, which rounds halves to even: ~0.13% of weights off by one level.)"""
    o, i = w.shape; wb = w.float().reshape(o, i // 32, 32)
    idx = wb.abs().argmax(-1, keepdim=True); mx = torch.gather(wb, -1, idx)
    d = mx / -8.0; inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = torch.clamp(torch.trunc(wb * inv + 8.5), 0, 15)
    return ((q - 8) * d.half().float()).reshape(o, i).to(w.dtype)
class FQLinear(nn.Module):
    def __init__(self, lin): super().__init__(); self.lin = lin
    def forward(self, x):
        w = self.lin.weight; wq = w + (q4_0_fake(w) - w).detach()   # straight-through estimator
        return F.linear(x, wq, self.lin.bias)
def wrap(m):
    for n, c in m.named_children():
        if isinstance(c, nn.Linear) and c.in_features % 32 == 0: setattr(m, n, FQLinear(c))
        else: wrap(c)
torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(MP)
teacher = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).to(dev).eval()
for p in teacher.parameters(): p.requires_grad_(False)
student = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).to(dev)
wrap(student.model)   # decoder blocks; tied embedding / lm_head left in BF16 for this toy run
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0)
seqs = torch.load(a.data)            # [N, 512] token ids, fixed order
B = 4; start = 0
if os.path.exists(CK):
    ck = torch.load(CK, weights_only=False)
    student.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"])
    print(json.dumps({"event": "resumed", "from_step": start, "t": time.time()}), flush=True)
    # drop log lines past the checkpoint (steps that will be recomputed)
    if os.path.exists(LOG):
        keep = [l for l in open(LOG) if json.loads(l)["step"] <= start]; open(LOG, "w").writelines(keep)
torch.cuda.reset_peak_memory_stats()
for step in range(start + 1, a.steps + 1):
    x = seqs[(step - 1) * B: step * B].to(dev)
    with torch.no_grad(): tl = teacher(input_ids=x).logits.float()
    sl = student(input_ids=x).logits.float()
    loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean") / x.shape[1]
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
    with open(LOG, "a") as f: f.write(json.dumps({"step": step, "loss": loss.item(), "t": time.time(), "peak_alloc_mib": torch.cuda.max_memory_allocated() // 2**20}) + "\n")
    print(f"step {step} loss {loss.item():.6f}", flush=True)
    if step % a.ckpt_every == 0:
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "step": step, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, CK + ".tmp")
        os.replace(CK + ".tmp", CK)
print(json.dumps({"event": "done", "peak_alloc_mib": torch.cuda.max_memory_allocated() // 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20}), flush=True)
