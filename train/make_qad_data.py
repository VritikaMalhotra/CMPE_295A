"""Build the token sequences for the R3 toy QAD run (C7 data step, minimal M1 version).

The Oct 8 S7 spike's ``seqs.pt`` was not kept, so M1 regenerates a fixed, documented set:
WikiText-2 (raw) train split, concatenated with "\n\n", tokenized with the Qwen3-0.6B tokenizer, cut into
non-overlapping 512-token sequences in file order. The first 200 feed the 50 x 4 training steps; the next 16
are held out for export_qad.py's KL check. The SHA-256 of the saved tensor goes into ``seqs.json``.

M2 replaces this with the agentic calibration data (C7 proper).

    python train/make_qad_data.py --tokenizer /content/work/Qwen3-0.6B --out /content/work/s7/seqs.pt
"""
import argparse
import hashlib
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=216)
ap.add_argument("--seq-len", type=int, default=512)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
need = a.n * a.seq_len
ids: list[int] = []
for line in ds["text"]:
    if not line.strip():
        continue
    ids += tok(line, add_special_tokens=False)["input_ids"]
    if len(ids) >= need:
        break
seqs = torch.tensor(ids[:need], dtype=torch.long).reshape(a.n, a.seq_len)
torch.save(seqs, a.out)
digest = hashlib.sha256(seqs.numpy().tobytes()).hexdigest()
meta = {"source": "Salesforce/wikitext wikitext-2-raw-v1 train", "n": a.n, "seq_len": a.seq_len,
        "train": "first 200 (50 steps x batch 4)", "heldout": "last 16", "sha256_int64": digest}
json.dump(meta, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=2)
print(json.dumps(meta))
