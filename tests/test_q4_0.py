"""R3: the QAD fake quantizer must match llama.cpp's Q4_0 bit for bit (reviewer finding: torch.round
rounds halves to even, llama.cpp rounds them up). Reference = llama.cpp's gguf-py Q4_0 (mirrors
quantize_row_q4_0_ref). Skips when llama.cpp/gguf-py or torch is not present."""
import os
import sys

import numpy as np
import pytest

GGUF_PY = os.environ.get("GGUF_PY", os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp", "gguf-py"))
for cand in (GGUF_PY, os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py")):
    if os.path.isdir(cand):
        sys.path.insert(0, cand)
        break
gq = pytest.importorskip("gguf.quants")
torch = pytest.importorskip("torch")


def bf16(x):
    return torch.tensor(x).to(torch.bfloat16)


@pytest.mark.parametrize("std", [0.02, 0.05, 0.3])
def test_fake_quant_matches_llama_cpp(std):
    from train.export_qad import q4_0_fake
    w = bf16(np.random.default_rng(1).normal(0, std, (256, 1024)).astype(np.float32))
    ours = q4_0_fake(w).float().numpy()
    ref = gq.Q4_0.dequantize(gq.Q4_0.quantize(w.float().numpy())).astype(np.float32)
    assert np.array_equal(ours, ref), f"max diff {np.abs(ours - ref).max()}"


def test_training_and_export_quantizers_are_identical():
    import importlib.util
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "train", "qad_toy.py")).read()
    body = re.search(r"def q4_0_fake\(w\):.*?return [^\n]+", src, re.S).group(0)
    ns = {"torch": torch}
    exec(body, ns)
    from train.export_qad import q4_0_fake
    w = bf16(np.random.default_rng(2).normal(0, 0.03, (64, 256)).astype(np.float32))
    # training returns the weight in the model's dtype (bf16) for the forward pass; same levels and scales
    assert torch.equal(ns["q4_0_fake"](w), q4_0_fake(w).to(w.dtype))
