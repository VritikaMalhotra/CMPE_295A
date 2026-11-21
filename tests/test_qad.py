"""R3 trainer (train/qad.py): quantizer, fake-quant coverage on a tied model, and bit-identical resume.

Needs torch + transformers (the training env); the resume test needs a GPU. Run on the cluster:
    /scratch/cmpe295-bertin/envs/qad-data/bin/python -m pytest -q tests/test_qad.py
"""
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))
import qad  # noqa: E402


def tiny_model(path, tied=True):
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                                   tie_word_embeddings=tied, max_position_embeddings=256)
    torch.manual_seed(0)
    m = transformers.Qwen3ForCausalLM(cfg)
    m.save_pretrained(path)
    return m


def test_q4_0_fake_matches_gguf_reference():
    llama = os.environ.get("LLAMA_DIR")
    if not llama or not os.path.isdir(os.path.join(llama, "gguf-py")):
        pytest.skip("LLAMA_DIR/gguf-py not available")
    sys.path.insert(0, os.path.join(llama, "gguf-py"))
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize, quantize
    torch.manual_seed(1)
    w = (torch.randn(8, 256) * 0.05).float()
    w[0, :32] = 0.0                                   # an all-zero block (d = 0)
    ref = dequantize(quantize(w.numpy(), GGMLQuantizationType.Q4_0), GGMLQuantizationType.Q4_0)
    np.testing.assert_array_equal(qad.q4_0_fake(w).numpy(), ref.reshape(8, 256))


def test_fake_quant_covers_tied_head(tmp_path):
    m = tiny_model(tmp_path / "m")
    orig = m.model.embed_tokens.weight.detach().clone()
    names, distinct = qad.add_fake_quant(m)
    # 2 layers x 7 linears + embedding; the tied head is parametrized but is the same tensor
    assert distinct == 2 * 7 + 1
    assert "lm_head" in names and "model.embed_tokens" in names
    qad.FakeQuant.on = True
    expect = qad.q4_0_fake(orig)
    assert torch.equal(m.lm_head.weight.detach(), expect)
    assert torch.equal(m.model.embed_tokens.weight.detach(), expect)
    qad.FakeQuant.on = False
    assert torch.equal(m.lm_head.weight.detach(), orig)
    qad.FakeQuant.on = True


def test_straight_through_gradient(tmp_path):
    m = tiny_model(tmp_path / "m")
    qad.add_fake_quant(m)
    x = torch.randint(0, 256, (1, 16))
    m(input_ids=x).logits.sum().backward()
    g = m.model.layers[0].mlp.up_proj.parametrizations.weight.original.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def _pack(path, n=40, seq=32):
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, k in (("train", n), ("eval", 4)):
        np.save(os.path.join(path, f"{split}.npy"), rng.integers(0, 256, (k, seq), dtype=np.uint32))
    json.dump({"train": {"sha256": "test"}, "eval": {}}, open(os.path.join(path, "pack.json"), "w"))


def _train(model, pack, out, steps, arm="klce", stop_at=-1):
    cmd = [sys.executable, os.path.join(REPO, "train", "qad.py"), "train", "--model-dir", str(model), "--pack",
           str(pack), "--arm", arm, "--out", str(out), "--micro-batch", "2", "--grad-accum", "2", "--lr", "1e-3",
           "--warmup", "2", "--max-steps", str(steps), "--eval-every", "100", "--eval-seqs", "4",
           "--stop-at", str(stop_at)]
    subprocess.run(cmd, check=True)
    return [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resume_reproduces_losses(tmp_path):
    tiny_model(tmp_path / "m")
    _pack(tmp_path / "p")
    full = _train(tmp_path / "m", tmp_path / "p", tmp_path / "a", 6)
    # the same run cut at step 3 (as a walltime signal would), then resumed to 6
    _train(tmp_path / "m", tmp_path / "p", tmp_path / "b", 6, stop_at=3)
    cut = _train(tmp_path / "m", tmp_path / "p", tmp_path / "b", 6)
    la = [r["loss"] for r in full if "loss" in r]
    lb = [r["loss"] for r in cut if "loss" in r]
    assert la == lb, (la, lb)
    assert la[-1] < la[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_control_trains_without_fake_quant(tmp_path):
    tiny_model(tmp_path / "m")
    _pack(tmp_path / "p")
    rec = _train(tmp_path / "m", tmp_path / "p", tmp_path / "c", 2, arm="control")
    first = rec[0]                                  # step-0 eval: the untrained student
    # the untrained 16-bit student equals the teacher up to fp32-vs-bf16 norm weights; Q4_0 is far from it
    assert first["kl_16bit"] < 1e-3
    assert first["kl_q4_0"] > 10 * first["kl_16bit"]
