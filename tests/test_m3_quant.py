"""M3/R2 helpers: FR-E5 13-gram check and the R2b selection (scripts/benchmark_ngrams.py, scripts/rank_tensors.py)."""
import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ng = _load("benchmark_ngrams")
rk = _load("rank_tensors")

BENCH = "please book a flight from san francisco to new york on the fifth of may for two adults and one child"


def test_tokens_ignore_case_and_punctuation():
    assert ng.tokens("Book, a FLIGHT!") == ["book", "a", "flight"]


def test_clean_drops_a_line_containing_a_benchmark_13gram():
    lines = ["the cat sat on the mat", "Please book a flight from San Francisco to New York on the fifth of May!",
             "unrelated text"]
    kept, dropped = ng.clean(BENCH, lines)
    assert dropped == [1] and kept == [lines[0], lines[2]]


def test_clean_drops_both_lines_when_an_overlap_straddles_them():
    lines = ["please book a flight from san francisco", "to new york on the fifth of may", "fine"]
    kept, dropped = ng.clean(BENCH, lines)
    assert dropped == [0, 1] and kept == ["fine"]


def test_clean_keeps_text_with_only_shorter_overlaps():
    lines = ["book a flight from san francisco to new york on the"]   # 12 tokens of the benchmark
    kept, dropped = ng.clean(BENCH, lines)
    assert dropped == [] and kept == lines


def test_module_strings_collects_prompt_constants(tmp_path, monkeypatch):
    (tmp_path / "fake_prompts.py").write_text(
        'SYSTEM = "You are an expert in composing functions."\n'
        'TEMPLATES = {"classic": ["Use the tools.", "Reply briefly."]}\n'
        'LIMIT = 20\n_private_note = "kept too"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    found, note = ng.module_strings("fake_prompts")
    assert "You are an expert in composing functions." in found and "Use the tools." in found
    assert "kept too" in found and "classic" in found and 20 not in found
    assert note.startswith("fake_prompts:")
    assert ng.module_strings("no_such_module_xyz") == ([], "no_such_module_xyz unavailable (ModuleNotFoundError)")


def test_candidates_are_2d_block_weights_only():
    assert rk.is_candidate("blk.0.ffn_down.weight", [3072, 1024])
    assert not rk.is_candidate("blk.0.attn_norm.weight", [1024])
    assert not rk.is_candidate("token_embd.weight", [1024, 151936])
    assert not rk.is_candidate("output.weight", [1024, 151936])


def _toy(n_layers=8):
    t = [{"name": "token_embd.weight", "n": 4000, "candidate": False, "base_bits": 4.5}]
    for i in range(n_layers):
        for w, n in (("attn_q", 1000), ("ffn_down", 3000)):
            t.append({"name": f"blk.{i}.{w}.weight", "n": n, "candidate": True, "base_bits": 4.5})
        t.append({"name": f"blk.{i}.attn_norm.weight", "n": 10, "candidate": False, "base_bits": 32.0})
    return t


def test_select_reaches_the_budget_in_order_of_damage():
    tensors = _toy()
    scores = {x["name"]: (1.0 if "ffn_down" in x["name"] else 0.1) * (1 + int(x["name"].split(".")[1]))
              for x in tensors if x["candidate"]}
    sel = rk.select(tensors, scores, 5.5)
    assert sel["predicted_bpw"] >= 5.5
    # stopping one tensor earlier would have been under budget
    assert sel["n_chosen"] >= 1
    # the most damaging tensor goes first, and nothing less damaging is chosen before something more damaging
    ranked = [r["name"] for r in sel["ranking"]]
    assert sel["chosen"] == ranked[:sel["n_chosen"]]
    assert sel["chosen"][0] == "blk.7.ffn_down.weight"


def test_select_with_all_q4_0_at_the_base_and_no_budget_promotes_nothing():
    tensors = _toy()
    scores = {x["name"]: 1.0 for x in tensors if x["candidate"]}
    assert rk.select(tensors, scores, 4.0)["n_chosen"] == 0


def test_select_refuses_missing_scores():
    tensors = _toy()
    try:
        rk.select(tensors, {}, 5.5)
    except ValueError as e:
        assert "no score" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_tensor_type_lines_match_exactly_one_name():
    names = ["blk.1.ffn_down.weight", "blk.11.ffn_down.weight"]
    pat = rk.tensor_type_lines(names[:1])[0].rsplit("=", 1)[0]
    assert [n for n in names if re.search(pat, n)] == ["blk.1.ffn_down.weight"]


def test_parse_kld_reads_llama_perplexity_output():
    log = "...\n====== KL divergence statistics ======\nMean    KLD:   0.108329 ±   0.000958\nMaximum KLD:  12.5\n"
    assert rk.parse_kld(log) == (0.108329, 0.000958)


mk = _load("merge_kl")


def _kl(variants, base=20.0, **kw):
    d = {"model": "M", "eval_text_sha256": "abc", "ctx": 512, "chunks": 50, "complete": True,
         "variants": {v: {"kld_mean": 0.1, "ppl_base": base} for v in variants}}
    d.update(kw)
    return d


def test_merge_kl_joins_variants_from_comparable_runs():
    m = mk.merge([_kl(["Q4_0-pure", "Q4_K_M"]), _kl(["IQ4_XS"], base=20.00001)], ["a.json", "b.json"])
    assert list(m["variants"]) == ["Q4_0-pure", "Q4_K_M", "IQ4_XS"] and len(m["merged_from"]) == 2


def test_merge_kl_refuses_mismatched_settings_duplicates_and_base_drift():
    import pytest
    with pytest.raises(ValueError):
        mk.merge([_kl(["A"]), _kl(["B"], ctx=1024)], ["a", "b"])
    with pytest.raises(ValueError):
        mk.merge([_kl(["A"]), _kl(["A"])], ["a", "b"])
    with pytest.raises(ValueError):
        mk.merge([_kl(["A"]), _kl(["B"], base=21.0)], ["a", "b"])


def test_rank_settings_mismatch_flags_window_and_text_changes():
    cur = {"ctx": 512, "chunks": 120, "text": "/x/rank_text_t256.txt", "text_sha256": "aa"}
    recs = {"same": {**cur}, "chunks": {**cur, "chunks": 20}, "text": {**cur, "text_sha256": "bb"},
            "old": {"name": "old", "kld_mean": 0.1, "ctx": 512, "chunks": 120}}       # no stored text hash
    bad = rk.settings_mismatch(recs, cur)
    assert set(bad) == {"chunks", "text", "old"}
    assert bad["chunks"] == {"chunks": (20, 120)} and "text_sha256" in bad["old"]


def test_rank_run_refuses_scores_from_another_window(tmp_path, monkeypatch):
    text = tmp_path / "rank.txt"
    text.write_text("user: hi\nassistant: hello\n")
    out = tmp_path / "rank"
    (out / "scores").mkdir(parents=True)
    (out / "BF16.kld").write_text("x")
    st = {"ctx": 512, "chunks": 20, "text": str(text), "text_sha256": rk._sha256(str(text))}
    (out / "BF16.kld.settings.json").write_text(__import__("json").dumps(st))
    (out / "tensors.json").write_text(__import__("json").dumps({"tensors": [
        {"name": "blk.0.attn_q.weight", "n": 4, "candidate": True, "base_bits": 4.5}]}))
    (out / "scores" / "blk.0.attn_q.weight.json").write_text(__import__("json").dumps(
        {"name": "blk.0.attn_q.weight", "kld_mean": 0.1, **st, "chunks": 10}))
    for k, v in {"RANK_OUT": str(out), "RANK_TEXT": str(text), "TESTBED_OUT": str(tmp_path), "TESTBED_MODELS": str(tmp_path),
                 "LLAMA_DIR": str(tmp_path), "CMPE295_GROUP": str(tmp_path), "RANK_CHUNKS": "20"}.items():
        monkeypatch.setenv(k, v)
    import pytest
    with pytest.raises(SystemExit, match="other settings"):
        rk.main(["run", "--model", "M", "--task", "0", "--ntasks", "1"])
    monkeypatch.setenv("RANK_CHUNKS", "40")                      # base made with 20 chunks
    with pytest.raises(SystemExit, match="base logits were made with other settings"):
        rk.main(["run", "--model", "M", "--task", "0", "--ntasks", "1"])
