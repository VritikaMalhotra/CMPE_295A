"""FR-E17 parity scripts: prompt capture from recorded episodes, and the CUDA-vs-CPU comparison."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _r(i, tokens, content="", ptok="p"):
    return {"id": i, "model_file": "m.gguf", "prompt_tokens_sha256": ptok, "tokens": tokens, "content": content}


def test_compare_counts_identical_outputs_and_divergence(tmp_path):
    rep = _load("parity_report")
    call = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    a = rep._load(_rows(tmp_path / "a.jsonl", [_r(0, [1, 2, 3], call), _r(1, [1, 2, 3], "text"),
                                               _r(2, [5, 6], call), _r(3, [7], "x")]))
    b = rep._load(_rows(tmp_path / "b.jsonl", [_r(0, [1, 2, 3], call), _r(1, [1, 9, 3], "other"),
                                               _r(2, [5, 6, 8], call), _r(3, [7], "x", ptok="q")]))
    c = rep.compare(a, b)
    assert c["n"] == 4 and c["compared_on"] == "token_ids"
    assert c["output_identical"] == 0.5                           # ids 0 and 3
    assert c["first_divergence_min"] == 1 and c["first_divergence_median"] == 1.5   # positions 1 and 2
    assert c["prompt_tokens_identical"] == 0.75
    assert c["tool_call_decision_agree"] == 1.0 and c["decoded_calls_agree"] == 1.0
    assert c["pool_backends"] is False


def test_compare_falls_back_to_text(tmp_path):
    rep = _load("parity_report")
    a = rep._load(_rows(tmp_path / "a.jsonl", [_r(0, None, "same")]))
    b = rep._load(_rows(tmp_path / "b.jsonl", [_r(0, None, "same")]))
    c = rep.compare(a, b)
    assert c["compared_on"] == "text" and c["output_identical"] == 1.0 and c["pool_backends"]


def test_prompts_from_a_recorded_episode():
    pytest.importorskip("bfcl_eval")
    pp = _load("parity_prompts")
    from testbed.adapter import load_bfcl_multi_turn
    from testbed.spec import RunSpec
    g = os.path.join(ROOT, "tests", "golden")
    spec = RunSpec.from_dict(json.load(open(os.path.join(g, "ed3948a2fce8.spec.json"))))
    rows = [json.loads(line) for line in open(os.path.join(g, "ed3948a2fce8.jsonl"))]
    task = load_bfcl_multi_turn(category="multi_turn_base", ids=[spec.task_id])[0]
    got = pp.episode_prompts(spec, rows, task)
    assert len(got) == len(rows)
    assert [(x["turn"], x["step"]) for x in got] == [(r["turn"], r["step"]) for r in rows]
    assert all(x["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for x in got)
    assert got[1]["prompt"].startswith(got[0]["prompt"][:-len("<|im_start|>assistant\n<think>\n\n</think>\n\n")])
    assert rows[0]["raw_text"] in got[1]["prompt"]                # step 0's call is in step 1's prompt
    assert len({x["prompt_sha256"] for x in got}) == len(got)
