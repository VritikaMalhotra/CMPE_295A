"""Prompt handlers for SmolLM3-3B and LFM2.5 (testbed/prompt_formats.py).

Golden prompts come from each model's own chat_template.jinja (scripts/render_chat_template.py), so these tests
fail if the plain-Python renderings drift from the templates. Qwen3 must stay exactly as before.
"""
import json
import os

import pytest

from testbed import lfm2_format, prompt_formats, qwen_format, smollm3_format
from testbed.backends import parse_completion
from testbed.classifier import classify_step
from testbed.spec import StepRow

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "prompt_formats.json")
FORMATS = {"smollm3": prompt_formats.SMOLLM3, "lfm2.5-1.2b": prompt_formats.LFM25_1P2B,
           "lfm2.5-2.6b": prompt_formats.LFM25_2P6B}


def _golden():
    return json.load(open(GOLDEN))


@pytest.mark.parametrize("case", ["bfcl_two_turns", "tau2_policy", "with_reasoning"])
@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_prompt_matches_chat_template(case, fmt):
    c = _golden()["cases"][case]
    assert FORMATS[fmt].format_prompt(c["messages"], c["tools"], thinking=False) == c["expected"][fmt]


def test_golden_pins_the_smollm3_date():
    g = _golden()
    assert g["today"] == smollm3_format.TODAY
    assert f"Today Date: {smollm3_format.TODAY}\n" in g["cases"]["bfcl_two_turns"]["expected"]["smollm3"]


def test_dispatch_by_model_key():
    assert prompt_formats.for_model("qwen3-0.6b") is prompt_formats.QWEN3
    assert prompt_formats.for_model("qwen3-4b").format_prompt is qwen_format.format_prompt
    assert prompt_formats.for_model("smollm3-3b") is prompt_formats.SMOLLM3
    assert prompt_formats.for_model("lfm2.5-1.2b-instruct") is prompt_formats.LFM25_1P2B
    assert prompt_formats.for_model("lfm2.5-2.6b") is prompt_formats.LFM25_2P6B
    with pytest.raises(ValueError):
        prompt_formats.for_model("llama-3.2-1b")


def test_qwen_default_unchanged():
    """Backend and parse_completion default to the Qwen3 format, so existing callers get the old behaviour."""
    raw = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    j = {"content": raw, "tokens_predicted": 9, "stop_type": "eos"}
    a = parse_completion(j, n_prompt=10, budget=100, constraint=None, latency_ms=1.0)
    b = parse_completion(j, n_prompt=10, budget=100, constraint=None, latency_ms=1.0, fmt=prompt_formats.QWEN3)
    assert a == b and [c.name for c in a.tool_calls] == ["ls"] and a.tool_call_blocks == 1


# --- LFM2.5 Pythonic calls ---------------------------------------------------------------------------------

def test_lfm_parse_calls():
    raw = ("<|tool_call_start|>[mv(source='a b.pdf', destination=\"temp\"), ls(a=True), "
           "pay(amount=-2.5, ids=[1, 2], meta={\"refund\": false, \"note\": null}, x=None)]<|tool_call_end|>")
    calls, n = lfm2_format.extract_tool_calls(raw)
    assert calls == [{"name": "mv", "arguments": {"source": "a b.pdf", "destination": "temp"}},
                     {"name": "ls", "arguments": {"a": True}},
                     {"name": "pay", "arguments": {"amount": -2.5, "ids": [1, 2],
                                                   "meta": {"refund": False, "note": None}, "x": None}}]
    assert n == 3


@pytest.mark.parametrize("calls", [
    [{"name": "ls", "arguments": {}}],
    [{"name": "cd", "arguments": {"folder": "it's \"q\"\nx\\y"}}, {"name": "ls", "arguments": {"a": False}}],
    [{"name": "book", "arguments": {"n": 3, "price": 12.5, "who": ["a", "é"], "opt": {"k": [1, None, True]}}}],
])
def test_lfm_round_trip(calls):
    """What the template writes for a call parses back to the same call."""
    got, n = lfm2_format.extract_tool_calls("Sure. " + lfm2_format.render_tool_calls(calls))
    assert got == calls and n == len(calls)


def test_lfm_surrogate_escapes_read_as_text():
    """Python string syntax keeps the escapes ``\\ud83d\\ude80`` as two lone surrogates, which UTF-8 cannot encode
    (the next step's prompt hash then fails: LFM2.5 reference, 2026-11-11). A pair reads as the character it
    encodes, as JSON does for Qwen3's calls; an unpaired surrogate is not text, so the block does not parse."""
    calls, n = lfm2_format.extract_tool_calls(
        "<|tool_call_start|>[post(text='done \\ud83d\\ude80', tags=['\\ud83d\\ude00'])]<|tool_call_end|>")
    assert calls == [{"name": "post", "arguments": {"text": "done \U0001F680", "tags": ["\U0001F600"]}}] and n == 1
    json.dumps(calls, ensure_ascii=False).encode("utf-8")              # encodable, so the prompt can be hashed
    calls, n = lfm2_format.extract_tool_calls("<|tool_call_start|>[post(text='x \\ud83d')]<|tool_call_end|>")
    assert (calls, n) == ([], 1)


@pytest.mark.parametrize("raw,n_calls,n_blocks", [
    ("<|tool_call_start|>[ls('docs')]<|tool_call_end|>", 0, 1),                     # positional argument
    ("<|tool_call_start|>[ls(a=True]<|tool_call_end|>", 0, 1),                       # syntax error
    ("<|tool_call_start|>[ls()]", 0, 1),                                             # never closed
    ("<|tool_call_start|>[ls()]<|tool_call_end|><|tool_call_start|>[x(]<|tool_call_end|>", 1, 2),
    ("<|tool_call_start|>[os.system(cmd=open('f').read())]<|tool_call_end|>", 0, 1),  # not a literal
    ("I will list the files.", 0, 0),
])
def test_lfm_bad_blocks_counted(raw, n_calls, n_blocks):
    calls, n = lfm2_format.extract_tool_calls(raw)
    assert (len(calls), n) == (n_calls, n_blocks)


def test_lfm_2p6b_answer_split():
    """The 2.6B prompt ends inside <think>, so the raw answer has only the closing tag."""
    reasoning, answer = lfm2_format.split_think("Need the list.</think><|tool_call_start|>[ls()]<|tool_call_end|>")
    assert reasoning == "Need the list." and answer.startswith("<|tool_call_start|>")
    assert prompt_formats.LFM25_2P6B.format_prompt([{"role": "user", "content": "hi"}], []).endswith(
        "<|im_start|>assistant\n<think>")
    assert prompt_formats.LFM25_1P2B.format_prompt([{"role": "user", "content": "hi"}], []).endswith(
        "<|im_start|>assistant\n")


def test_lfm_parse_completion():
    j = {"content": "<|tool_call_start|>[cd(folder='temp')]<|tool_call_end|>", "tokens_predicted": 12}
    g = parse_completion(j, n_prompt=5, budget=50, constraint=None, latency_ms=1.0, fmt=prompt_formats.LFM25_1P2B)
    assert [(c.name, c.arguments) for c in g.tool_calls] == [("cd", {"folder": "temp"})]
    assert g.tool_call_blocks == 1 and g.bad_call_blocks == 0


# --- SmolLM3 ----------------------------------------------------------------------------------------------

def test_smollm3_parse_and_history():
    raw = '<tool_call>{"name": "ls", "arguments": {"a": true}}</tool_call>\n<tool_call>\n{"name": "mv"\n</tool_call>'
    calls, n = smollm3_format.extract_tool_calls(raw)
    assert calls == [{"name": "ls", "arguments": {"a": True}}] and n == 2
    hist = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "ls", "arguments": {"a": True}}]},
            {"role": "tool", "content": "[]"}]
    p = smollm3_format.format_prompt(hist, [{"name": "ls"}])
    assert ('<|im_start|>assistant\n<think>\n\n</think>\n<tool_call>\n{"name": "ls", "arguments": {"a": true}}\n'
            '</tool_call><|im_end|>\n<|im_start|>user\n[]<|im_end|>\n') in p
    assert "{'name': 'ls'}" in p                       # tools printed with Jinja's string filter (Python repr)
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n")


# --- classifier ---------------------------------------------------------------------------------------------

def _row(raw, decoded, n_blocks):
    return StepRow(run_id="r", config_key="c", task_id="t", seed=1, turn=0, step=0, variant="bf16", backend="cuda",
                   repair="none", constraint="none", prompt_tokens=1, completion_tokens=1, ttft_ms=None,
                   decode_ms=None, latency_ms=1.0, stop_reason="eos", raw_text=raw, n_tool_call_blocks=n_blocks,
                   decoded_calls=decoded, execution_results=[], known_tools=True, schema_errors=[],
                   turn_has_ground_truth=True, is_last_step_of_turn=True, forced_quit=False, benchmark="bfcl")


def test_classifier_lfm_marker():
    assert classify_step(_row("<|tool_call_start|>[ls(]", [], 1)) == "malformed"
    assert classify_step(_row("<|tool_call_start|>", [], 0)) == "malformed"     # marker, nothing parsed
    assert classify_step(_row("<tool_call>", [], 0)) == "malformed"             # Qwen rule unchanged


# --- transport: BOS and the 2.6B think prefix ------------------------------------------------------------

def test_transport_settings():
    """New formats send token ids (exact template tokens, no server-side BOS rule); Qwen3 still sends text."""
    assert not prompt_formats.QWEN3.send_tokens and prompt_formats.QWEN3.raw_prefix == ""
    assert all(f.send_tokens for f in FORMATS.values())
    for f in ("lfm2.5-1.2b", "lfm2.5-2.6b"):
        assert FORMATS[f].format_prompt([{"role": "user", "content": "hi"}], []).startswith(lfm2_format.BOS)
    assert prompt_formats.SMOLLM3.format_prompt([{"role": "user", "content": "hi"}], []).startswith(
        "<|im_start|>system")                                  # SmolLM3's template has no BOS


def test_lfm_2p6b_raw_prefix_keeps_reasoning_out_of_the_answer():
    j = {"content": "Which folder? Probably temp.</think>Done, moved.", "tokens_predicted": 12, "stop_type": "eos"}
    g = parse_completion(j, n_prompt=5, budget=50, constraint=None, latency_ms=1.0, fmt=prompt_formats.LFM25_2P6B)
    assert g.raw_text.startswith("<think>Which folder?") and g.text == "Done, moved."
    row = _row(g.raw_text, [], 0)
    # the reasoning's "?" must not count: the visible answer is a direct claim with no call
    assert classify_step(row) == "missing_call_direct_answer"


def test_lfm_requests_keep_the_call_markers():
    """llama-server drops control tokens from content unless the request lists them as preserved tokens."""
    assert prompt_formats.QWEN3.preserve == () and prompt_formats.SMOLLM3.preserve == ()
    for f in ("lfm2.5-1.2b", "lfm2.5-2.6b"):
        assert {"<|tool_call_start|>", "<|tool_call_end|>"} <= set(FORMATS[f].preserve)
