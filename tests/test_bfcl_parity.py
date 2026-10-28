"""Parity with bfcl-eval 2026.3.23: prompt bytes, scoring rules, executor safety, malformed blocks."""
import ast
import json
import os
import textwrap

import pytest

bfcl_eval = pytest.importorskip("bfcl_eval")

from testbed.adapter import load_bfcl_multi_turn
from testbed.backends import ContextOverflow
from testbed.qwen_format import NON_THINKING_SUFFIX, format_prompt
from testbed.runner import run_episode
from testbed.spec import GenerateResult, RunSpec, ToolCall


def _bfcl_format_prompt():
    """bfcl's QwenFCHandler._format_prompt, lifted from source (importing the handler needs parser deps)."""
    path = os.path.join(os.path.dirname(bfcl_eval.__file__), "model_handler", "local_inference", "qwen_fc.py")
    tree = ast.parse(open(path).read())
    fn = next(n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
              if isinstance(n, ast.FunctionDef) and n.name == "_format_prompt")
    fn.decorator_list = []
    ns = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return lambda messages, tools: ns["_format_prompt"](None, messages, tools)


TOOLS = [{"name": "ls", "description": "List.", "parameters": {"type": "dict", "properties": {}, "required": []}},
         {"name": "cd", "description": "Change dir.", "parameters": {"type": "dict", "properties": {
             "folder": {"type": "string", "description": "x"}}, "required": ["folder"]}}]
CD = {"name": "cd", "arguments": {"folder": "docs"}}
LS = {"name": "ls", "arguments": {}}
HISTORIES = {
    "one_call": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [CD]},
                 {"role": "tool", "name": "cd(folder='docs')", "content": "{}"}],
    "two_calls_two_results": [{"role": "user", "content": "go"},
                              {"role": "assistant", "content": "", "tool_calls": [CD, LS]},
                              {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"},
                              {"role": "assistant", "content": "done"}, {"role": "user", "content": "again"}],
    "repeated_identical_call": [{"role": "user", "content": "go"},
                                {"role": "assistant", "content": "", "tool_calls": [CD, CD]},
                                {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}],
    "reasoning_after_last_query": [{"role": "user", "content": "go"},
                                   {"role": "assistant", "content": "", "tool_calls": [LS],
                                    "reasoning_content": "think"}, {"role": "tool", "content": "x"}],
}


@pytest.mark.parametrize("name", sorted(HISTORIES))
def test_prompt_is_byte_identical_to_bfcl(name):
    bfcl = _bfcl_format_prompt()
    ours = format_prompt(HISTORIES[name], TOOLS, thinking=False)
    assert ours == bfcl(HISTORIES[name], TOOLS) + NON_THINKING_SUFFIX


# ---------------------------------------------------------------- runner semantics with a scripted model
class Scripted:
    """Replays replies; each reply is raw completion text (or an exception to raise)."""

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, tools, constraint=None, seed=0):
        from testbed.qwen_format import extract_tool_calls, split_think
        r = self.replies.pop(0) if self.replies else "Done."
        if isinstance(r, Exception):
            raise r
        _, cleaned = split_think(r)
        calls, n = extract_tool_calls(cleaned)
        good = [c for c in calls if isinstance(c, dict) and "name" in c and "arguments" in c]
        return GenerateResult(text=cleaned, raw_text=r, tool_calls=[ToolCall(c["name"], c["arguments"]) for c in good],
                              tool_call_blocks=n, prompt_tokens=1, completion_tokens=1, ttft_ms=1.0, decode_ms=1.0,
                              latency_ms=1.0, stop_reason="eos", grammar_used=False,
                              bad_call_blocks=len(calls) - len(good))


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def gt_replies(task):
    """A perfect model: each turn's ground-truth calls in one reply, then a closing text reply."""
    out = []
    for turn in task.ground_truth:
        if turn:
            blocks = []
            for py in turn:
                node = ast.parse(py).body[0].value
                params = list(next(t for t in task.tools if t["name"] == node.func.id)["parameters"]["properties"])
                args = {params[i]: ast.literal_eval(a) for i, a in enumerate(node.args)}   # positional, e.g. sort('f')
                args.update({k.arg: ast.literal_eval(k.value) for k in node.keywords})
                blocks.append(call(node.func.id, **args))
            out.append("\n".join(blocks))
        out.append("Done.")
    return out


@pytest.fixture(scope="module")
def task():
    return load_bfcl_multi_turn("multi_turn_base", ids=["multi_turn_base_0"])[0]


def spec(seed=1, **kw):
    return RunSpec(experiment="t", model="m", variant="q4_0", backend="cpu", repair="none", constraint="none",
                   benchmark="bfcl_multi_turn_base", task_id="multi_turn_base_0", seed=seed, **kw)


def test_ground_truth_replay_is_valid(task):
    ep = run_episode(spec(seed=11), task, Scripted(gt_replies(task)), None, lambda r: None)
    assert ep.valid, ep.error_type


def test_forced_quit_in_last_turn_is_still_scored(task):
    """bfcl fails force-terminated entries only when the turn count is short; in the last turn it scores."""
    replies = gt_replies(task)[:-1] + [call("ls")] * 30      # last turn done right, then loops on ls()
    ep = run_episode(spec(seed=12, max_steps_per_turn=3), task, Scripted(replies), None, lambda r: None)
    assert ep.forced_quit and ep.n_turns == len(task.ground_truth)
    assert ep.error_type != "multi_turn:force_terminated"


def test_forced_quit_before_last_turn_fails(task):
    replies = [call("ls")] * 30
    ep = run_episode(spec(seed=13, max_steps_per_turn=3), task, Scripted(replies), None, lambda r: None)
    assert ep.forced_quit and not ep.valid and ep.error_type == "multi_turn:force_terminated"


def test_malformed_block_is_bfcl_inference_error(task):
    bad = '<tool_call>\n[{"name": "ls", "arguments": {}}]\n</tool_call>'
    ep = run_episode(spec(seed=14), task, Scripted([bad]), None, lambda r: None)
    assert not ep.valid and ep.error_type == "multi_turn:inference_error"


def test_context_overflow_is_a_scored_failure(task):
    ep = run_episode(spec(seed=15), task, Scripted([ContextOverflow("full")]), None, lambda r: None)
    assert not ep.valid and ep.context_overflow and ep.error_type == "multi_turn:context_overflow"


def test_unknown_tool_name_is_never_evaluated(task, tmp_path):
    marker = tmp_path / "pwned"
    evil = "<tool_call>\n" + json.dumps({"name": f"__import__('pathlib').Path('{marker}').touch", "arguments": {}}) \
           + "\n</tool_call>"
    rows = []
    run_episode(spec(seed=16), task, Scripted([evil]), None, rows.append)
    assert not marker.exists()
    assert rows[0].execution_results[0].startswith("Error during execution: name")
    assert rows[0].failure_class in ("", "unknown_tool")
