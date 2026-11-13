"""Parity with bfcl-eval 2026.3.23: prompt bytes, scoring rules, executor safety, malformed blocks,
for all three multi-turn categories (base, miss_func, miss_param)."""
import ast
import contextlib
import hashlib
import json
import os
import sys
import textwrap
import types

import pytest

bfcl_eval = pytest.importorskip("bfcl_eval")

from testbed.adapter import BFCL_CATEGORIES, bfcl_sample, load_bfcl_multi_turn, sample_task_ids
from testbed.backends import ContextOverflow
from testbed.qwen_format import NON_THINKING_SUFFIX, format_prompt
from testbed.runner import run_episode, score_episode
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


def call(name, /, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def gt_replies(task):
    """A perfect model: each turn's ground-truth calls in one reply, then a closing text reply."""
    out = []
    for turn in task.ground_truth:
        if turn:
            blocks = []
            for py in turn:
                node = ast.parse(py).body[0].value
                params = list(next(t for t in task.all_tools if t["name"] == node.func.id)["parameters"]["properties"])
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


# ---------------------------------------------------------------- all three categories
def _bfcl_entries(category):
    """bfcl's own ``load_dataset_entry`` (tools populated, held-out functions popped, hint added). bfcl-eval is
    installed with --no-deps, so its only missing import, filelock, is stubbed (it only guards file reads)."""
    if "filelock" not in sys.modules:
        try:
            import filelock  # noqa: F401
        except ImportError:
            stub = types.ModuleType("filelock")
            stub.FileLock = lambda path: contextlib.nullcontext()
            sys.modules["filelock"] = stub
    from bfcl_eval.utils import load_dataset_entry
    return {e["id"]: e for e in load_dataset_entry(category)}


def _bfcl_messages(entry, upto_turn=None):
    """The message list bfcl's inference_multi_turn_prompting builds, with a fixed assistant reply after each
    turn: the holdout turn's empty message list becomes bfcl's additional-function user message."""
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING
    holdout = entry.get("missed_function", {})
    msgs = []
    for i, turn in enumerate(entry["question"]):
        if str(i) in holdout:
            assert len(turn) == 0
            turn = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                functions=holdout[str(i)])}]
        msgs += turn
        if upto_turn is not None and i == upto_turn:
            break
        msgs.append({"role": "assistant", "content": "ok"})
    return msgs


def _our_messages(task, upto_turn=None):
    msgs = []
    for i, turn in enumerate(task.turns):
        msgs += turn
        if upto_turn is not None and i == upto_turn:
            break
        msgs.append({"role": "assistant", "content": "ok"})
    return msgs


@pytest.mark.parametrize("category", BFCL_CATEGORIES)
def test_tools_and_holdouts_match_bfcl_loader(category):
    bfcl = _bfcl_entries(category)
    ours = load_bfcl_multi_turn(category)
    assert len(ours) == len(bfcl) == 200
    for t in ours:
        e = bfcl[t.id]
        assert t.category == category and t.benchmark == f"bfcl_{category}"
        assert t.tools == e["function"], t.id
        assert t.missed_function == e.get("missed_function", {}), t.id


@pytest.mark.parametrize("category", BFCL_CATEGORIES)
def test_prompt_is_byte_identical_to_bfcl_for_every_task(category):
    """Every task of the category, prompt at the end of the last turn (and at the holdout turn for miss_func)."""
    fmt = _bfcl_format_prompt()
    bfcl = _bfcl_entries(category)
    for t in load_bfcl_multi_turn(category):
        e = bfcl[t.id]
        points = [len(t.turns) - 1] + [int(k) for k in t.missed_function]
        for upto in points:
            ours = format_prompt(_our_messages(t, upto), t.tools, thinking=False)
            assert ours == fmt(_bfcl_messages(e, upto), e["function"]) + NON_THINKING_SUFFIX, (t.id, upto)


def test_miss_func_holdout_is_announced_at_its_turn():
    t = load_bfcl_multi_turn("multi_turn_miss_func", ids=["multi_turn_miss_func_0"])[0]
    assert {k: [d["name"] for d in v] for k, v in t.missed_function.items()} == {"3": ["sort"]}
    assert "sort" not in {d["name"] for d in t.tools} and "sort" in t.tool_names
    assert t.tool_schema("sort") is not None
    (msg,) = t.turns[3]
    assert msg["role"] == "user" and msg["content"].startswith("[{'name': 'sort'")
    assert msg["content"].endswith("\nI have updated some more functions you can choose from. What about now?")
    assert "Python 3 syntax" not in msg["content"]           # bfcl adds the hint after popping held-out docs
    assert t.raw["question"][3] == []                        # the source entry is not modified


@pytest.mark.parametrize("category", BFCL_CATEGORIES)
def test_ground_truth_replay_is_valid_on_the_whole_sample(category):
    """All 100 sampled tasks of the category: replaying the ground truth scores valid."""
    bad = []
    for t in load_bfcl_multi_turn(category, ids=sample_task_ids(category)):
        s = RunSpec(experiment="t", model="m", variant="q4_0", backend="cpu", repair="none", constraint="none",
                    benchmark=f"bfcl_{category}", task_id=t.id, seed=21)
        ep = run_episode(s, t, Scripted(gt_replies(t)), None, lambda r: None)
        if not ep.valid:
            bad.append((t.id, ep.error_type))
    assert not bad


def test_score_episode_passes_the_task_category(monkeypatch):
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as mtc
    seen = []
    monkeypatch.setattr(mtc, "multi_turn_checker", lambda *a: seen.append(a[3]) or {"valid": True})
    for c in BFCL_CATEGORIES:
        t = load_bfcl_multi_turn(c, ids=[f"{c}_0"])[0]
        score_episode(t, [[] for _ in t.ground_truth], "m")
    assert seen == list(BFCL_CATEGORIES)


# Pinned sample: changing the seed, the sampler or the data changes these hashes (and the pre-registration).
SAMPLE_SHA256 = {
    "multi_turn_base": "35faa1327bf35e08245f2edceac6832c419d031dfcc354d0c9b44b6761921cbc",
    "multi_turn_miss_func": "a4fd74fc115aff5144026e4c45411c2676ed05537f7549d678b622c24f2957ba",
    "multi_turn_miss_param": "e49434b4708eae983e05ac4e5eef1c256d4c8de67cd3b0e87abf6a325bdd018b",
}


def test_fixed_seed_sample_is_100_per_category_and_pinned():
    pool = bfcl_sample()
    assert list(pool) == [f"bfcl_{c}" for c in BFCL_CATEGORIES]
    for c in BFCL_CATEGORIES:
        ids = pool[f"bfcl_{c}"]
        assert len(ids) == len(set(ids)) == 100 and all(i.startswith(c + "_") for i in ids)
        assert ids == sorted(ids, key=lambda i: int(i.rsplit("_", 1)[1]))      # file order
        assert ids == sample_task_ids(c)                                       # deterministic
        assert hashlib.sha256(",".join(ids).encode()).hexdigest() == SAMPLE_SHA256[c], c
    idx = [{i.rsplit("_", 1)[1] for i in pool[f"bfcl_{c}"]} for c in BFCL_CATEGORIES]
    assert not (idx[0] == idx[1] == idx[2])                                    # independent draws
    assert sample_task_ids("multi_turn_base", seed=1) != pool["bfcl_multi_turn_base"]
