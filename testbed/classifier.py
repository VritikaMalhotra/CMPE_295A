"""C6 Failure Classifier, rules v0: one class per logged step.

Extends the Oct 8 pilot's ``failure_class()`` (spikes/s4_bfcl_single_turn.py) from single-turn
outputs to multi-turn steps. Order matters: the first rule that fires wins. The written codebook is
docs/codebook_v0.md; keep the two in step.

    truncated      generation hit n_predict (output cut off)
    malformed      a tool-call block that does not parse: <tool_call> not one {"name", "arguments"} JSON
                   object (Qwen3, SmolLM3), or <|tool_call_start|> not keyword-only calls (LFM2.5)
    forced_quit    the 21st step of a turn (bfcl's MAXIMUM_STEP_LIMIT exceeded); checked before the
                   call-content rules, since that step is usually a call inside an error loop
    unknown_tool   a call names a tool that is not in the task's tool list (hallucinated tool)
    invalid_args   a call violates the tool's schema (missing required, unknown argument, wrong type)
    exec_error     schema-valid call(s), but the executor returned "Error during execution"
    ok_call        schema-valid call(s) that executed
    missing_call   first step of a turn that has ground truth, answered in text with no call
    abstain_ok     no call on a turn whose ground truth is empty (correct refusal)
    end_turn       no call after one or more calls: the model closes the turn
    reply          tau2 only: a text message to the simulated user (tau2 has no per-turn ground truth,
                   so a text reply is neither a missing call nor a refusal; episode reward judges it)

Classes added by decisions 3a/3b (Ekant 2026-11-08; advisor to confirm), on by default; ``classify_step(row,
proposed=False)`` gives the earlier labels (frozen logs can be relabelled without rerunning):

    tool_error_payload          checked after exec_error: schema-valid call(s) that executed, but a result is a
                                JSON object with a non-empty "error" key (the tool reported an error without
                                raising, e.g. GorillaFileSystem's {"error": "cd: x: No such directory"})
    missing_call_direct_answer  refines missing_call: the text answers the request itself, i.e. it neither asks
                                the user something (no "?") nor announces a call it then does not make
                                (Oct 8 pilot: 49 of 58 no-call outputs of Q4_0 answered directly)

Episode-level correctness (state + response checks) stays with bfcl's multi_turn_checker; these
classes describe *how* a step behaved, for the damage map.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import Task
    from .spec import StepRow

# tool_error_payload and missing_call_direct_answer: decisions 3a/3b (Ekant 2026-11-08; advisor to confirm)
PROPOSED_CLASSES = ["tool_error_payload", "missing_call_direct_answer"]
CLASSES = ["truncated", "malformed", "forced_quit", "unknown_tool", "invalid_args", "exec_error",
           "tool_error_payload", "ok_call", "missing_call", "missing_call_direct_answer", "abstain_ok",
           "end_turn", "reply"]
ALL_CLASSES = CLASSES
FAILURES = {"truncated", "malformed", "unknown_tool", "invalid_args", "exec_error", "forced_quit", "missing_call",
            "tool_error_payload", "missing_call_direct_answer"}

_CALL_MARKERS = ("<tool_call>", "<|tool_call_start|>")
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_ANNOUNCE = re.compile(r"\b(i'll|i will|let me|i am going to|i'm going to|i need to|i should)\b[^.?!\n]*"
                       r"\b(call|use|invoke|run|execute|check|look up|fetch|search)\b", re.I)

_PY_TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
             "array": (list, tuple), "object": (dict,)}


def _type_ok(value, schema: dict) -> bool:
    t = schema.get("type")
    if t is None:
        return True
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _PY_TYPES.get(t, (object,)))


def schema_errors(call: dict, task: "Task") -> list[str]:
    """Problems with one call against the task's tool schema (empty list = valid)."""
    name = call.get("name")
    schema = task.tool_schema(name) if name else None
    if schema is None:
        return []                       # unknown tool is its own class
    args = call.get("arguments")
    if not isinstance(args, dict):
        return [f"{name}: arguments is not an object"]
    errs = []
    props = schema.get("properties", {})
    for r in schema.get("required", []):
        if r not in args:
            errs.append(f"{name}: missing required '{r}'")
    for k, v in args.items():
        if k not in props:
            errs.append(f"{name}: unknown argument '{k}'")
        elif not _type_ok(v, props[k]):
            errs.append(f"{name}: '{k}' should be {props[k].get('type')}")
        elif "enum" in props[k] and v not in props[k]["enum"]:
            errs.append(f"{name}: '{k}' not in enum")
    return errs


def error_payload(result) -> bool:
    """A tool result that is a JSON object with a non-empty "error" key (an error returned, not raised)."""
    try:
        obj = json.loads(str(result))
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and bool(obj.get("error"))


def answers_directly(raw_text: str) -> bool:
    """Visible text (thinking removed) that neither asks the user anything nor announces a call."""
    text = _THINK.sub("", raw_text or "").strip()
    return bool(text) and "?" not in text and not _ANNOUNCE.search(text)


def classify_step(row: "StepRow", proposed: bool = True) -> str:
    """One class per step (codebook v0). ``proposed=False`` gives the labels from before decisions 3a/3b
    (without tool_error_payload and missing_call_direct_answer), e.g. to compare with older logs."""
    n_parsed = len(row.decoded_calls)
    if row.stop_reason == "limit":
        return "truncated"
    # call markers of every prompt format (prompt_formats.ALL_CALL_MARKERS): "<tool_call>" (Qwen3, SmolLM3) and
    # "<|tool_call_start|>" (LFM2.5); a Qwen3 step never contains the LFM marker, so Qwen3 labels are unchanged
    if row.n_tool_call_blocks > n_parsed or (n_parsed == 0 and any(k in row.raw_text for k in _CALL_MARKERS)):
        return "malformed"
    if row.forced_quit:
        return "forced_quit"
    if n_parsed and not row.known_tools:
        return "unknown_tool"
    if row.schema_errors:
        return "invalid_args"
    if any(str(r).startswith("Error during execution") for r in row.execution_results):
        return "exec_error"
    if proposed and n_parsed and any(error_payload(r) for r in row.execution_results):
        return "tool_error_payload"
    if n_parsed:
        return "ok_call"
    if str(getattr(row, "benchmark", "") or "").startswith("tau2"):
        return "reply"
    if not row.turn_has_ground_truth:
        return "abstain_ok"
    if row.step == 0:
        return "missing_call_direct_answer" if proposed and answers_directly(row.raw_text) else "missing_call"
    return "end_turn"
