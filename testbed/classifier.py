"""C6 Failure Classifier, rules v0: one class per logged step.

Owner: Vritika Malhotra (draft by Yashashav for review).

Extends the Oct 8 pilot's ``failure_class()`` (spikes/s4_bfcl_single_turn.py) from single-turn
outputs to multi-turn steps, and adds the proposed "missing call with a direct answer" class
(Workbook v3, decisions for the advisor). Order matters: the first rule that fires wins.

    truncated      generation hit n_predict (output cut off)
    malformed      a <tool_call> block that is not one valid {"name", "arguments"} JSON object
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

Episode-level correctness (state + response checks) stays with bfcl's multi_turn_checker; these
classes describe *how* a step behaved, for the damage map.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import Task
    from .spec import StepRow

CLASSES = ["truncated", "malformed", "forced_quit", "unknown_tool", "invalid_args", "exec_error",
           "ok_call", "missing_call", "abstain_ok", "end_turn", "reply"]
FAILURES = {"truncated", "malformed", "unknown_tool", "invalid_args", "exec_error", "forced_quit", "missing_call"}

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


def classify_step(row: "StepRow") -> str:
    n_parsed = len(row.decoded_calls)
    if row.stop_reason == "limit":
        return "truncated"
    if row.n_tool_call_blocks > n_parsed or ("<tool_call>" in row.raw_text and n_parsed == 0):
        return "malformed"
    if row.forced_quit:
        return "forced_quit"
    if n_parsed and not row.known_tools:
        return "unknown_tool"
    if row.schema_errors:
        return "invalid_args"
    if any(str(r).startswith("Error during execution") for r in row.execution_results):
        return "exec_error"
    if n_parsed:
        return "ok_call"
    if str(getattr(row, "benchmark", "") or "").startswith("tau2"):
        return "reply"
    if not row.turn_has_ground_truth:
        return "abstain_ok"
    if row.step == 0:
        return "missing_call"
    return "end_turn"
