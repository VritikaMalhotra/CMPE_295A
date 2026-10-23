"""C2 Episode Runner: synchronous multi-turn loop (model -> tool executor -> model), fixed seeds.

Owner: Ekant Kapgate (draft by Yashashav for review).

Follows bfcl-eval 2026.3.23 ``BaseHandler.inference_multi_turn_prompting`` step for step:
a turn ends when the model's reply decodes to no calls (or fails to decode); a turn is force-quit
after 20 steps; execution results go back as ``tool`` messages; the episode is scored like bfcl's
``multi_turn_runner``: failed (``multi_turn:force_terminated``) only when the number of model turns
differs from the ground truth, otherwise by ``multi_turn_checker`` alone.

Deviations from bfcl, all deliberate and logged:
* a reply with a JSON block that is not a ``{"name", "arguments"}`` object: bfcl stores it in history and
  crashes on the next prompt, scoring the entry ``multi_turn:inference_error``; we score it the same, at once.
* a prompt that outgrows the context: bfcl errors the entry; we score ``multi_turn:context_overflow``.
* a call naming a function outside the task's tool list is not ``eval``-ed (bfcl evals any name); it gets
  the error string bfcl's executor returns for an undefined name. Each episode uses its ``run_id`` as bfcl's ``model_name``, because the
executor keeps tool state in module globals keyed by model name + task id.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from .adapter import Task
from .backends import ContextOverflow
from .classifier import schema_errors
from .qwen_format import to_python_calls
from .spec import EpisodeResult, RunSpec, StepRow


def _executor():
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    return execute_multi_turn_func_call


def score_episode(task: Task, decoded: list[list[list[str]]], model_name: str) -> dict:
    """bfcl's multi_turn_runner after inference: turn-count check, then multi_turn_checker.
    Empty steps are dropped first, as bfcl's eval does (is_empty_execute_response)."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    if len(decoded) != len(task.ground_truth):
        return {"valid": False, "error_type": "multi_turn:force_terminated"}
    cleaned = [[step for step in turn if step and not (len(step) == 1 and not step[0])] for turn in decoded]
    return multi_turn_checker(cleaned, task.ground_truth, task.checker_entry(), "multi_turn_base", model_name)


def _execute_safe(execute, py_calls, calls, task, model_name):
    """Run the calls through bfcl's executor, except names outside the task's tool list (never eval-ed)."""
    if all(c["name"] in task.tool_names for c in calls):
        results, _ = execute(py_calls, task.initial_config, task.involved_classes, model_name, task.id)
        return results
    results = []
    for py, c in zip(py_calls, calls):
        if c["name"] in task.tool_names:
            r, _ = execute([py], task.initial_config, task.involved_classes, model_name, task.id)
            results += r
        else:
            results.append(f"Error during execution: name '{c['name']}' is not defined")
    return results


def run_episode(spec: RunSpec, task: Task, backend, grammar_for: Optional[Callable[[list[dict]], str]],
                log_step: Callable[[StepRow], None]) -> EpisodeResult:
    """``backend`` has ``generate(messages, tools, constraint, seed)``; ``grammar_for(tools)`` builds R1's GBNF."""
    from .qwen_format import split_think
    execute = _executor()
    model_name = f"m{spec.run_id}"
    t0 = time.time()
    # touch the instances once, as bfcl does, so they start from initial_config
    execute([], task.initial_config, task.involved_classes, model_name, task.id)
    grammar = grammar_for(task.tools) if grammar_for else None
    messages: list[dict] = []
    decoded_all: list[list[list[str]]] = []
    n_steps = 0
    forced_quit = False
    verdict = None
    overflow = False
    for turn_idx, turn_msgs in enumerate(task.turns):
        messages.extend(turn_msgs)
        turn_decoded: list[list[str]] = []
        gt_nonempty = len(task.ground_truth[turn_idx]) > 0
        step = 0
        while True:
            try:
                g = backend.generate(messages, task.tools, constraint=grammar, seed=spec.seed)
            except ContextOverflow:
                overflow = True
                verdict = {"valid": False, "error_type": "multi_turn:context_overflow"}
                break
            n_steps += 1
            calls = [{"name": c.name, "arguments": c.arguments} for c in g.tool_calls]
            decode_ok = g.bad_call_blocks == 0
            try:
                py_calls = to_python_calls(calls) if decode_ok else []
            except Exception:
                py_calls, decode_ok = [], False
            turn_decoded.append(py_calls)
            reasoning = split_think(g.raw_text)[0]
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": calls,
                                 "reasoning_content": reasoning})
            else:
                messages.append({"role": "assistant", "content": g.text, "reasoning_content": reasoning})
            ends_turn = (not decode_ok) or len(py_calls) == 0
            results: list[str] = []
            if not ends_turn:
                results = _execute_safe(execute, py_calls, calls, task, model_name)
                for r, c in zip(results, py_calls):
                    messages.append({"role": "tool", "name": c, "content": r})
            step += 1
            quit_now = (not ends_turn) and step > spec.max_steps_per_turn
            errs = []
            for c in calls:
                errs += schema_errors(c, task)
            log_step(StepRow(
                run_id=spec.run_id, config_key=spec.config_key, task_id=task.id, seed=spec.seed,
                turn=turn_idx, step=step - 1, variant=spec.variant, backend=spec.backend, repair=spec.repair,
                constraint=spec.constraint, prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens,
                ttft_ms=g.ttft_ms, decode_ms=g.decode_ms, latency_ms=g.latency_ms, stop_reason=g.stop_reason,
                raw_text=g.raw_text, n_tool_call_blocks=g.tool_call_blocks, decoded_calls=py_calls,
                execution_results=results, known_tools=all(c["name"] in task.tool_names for c in calls),
                schema_errors=errs, turn_has_ground_truth=gt_nonempty, is_last_step_of_turn=ends_turn or quit_now,
                forced_quit=quit_now))
            if g.bad_call_blocks:      # bfcl: the bad block crashes the next prompt build -> inference error
                verdict = {"valid": False, "error_type": "multi_turn:inference_error"}
                break
            if ends_turn:
                break
            if quit_now:
                forced_quit = True
                break
        decoded_all.append(turn_decoded)
        if forced_quit or verdict is not None:
            break
    if verdict is None:
        verdict = score_episode(task, decoded_all, model_name)
    return EpisodeResult(
        run_id=spec.run_id, config_key=spec.config_key, task_id=task.id, seed=spec.seed, variant=spec.variant,
        backend=spec.backend, repair=spec.repair, constraint=spec.constraint, valid=bool(verdict.get("valid")),
        error_type=str(verdict.get("error_type", "")), n_turns=len(decoded_all), n_steps=n_steps,
        forced_quit=forced_quit, wall_s=time.time() - t0, context_overflow=overflow)
