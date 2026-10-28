"""C6 rules v0: one class per step, first matching rule wins."""
from testbed.classifier import classify_step
from testbed.spec import StepRow


def row(**kw):
    base = dict(run_id="r", config_key="c", task_id="t", seed=1, turn=0, step=0, variant="q4_0", backend="cuda",
                repair="none", constraint="none", prompt_tokens=10, completion_tokens=5, ttft_ms=1.0, decode_ms=1.0,
                latency_ms=2.0, stop_reason="eos", raw_text="", n_tool_call_blocks=0, decoded_calls=[],
                execution_results=[], known_tools=True, schema_errors=[], turn_has_ground_truth=True,
                is_last_step_of_turn=False, forced_quit=False)
    base.update(kw)
    return StepRow(**base)


def test_classes():
    assert classify_step(row(stop_reason="limit", decoded_calls=["ls()"])) == "truncated"
    assert classify_step(row(raw_text="<tool_call>\n{bad\n</tool_call>", n_tool_call_blocks=1)) == "malformed"
    assert classify_step(row(decoded_calls=["fly()"], n_tool_call_blocks=1, known_tools=False)) == "unknown_tool"
    assert classify_step(row(decoded_calls=["ls(a=1)"], n_tool_call_blocks=1, schema_errors=["ls: unknown a"])) == "invalid_args"
    assert classify_step(row(decoded_calls=["cd(folder='x')"], n_tool_call_blocks=1,
                             execution_results=["Error during execution: no such folder"])) == "exec_error"
    assert classify_step(row(decoded_calls=["ls()"], n_tool_call_blocks=1, forced_quit=True)) == "forced_quit"
    assert classify_step(row(decoded_calls=["ls()"], n_tool_call_blocks=1, execution_results=["{}"])) == "ok_call"
    assert classify_step(row(raw_text="Sure, done.")) == "missing_call"
    assert classify_step(row(raw_text="I can't.", turn_has_ground_truth=False)) == "abstain_ok"
    assert classify_step(row(raw_text="Done.", step=2)) == "end_turn"


def test_forced_quit_wins_over_call_errors():
    assert classify_step(row(decoded_calls=["cd(folder='x')"], n_tool_call_blocks=1, forced_quit=True,
                             execution_results=["Error during execution: no such folder"])) == "forced_quit"


def test_tau2_text_is_a_reply_not_a_missing_call():
    t = dict(benchmark="tau2_verified_airline", turn_has_ground_truth=False)
    assert classify_step(row(raw_text="Could you give me your user id?", **t)) == "reply"
    assert classify_step(row(raw_text="<tool_call>\n{bad\n</tool_call>", n_tool_call_blocks=1, **t)) == "malformed"
    assert classify_step(row(decoded_calls=['{"name": "get_user_details"}'], n_tool_call_blocks=1,
                             execution_results=["Error during execution: User not found"], **t)) == "exec_error"
    # rows written before tau2 joined (benchmark "") keep their BFCL classes
    assert classify_step(row(raw_text="Sure, done.")) == "missing_call"
