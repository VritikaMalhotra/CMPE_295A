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
    assert classify_step(row(raw_text="Sure, done.")) == "missing_call_direct_answer"   # a direct answer (decision 3a)
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
    assert classify_step(row(raw_text="Sure, done.")) == "missing_call_direct_answer"   # a direct answer (decision 3a)


# ---------------------------------------------------- classes added by decisions 3a/3b (on by default)
def test_decided_classes_on_by_default_and_old_labels_available():
    payload = row(decoded_calls=["cd(folder='x')"], n_tool_call_blocks=1,
                  execution_results=['{"error": "cd: x: No such directory"}'])
    assert classify_step(payload) == "tool_error_payload"
    assert classify_step(row(raw_text="The capital of France is Paris.")) == "missing_call_direct_answer"
    assert classify_step(payload, proposed=False) == "ok_call"                       # labels before the decision
    assert classify_step(row(raw_text="The capital of France is Paris."), proposed=False) == "missing_call"


def test_tool_error_payload():
    call = dict(decoded_calls=["cd(folder='x')"], n_tool_call_blocks=1)
    assert classify_step(row(execution_results=['{"error": "cd: x: No such directory"}'], **call),
                         proposed=True) == "tool_error_payload"
    # one of several results is enough
    assert classify_step(row(execution_results=['{"current_working_directory": "x"}', '{"error": "mkdir: exists"}'],
                             decoded_calls=["cd(folder='x')", "mkdir(dir_name='y')"], n_tool_call_blocks=2),
                         proposed=True) == "tool_error_payload"
    # not payloads: empty error, other keys, plain strings that mention errors, non-JSON
    for res in ['{"error": ""}', '{"result": "error"}', '"error"', "No error.", "[1, 2]", "{}"]:
        assert classify_step(row(execution_results=[res], **call), proposed=True) == "ok_call", res
    # a raised error stays exec_error, and earlier rules still win
    assert classify_step(row(execution_results=["Error during execution: boom"], **call), proposed=True) == "exec_error"
    assert classify_step(row(execution_results=['{"error": "x"}'], forced_quit=True, **call),
                         proposed=True) == "forced_quit"


def test_missing_call_with_a_direct_answer():
    p = dict(proposed=True)
    assert classify_step(row(raw_text="The capital of France is Paris."), **p) == "missing_call_direct_answer"
    assert classify_step(row(raw_text="<think>\nI know this.\n</think>\n\nIt is 42."), **p) == \
        "missing_call_direct_answer"
    # asks the user, or announces a call it does not make: plain missing_call
    assert classify_step(row(raw_text="Which folder do you mean?"), **p) == "missing_call"
    assert classify_step(row(raw_text="I will call the sort function on the file."), **p) == "missing_call"
    assert classify_step(row(raw_text="Let me check the folder first."), **p) == "missing_call"
    assert classify_step(row(raw_text="<think>\nhmm\n</think>\n\n"), **p) == "missing_call"   # nothing visible
    # only the first step of a turn with ground truth; other no-call steps keep their classes
    assert classify_step(row(raw_text="Done.", step=2), **p) == "end_turn"
    assert classify_step(row(raw_text="It is 42.", turn_has_ground_truth=False), **p) == "abstain_ok"
    assert classify_step(row(raw_text="It is 42.", benchmark="tau2_verified_retail", turn_has_ground_truth=False),
                         **p) == "reply"


def test_class_lists():
    from testbed.classifier import ALL_CLASSES, CLASSES, FAILURES, PROPOSED_CLASSES
    assert ALL_CLASSES == CLASSES and set(PROPOSED_CLASSES) <= set(CLASSES)        # decided: in v0 (3a/3b)
    assert set(PROPOSED_CLASSES) <= FAILURES <= set(CLASSES)
    assert CLASSES.index("tool_error_payload") == CLASSES.index("exec_error") + 1
    assert CLASSES.index("missing_call_direct_answer") == CLASSES.index("missing_call") + 1
