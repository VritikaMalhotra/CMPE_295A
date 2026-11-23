"""R2c step-selective routing (testbed.backends.RoutingBackend) and its run-spec encoding."""
import pytest

from testbed import prompt_formats, registry as reg
from testbed.backends import RoutingBackend, opens_call
from testbed.spec import GenerateResult

TASKS = [f"multi_turn_base_{i}" for i in range(5)]


def test_first_token_rule():
    q = prompt_formats.QWEN3.call_markers
    assert opens_call("<tool_call>", q)
    assert opens_call("\n<tool_call>", q)                 # leading whitespace
    assert opens_call("<tool", q)                         # a template that splits the marker
    assert not opens_call("I", q) and not opens_call("", q) and not opens_call("To", q)
    lfm = prompt_formats.for_model("lfm2.5-1.2b-instruct").call_markers
    assert opens_call("<|tool_call_start|>", lfm) and not opens_call("Sure", lfm)


class Fake:
    def __init__(self, name, first, fmt=prompt_formats.QWEN3):
        self.name, self.first, self.fmt, self.calls = name, first, fmt, []

    def probe(self, messages, tools, constraint=None, seed=0):
        self.calls.append(("probe", seed))
        return self.first, 123, 4.5

    def generate(self, messages, tools, constraint=None, seed=0):
        self.calls.append(("generate", seed))
        return GenerateResult(text=self.name, raw_text=self.name, tool_calls=[], tool_call_blocks=0, prompt_tokens=123,
                              completion_tokens=7, ttft_ms=1.0, decode_ms=2.0, latency_ms=3.0, stop_reason="eos",
                              grammar_used=False)


@pytest.mark.parametrize("first,served", [("<tool_call>", "q8_0"), ("Hello", "r2c_tool")])
def test_routing_sends_call_steps_to_the_high_copy(first, served):
    low, high = Fake("low", first), Fake("high", "unused")
    rb = RoutingBackend(low, high, "r2c_tool", "q8_0")
    g = rb.generate([], [], seed=6)
    assert g.served_by == served and g.probe_prompt_tokens == 123 and g.probe_ms == 4.5
    assert low.calls[0] == ("probe", 6)                    # the probe uses the step's own seed
    gen = high if served == "q8_0" else low
    assert ("generate", 6) in gen.calls
    assert rb.counts[served] == 1


def test_thinking_models_are_refused():
    think = prompt_formats.for_model("lfm2.5-2.6b")
    with pytest.raises(ValueError):
        RoutingBackend(Fake("l", "x", think), Fake("h", "x", think), "r2c_tool", "q8_0")


def test_route_is_part_of_the_run_id(skeleton_exp):
    before = {s.run_id for s in reg.expand(skeleton_exp, TASKS)}
    route = {"route": {"policy": "tool_call", "high": "q8_0"}}
    exp = {**skeleton_exp, "repairs": list(skeleton_exp["repairs"]) + [
        {"repair": "r2", "variant": "r2c_tool", "backend": "cuda", "constraint": "none", "server": route}]}
    specs = reg.expand(exp, TASKS)
    r2c = [s for s in specs if s.variant == "r2c_tool"]
    assert r2c and all(s.server["route"]["high"] == "q8_0" and "ctx" in s.server for s in r2c)
    assert before <= {s.run_id for s in specs}             # existing specs keep their ids
