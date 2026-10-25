"""TestbedAgent bookkeeping (turns, steps, tool results, malformed and empty replies) against a minimal
stand-in for the tau2 classes it touches. The real harness runs in the Colab integration run; this test
pins the testbed side: what gets logged and what goes back to the orchestrator."""
import sys
import types
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import pytest

from testbed.spec import GenerateResult, RunSpec, ToolCall as TBCall


def _install_fake_tau2(monkeypatch):
    S = TypeVar("S")

    class LocalAgent(Generic[S]):
        def __init__(self, tools, domain_policy):
            self.tools, self.domain_policy = tools, domain_policy

    @dataclass
    class SystemMessage:
        role: str
        content: str

    @dataclass
    class UserMessage:
        role: str
        content: str

    @dataclass
    class ToolCall:
        id: str
        name: str
        arguments: dict

    @dataclass
    class AssistantMessage:
        role: str
        content: Optional[str] = None
        tool_calls: Optional[list] = None

    @dataclass
    class ToolMessage:
        id: str
        role: str
        content: str
        error: bool = False

    @dataclass
    class MultiToolMessage:
        role: str
        tool_messages: list = field(default_factory=list)

    @dataclass
    class LLMAgentState:
        system_messages: list
        messages: list

    mods = {
        "tau2": types.ModuleType("tau2"), "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.base": types.ModuleType("tau2.agent.base"),
        "tau2.agent.llm_agent": types.ModuleType("tau2.agent.llm_agent"),
        "tau2.data_model": types.ModuleType("tau2.data_model"),
        "tau2.data_model.message": types.ModuleType("tau2.data_model.message"),
    }
    mods["tau2.agent.base"].LocalAgent = LocalAgent
    mods["tau2.agent.llm_agent"].AGENT_INSTRUCTION = "be an agent"
    mods["tau2.agent.llm_agent"].SYSTEM_PROMPT = "<instructions>\n{agent_instruction}\n</instructions>\n<policy>\n{domain_policy}\n</policy>"
    mods["tau2.agent.llm_agent"].LLMAgentState = LLMAgentState
    m = mods["tau2.data_model.message"]
    for c in (SystemMessage, UserMessage, ToolCall, AssistantMessage, ToolMessage, MultiToolMessage):
        setattr(m, c.__name__, c)
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return m


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, messages, tools, constraint=None, seed=0):
        self.prompts.append(messages)
        raw = self.replies.pop(0)
        from testbed import qwen_format
        calls, n = qwen_format.extract_tool_calls(raw)
        good = [c for c in calls if isinstance(c, dict) and "name" in c and "arguments" in c]
        return GenerateResult(text=raw, raw_text=raw, tool_calls=[TBCall(c["name"], c["arguments"]) for c in good],
                              tool_call_blocks=n, prompt_tokens=100, completion_tokens=10, ttft_ms=1.0, decode_ms=1.0,
                              latency_ms=2.0, stop_reason="eos", grammar_used=False,
                              bad_call_blocks=len(calls) - len(good))


TOOLS = [{"type": "function", "function": {"name": "get_user_details", "description": "d",
          "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}]


def _agent(monkeypatch, replies):
    m = _install_fake_tau2(monkeypatch)
    from testbed.tau2_bench import Tau2Task, make_agent_class
    task = Tau2Task(id="0", benchmark="tau2_verified_airline", tools=TOOLS, policy="policy")
    spec = RunSpec(experiment="t", model="m", variant="q4_0", backend="cuda", repair="none", constraint="none",
                   benchmark="tau2_verified_airline", task_id="0", seed=1)
    rows = []
    be = ScriptedBackend(replies)
    Agent = make_agent_class()
    a = Agent(tools=[], domain_policy="policy", task=task, spec=spec, backend=be, grammar=None, log_step=rows.append)
    return m, a, be, rows


CALL = '<tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "emma_kim_9957"}}\n</tool_call>'


def test_turns_steps_and_tool_results(monkeypatch):
    m, a, be, rows = _agent(monkeypatch, ["What is your user id?", CALL, "Your reservation cannot be cancelled."])
    st = a.get_init_state([m.AssistantMessage(role="assistant", content="Hi! How can I help you today?")])
    msg, st = a.generate_next_message(m.UserMessage(role="user", content="Cancel EHGLP3"), st)
    assert msg.content == "What is your user id?" and msg.tool_calls is None
    msg, st = a.generate_next_message(m.UserMessage(role="user", content="emma_kim_9957"), st)
    assert msg.content is None and msg.tool_calls[0].name == "get_user_details"
    msg, st = a.generate_next_message(m.ToolMessage(id=msg.tool_calls[0].id, role="tool", content='{"name": "Emma"}'), st)
    a.stop(m.UserMessage(role="user", content="###STOP###"), st)
    assert [(r.turn, r.step) for r in rows] == [(0, 0), (1, 0), (1, 1)]
    assert [r.decoded_calls != [] for r in rows] == [False, True, False]
    assert rows[1].execution_results == ['{"name": "Emma"}'] and rows[1].schema_errors == []
    assert all(r.benchmark == "tau2_verified_airline" and r.is_last_step_of_turn == (i != 1) for i, r in enumerate(rows))
    # the prompt carries tau2's system prompt, the greeting, and the tool result as a tool message
    p = be.prompts[2]
    assert p[0]["role"] == "system" and "policy" in p[0]["content"]
    assert p[1] == {"role": "assistant", "content": "Hi! How can I help you today?"}
    assert p[-1] == {"role": "tool", "content": '{"name": "Emma"}'}
    assert p[-2]["tool_calls"] == [{"name": "get_user_details", "arguments": {"user_id": "emma_kim_9957"}}]


def test_tool_errors_get_the_bfcl_prefix(monkeypatch):
    m, a, be, rows = _agent(monkeypatch, [CALL, "Sorry."])
    st = a.get_init_state([])
    msg, st = a.generate_next_message(m.UserMessage(role="user", content="hi"), st)
    a.generate_next_message(m.ToolMessage(id="x", role="tool", content="User not found", error=True), st)
    a.stop(None, st)
    assert rows[0].execution_results == ["Error during execution: User not found"]


def test_malformed_block_goes_to_the_user_as_text(monkeypatch):
    bad = '<tool_call>\n["get_user_details"]\n</tool_call>'
    m, a, be, rows = _agent(monkeypatch, [bad])
    st = a.get_init_state([])
    msg, st = a.generate_next_message(m.UserMessage(role="user", content="hi"), st)
    a.stop(None, st)
    assert msg.tool_calls is None and msg.content == bad and rows[0].decoded_calls == []
    from testbed.classifier import classify_step
    assert classify_step(rows[0]) == "malformed"


def test_empty_reply_ends_the_episode(monkeypatch):
    from testbed.tau2_bench import AgentEmptyReply
    m, a, be, rows = _agent(monkeypatch, ["   "])
    st = a.get_init_state([])
    with pytest.raises(AgentEmptyReply):
        a.generate_next_message(m.UserMessage(role="user", content="hi"), st)
    assert len(rows) == 1
