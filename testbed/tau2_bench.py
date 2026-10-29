"""tau2-bench-verified airline: C1 loader + the C2 episode for a simulated-user benchmark.

The harness is Amazon AGI's tau2-bench-verified (pinned commit in the experiment config), used unmodified:
its environment executes tools and keeps the database, its ``UserSimulator`` plays the customer, its
``Orchestrator`` routes messages and enforces the step and error limits, and its evaluator gives the reward.
Only the agent is ours: ``TestbedAgent`` renders the conversation with the same Qwen3 prompt code as BFCL
(``qwen_format``), calls C3 ``Backend.generate`` (so R1 grammars, R2/R3 model files, both backends and the
step timings work unchanged) and logs one ``StepRow`` per model step for C5/C6.

The simulated user and the NL-assertion judge reach the self-hosted simulator through LiteLLM's
``openai/`` route with ``api_base`` pointing at the C4 cache proxy (``simuser_cache``), so a finished
episode replays without simulator calls.

Mapping to the testbed's records:
* turn  = index of the user message the agent is answering (0 = the opening request); step = model call
  within that turn, as in BFCL.
* a reply that parses to tool calls goes to the environment; anything else goes to the user as text (a
  malformed ``<tool_call>`` block is sent as text, as an OpenAI-style parser would leave it in ``content``).
* an empty reply ends the episode (tau2's orchestrator rejects an empty agent message), scored 0.
* tool results that tau2 flags as errors are logged with bfcl's "Error during execution" prefix so C6's
  ``exec_error`` rule applies to both benchmarks.
* episode valid iff tau2's reward == 1.0 (its default airline reward basis: database state and
  communicated info). The NL-assertion judge also runs, but only as a diagnostic, since it is not part of the
  airline reward basis.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .backends import ContextOverflow
from .classifier import schema_errors
from .spec import EpisodeResult, RunSpec, StepRow

DOMAIN = "airline"
BENCHMARK = "tau2_verified_airline"


class SimulatorUnavailable(RuntimeError):
    """The simulated user could not be reached (or the C4 budget stopped it): retry, never score."""


class AgentEmptyReply(RuntimeError):
    """The model produced neither text nor a tool call."""


_QUIET = False


def _quiet_litellm() -> None:
    """No network fetch at import, and tau2/LiteLLM logs at WARNING: their DEBUG/INFO lines (every message
    of every conversation) buried the run log on Oct 25. The full conversation is kept in tau2_sims/."""
    global _QUIET
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    if _QUIET:
        return
    _QUIET = True
    try:
        import logging
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING", filter=lambda r: "isn't mapped yet" not in r["message"]
                   and "has no cost" not in r["message"])
        for name in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy", "httpx"):
            logging.getLogger(name).setLevel(logging.ERROR)
    except Exception:
        pass


@dataclass
class Tau2Task:
    """C1 view of one tau2 task: what C6's schema checks and R1's grammar need, plus the tau2 object."""
    id: str
    benchmark: str
    tools: list[dict]                   # OpenAI function schemas ({"type": "function", "function": {...}})
    policy: str
    raw: object = field(repr=False, default=None)   # tau2.data_model.tasks.Task

    @property
    def functions(self) -> list[dict]:
        return [t["function"] for t in self.tools]

    @property
    def tool_names(self) -> set[str]:
        return {f["name"] for f in self.functions}

    def tool_schema(self, name: str) -> dict | None:
        for f in self.functions:
            if f["name"] == name:
                return f.get("parameters", {})
        return None


def write_actions(task_raw, write_tools) -> list[str]:
    """Gold actions of a tau2 task that change the database (empty = a do-nothing agent can pass)."""
    ec = getattr(task_raw, "evaluation_criteria", None)
    acts = (getattr(ec, "actions", None) or []) if ec is not None else []
    return [a.name for a in acts if a.name in set(write_tools)]


def load_tau2_tasks(ids: list[str], split: str = "base", write_tools: list[str] | None = None) -> list[Tau2Task]:
    """``write_tools`` given: every task must need at least one database write, so that an agent which
    never acts cannot score 1.0 (the trap tasks 0-2 fell into on Oct 25)."""
    _quiet_litellm()
    from tau2.registry import registry
    from tau2.run import load_tasks
    env = registry.get_env_constructor(DOMAIN)()
    tools = [t.openai_schema for t in env.get_tools()]
    policy = env.get_policy()
    by_id = {t.id: t for t in load_tasks(DOMAIN, split)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"tau2 {DOMAIN}/{split} has no tasks {missing}")
    if write_tools:
        no_write = [i for i in ids if not write_actions(by_id[i], write_tools)]
        if no_write:
            raise ValueError(f"tau2 tasks {no_write} need no database write; a do-nothing agent passes them")
    return [Tau2Task(id=i, benchmark=BENCHMARK, tools=tools, policy=policy, raw=by_id[i]) for i in ids]


def _to_testbed_messages(state_msgs) -> list[dict]:
    """tau2 messages -> the dicts qwen_format renders (same shapes the BFCL runner builds)."""
    from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
    out = []
    for m in state_msgs:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content or ""})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content or ""})
        elif isinstance(m, AssistantMessage):
            if m.tool_calls:
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls]})
            else:
                out.append({"role": "assistant", "content": m.content or ""})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content or ""})
    return out


def make_agent_class():
    """Built lazily so importing this module does not need tau2 installed."""
    from tau2.agent.base import LocalAgent
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgentState
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, ToolMessage

    class TestbedAgent(LocalAgent[LLMAgentState]):
        def __init__(self, tools, domain_policy, *, task: Tau2Task, spec: RunSpec, backend,
                     grammar: Optional[str], log_step: Callable[[StepRow], None]):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.task, self.spec, self.backend, self.grammar, self.log_step = task, spec, backend, grammar, log_step
            self.seed = spec.seed
            self.turn = -1
            self.step_in_turn = 0
            self.n_steps = 0
            self._pending: Optional[StepRow] = None
            self.overflow = False

        @property
        def system_prompt(self) -> str:
            return SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)

        def get_init_state(self, message_history=None):
            return LLMAgentState(system_messages=[SystemMessage(role="system", content=self.system_prompt)],
                                 messages=list(message_history or []))

        def set_seed(self, seed: int):
            self.seed = int(seed)

        def _flush(self, tool_msgs=None, last=False):
            row = self._pending
            if row is None:
                return
            if tool_msgs:
                row.execution_results = [("Error during execution: " if t.error else "") + (t.content or "")
                                         for t in tool_msgs]
            row.is_last_step_of_turn = last or not row.decoded_calls
            self.log_step(row)
            self._pending = None

        def generate_next_message(self, message, state):
            if isinstance(message, MultiToolMessage):
                self._flush(message.tool_messages)
                state.messages.extend(message.tool_messages)
            elif isinstance(message, ToolMessage):
                self._flush([message])
                state.messages.append(message)
            else:                                   # a user message opens a new turn
                self._flush(last=True)
                state.messages.append(message)
                self.turn += 1
                self.step_in_turn = 0
            msgs = _to_testbed_messages(state.system_messages + state.messages)
            try:
                g = self.backend.generate(msgs, self.task.tools, constraint=self.grammar, seed=self.seed)
            except ContextOverflow:
                self.overflow = True
                raise
            self.n_steps += 1
            calls = [{"name": c.name, "arguments": c.arguments if isinstance(c.arguments, dict) else {}}
                     for c in g.tool_calls]
            decode_ok = g.bad_call_blocks == 0 and all(isinstance(c.arguments, dict) for c in g.tool_calls)
            use_calls = calls if (calls and decode_ok) else []
            errs = []
            for c in use_calls:
                errs += schema_errors(c, self.task)
            self._pending = StepRow(
                run_id=self.spec.run_id, config_key=self.spec.config_key, task_id=self.task.id, seed=self.spec.seed,
                turn=max(self.turn, 0), step=self.step_in_turn, variant=self.spec.variant, backend=self.spec.backend,
                repair=self.spec.repair, constraint=self.spec.constraint, prompt_tokens=g.prompt_tokens,
                completion_tokens=g.completion_tokens, ttft_ms=g.ttft_ms, decode_ms=g.decode_ms,
                latency_ms=g.latency_ms, stop_reason=g.stop_reason, raw_text=g.raw_text,
                n_tool_call_blocks=g.tool_call_blocks,
                decoded_calls=[json.dumps(c, sort_keys=True) for c in use_calls], execution_results=[],
                known_tools=all(c["name"] in self.task.tool_names for c in use_calls), schema_errors=errs,
                turn_has_ground_truth=False, is_last_step_of_turn=False, forced_quit=False, benchmark=BENCHMARK)
            self.step_in_turn += 1
            if use_calls:
                msg = AssistantMessage(role="assistant", content=None, tool_calls=[
                    ToolCall(id=f"call_{self.n_steps}_{i}", name=c["name"], arguments=c["arguments"])
                    for i, c in enumerate(use_calls)])
            else:
                text = (g.text or "").strip()
                if not text:
                    self._flush(last=True)
                    raise AgentEmptyReply("empty agent reply")
                msg = AssistantMessage(role="assistant", content=text)
            state.messages.append(msg)
            return msg, state

        def stop(self, message=None, state=None):
            self._flush([message] if isinstance(message, ToolMessage) else
                        (message.tool_messages if isinstance(message, MultiToolMessage) else None), last=True)

    return TestbedAgent


def _litellm_args(api_base: str, temperature: float = 0.0) -> dict:
    return {"api_base": api_base, "api_key": "none", "temperature": float(temperature)}


def run_tau2_episode(spec: RunSpec, task: Tau2Task, backend, grammar_for, log_step, *, api_base: str,
                     cache=None, sim_dir: Optional[str] = None) -> EpisodeResult:
    """One tau2 episode through the unmodified orchestrator and evaluator. ``cache`` (a SimUserCache) is
    checked for upstream failures: any during the episode raises SimulatorUnavailable (episode stays pending)."""
    _quiet_litellm()
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.user.user_simulator import UserSimulator

    t2 = dict(spec.server.get("tau2", {}))
    sim = dict(spec.server.get("simuser", {}))
    # the judge shares the simulator (Workbook Ch. 3); patched at module level, where the evaluator reads it
    nl.DEFAULT_LLM_NL_ASSERTIONS = "openai/judge"
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = _litellm_args(api_base, sim.get("temperature", 0.0))

    env = registry.get_env_constructor(DOMAIN)()
    try:
        user_tools = env.get_user_tools()
    except Exception:
        user_tools = None
    Agent = make_agent_class()
    grammar = grammar_for(task.functions) if grammar_for else None
    agent = Agent(tools=env.get_tools(), domain_policy=env.get_policy(), task=task, spec=spec, backend=backend,
                  grammar=grammar, log_step=log_step)
    user = UserSimulator(tools=user_tools, instructions=str(task.raw.user_scenario), llm="openai/simuser",
                         llm_args=_litellm_args(api_base, sim.get("temperature", 0.0)))
    orch = Orchestrator(domain=DOMAIN, agent=agent, user=user, environment=env, task=task.raw,
                        max_steps=int(t2.get("max_steps", 200)), max_errors=int(t2.get("max_errors", 10)),
                        seed=spec.seed, solo_mode=False, validate_communication=False)
    fails0 = cache.upstream_failures if cache is not None else 0
    t0 = time.time()
    termination, error_type, reward, sim_run, nl_info = "", "", 0.0, None, None
    try:
        sim_run = orch.run()
        tr = sim_run.termination_reason
        termination = str(getattr(tr, "value", tr))     # "user_stop", "max_steps", ... (an enum in tau2)
    except ContextOverflow:
        termination, error_type = "context_overflow", "tau2:context_overflow"
    except AgentEmptyReply:
        termination, error_type = "agent_error", "tau2:empty_agent_reply"
    except Exception as e:
        if cache is not None and cache.upstream_failures > fails0:
            raise SimulatorUnavailable(str(e)) from e
        if "simulator" in str(e).lower() or "ServiceUnavailable" in type(e).__name__:
            raise SimulatorUnavailable(str(e)) from e
        raise
    if cache is not None and cache.upstream_failures > fails0:
        raise SimulatorUnavailable("simulator failed during the episode")
    if sim_run is not None:
        info = evaluate_simulation(simulation=sim_run, task=task.raw, evaluation_type=EvaluationType.ALL,
                                   solo_mode=False, domain=DOMAIN)
        reward = float(info.reward)
        sim_run.reward_info = info
        error_type = "" if reward == 1.0 else f"tau2:{termination}:reward_{reward:g}"
        try:                                    # diagnostic only: NL assertions are not in the airline basis
            nl_res = nl.NLAssertionsEvaluator.calculate_reward(task=task.raw, full_trajectory=sim_run.messages)
            nl_info = {"reward": nl_res.reward,
                       "checks": [c.model_dump() for c in (nl_res.nl_assertions or [])]}
        except Exception as e:
            if cache is not None and cache.upstream_failures > fails0:
                raise SimulatorUnavailable(str(e)) from e
            nl_info = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if sim_dir:
        os.makedirs(sim_dir, exist_ok=True)
        with open(os.path.join(sim_dir, f"{spec.run_id}.json.part"), "w") as f:
            json.dump({"run_id": spec.run_id, "termination": termination, "reward": reward, "nl_judge": nl_info,
                       "simulation": sim_run.model_dump(mode="json") if sim_run is not None else None}, f)
        os.replace(os.path.join(sim_dir, f"{spec.run_id}.json.part"), os.path.join(sim_dir, f"{spec.run_id}.json"))
    n_user_turns = agent.turn + 1
    return EpisodeResult(
        run_id=spec.run_id, config_key=spec.config_key, task_id=task.id, seed=spec.seed, variant=spec.variant,
        backend=spec.backend, repair=spec.repair, constraint=spec.constraint, valid=reward == 1.0,
        error_type=error_type, n_turns=n_user_turns, n_steps=agent.n_steps,
        forced_quit=termination.endswith("max_steps"), wall_s=time.time() - t0,
        context_overflow=termination == "context_overflow")
