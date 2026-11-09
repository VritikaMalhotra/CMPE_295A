"""C1 for tau2: all verified tasks across domains, with the database-write rule, against a stand-in tau2."""
import sys
import types
from types import SimpleNamespace as NS

import pytest

from testbed import tau2_bench as tb


def _task(i, *actions):
    return NS(id=i, evaluation_criteria=NS(actions=[NS(name=n, requestor=r) for n, r in actions]))


TASKS = {
    "airline": [_task("0", ("get_user_details", "assistant")), _task("1", ("cancel_reservation", "assistant"))],
    "retail": [_task("0", ("cancel_pending_order", "assistant")), _task("1")],
    # telecom: a write the simulated user does on its phone is not the agent's write
    "telecom": [_task("[a]x", ("toggle_airplane_mode", "user")), _task("[b]y", ("enable_roaming", "assistant"))],
}
WRITES = {"airline": {"cancel_reservation"}, "retail": {"cancel_pending_order"},
          "telecom": {"enable_roaming", "toggle_airplane_mode"}}   # toggle_* would be a user tool; tests requestor


@pytest.fixture
def fake_tau2(monkeypatch):
    class Kit:
        def __init__(self, d):
            self.d = d

        def get_tools(self):
            return {n: None for n in WRITES[self.d] | {"get_details"}}

        def tool_type(self, n):
            return "write" if n in WRITES[self.d] else "read"

    class Env:
        def __init__(self, d):
            self.tools = Kit(d)

        def get_tools(self):
            return [NS(openai_schema={"type": "function", "function": {"name": n, "parameters": {}}})
                    for n in sorted(self.tools.get_tools())]

        def get_policy(self):
            return "policy"

    loaded = []

    def load_tasks(d, split):
        loaded.append((d, split))
        return TASKS[d]

    mods = {
        "tau2": types.ModuleType("tau2"),
        "tau2.registry": types.SimpleNamespace(registry=NS(get_env_constructor=lambda d: (lambda: Env(d)))),
        "tau2.run": types.SimpleNamespace(load_tasks=load_tasks),
        "tau2.environment": types.ModuleType("tau2.environment"),
        "tau2.environment.toolkit": types.SimpleNamespace(ToolType=NS(WRITE="write")),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(tb, "_quiet_litellm", lambda: None)
    return loaded


def test_all_domains_without_the_write_rule(fake_tau2):
    tasks, excluded = tb.load_tau2_verified(require_write=False)
    assert [(t.domain, t.id) for t in tasks] == [(d, t.id) for d in tb.VERIFIED_DOMAINS for t in TASKS[d]]
    assert {t.benchmark for t in tasks} == {f"tau2_verified_{d}" for d in tb.VERIFIED_DOMAINS}
    assert excluded == {d: [] for d in tb.VERIFIED_DOMAINS}
    assert all(t.tool_names == WRITES[t.domain] | {"get_details"} for t in tasks)


def test_write_rule_uses_tau2_write_marks_and_agent_actions_only(fake_tau2):
    tasks, excluded = tb.load_tau2_verified()
    assert [(t.domain, t.id) for t in tasks] == [("airline", "1"), ("retail", "0"), ("telecom", "[b]y")]
    assert excluded == {"airline": ["0"], "retail": ["1"], "telecom": ["[a]x"]}
    assert {s for _, s in fake_tau2} == {"base"}


def test_explicit_write_tools_override_per_domain(fake_tau2):
    tasks, excluded = tb.load_tau2_verified(domains=("airline",), write_tools={"airline": ["get_user_details"]})
    assert [t.id for t in tasks] == ["0"] and excluded == {"airline": ["1"]}


def test_single_domain_loader_keeps_its_airline_default(fake_tau2):
    (t,) = tb.load_tau2_tasks(["1"], write_tools=["cancel_reservation"])
    assert t.domain == "airline" and t.benchmark == tb.BENCHMARK
    with pytest.raises(ValueError, match="need no database write"):
        tb.load_tau2_tasks(["0"], write_tools=["cancel_reservation"])
    (r,) = tb.load_tau2_tasks(["0"], domain="retail")
    assert r.benchmark == "tau2_verified_retail"


def test_domain_write_tools(fake_tau2):
    assert tb.domain_write_tools("airline") == ["cancel_reservation"]
