"""C10: planted-effect check of the registered mixed-effects models (analysis/mixed_effects.R)."""
import pytest

from testbed.mixed_effects import lme4_available, planted_effect_check, run_mixed_effects, simulate

needs_r = pytest.mark.skipif(not lme4_available(), reason="Rscript with lme4 not available")


def test_simulate_matches_logger_schema():
    s, e, truth = simulate(n_tasks=5, n_seeds=2)
    assert {"run_id", "task_id", "variant", "turn", "n_tool_call_blocks", "failure_class"} <= set(s.columns)
    assert {"run_id", "task_id", "variant", "valid", "n_steps"} <= set(e.columns)
    assert len(e) == 5 * 2 * 2 and e.n_steps.sum() == len(s)
    assert set(s.run_id) == set(e.run_id)
    assert not s[s.failure_class == "end_turn"].empty and set(truth) == {"A", "B", "C"}


@needs_r
def test_planted_effects_recovered_and_null_not_flagged():
    r = planted_effect_check()
    assert r["status"] == {"A": "ok", "B": "ok", "C": "ok"}
    assert r["recovered"], r["terms"]
    assert not r["null_false_positive"] and r["null_covered"], r["null_terms"]
    assert any("end" in n for n in r["notes"])        # "end" steps cannot fail and are dropped from model B


@needs_r
def test_model_a_switches_to_negative_binomial_above_threshold():
    s, e, truth = simulate(n_tasks=60, seed=3)
    m = run_mixed_effects(s, e, models="A", dispersion_threshold=0.0)["models"]["A"]
    assert m["status"] == "ok" and m["family"] == "negative_binomial" and m["nb_theta"] > 0
    q4 = next(r for r in m["fixed"] if r["term"] == "configurationq4_0")
    assert q4["ci_low"] <= truth["A"]["configurationq4_0"] <= q4["ci_high"]
