"""FR-E2: our episode verdicts against the official evaluators, on logged trajectories (no model calls).

* BFCL: each step's logged raw text goes through bfcl-eval's own path for a local Qwen3 FC model, exactly as
  ``bfcl evaluate`` scores a result file: ``QwenFCHandler._parse_query_response_prompting`` (strips the think
  block), ``eval_runner._evaluate_single_multi_turn_entry`` (turn-count check, ``decode_execute``,
  ``multi_turn_checker``), with prompt and ground-truth entries from bfcl's own loaders
  (``load_dataset_entry(include_language_specific_hint=False)``, ``load_ground_truth_entry``), not ours.
  SmolLM3 and LFM2.5 have no bfcl-eval handler: for them the same path runs with one change, the tool-call
  extraction comes from the testbed's own parser for that model (``testbed.prompt_formats``), so their BFCL
  re-scoring checks the scoring path but not the parser. The JSON records which parser each model used.
* tau2: each saved simulation (``tau2_sims/<run_id>.json``) is re-scored by tau2's ``evaluate_simulation``
  with the task loaded by tau2's ``load_tasks``; verdict = reward 1.0. Episodes with no simulation (the
  agent's or the simulated user's context ran out, or the agent's reply was empty) were scored without the
  evaluator and are counted apart.

bfcl-eval's eval_runner imports every API client (anthropic, google-genai, ...). Those packages are not
installed, so they are replaced by inert stand-ins; multi-turn scoring never calls them.

    python scripts/evaluator_agreement.py <ref_run_dir> [...] --out results/m2/evaluator_agreement.json
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import sys
import types
from collections import Counter
from types import SimpleNamespace

import pandas as pd

TARGET = 1.0            # FR-E2: 100% verdict agreement on >= 200 replayed trajectories
MIN_N = 200


# ---- bfcl-eval's runner with stand-ins for the API clients it imports but multi-turn scoring never uses ----
class _Inert:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Inert()

    def __getattr__(self, n):
        return _Inert()


class _StubModule(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return type(n, (_Inert, Exception), {}) if n[:1].isupper() else _Inert()


def _stub(name: str) -> None:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            m = _StubModule(n)
            m.__path__ = []
            sys.modules[n] = m


def official_bfcl():
    stubbed = []
    for _ in range(80):
        try:
            import bfcl_eval.eval_checker.eval_runner as er
            from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
            from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
            return er, QwenFCHandler, load_dataset_entry, load_ground_truth_entry, sorted(stubbed)
        except ModuleNotFoundError as e:
            if e.name.startswith("bfcl_eval"):
                raise
            stubbed.append(e.name)
            _stub(e.name)
            if e.name == "overrides":           # a decorator: must hand the method back unchanged
                sys.modules["overrides"].override = sys.modules["overrides"].final = lambda f: f
        except ImportError as e:
            m = re.match(r"cannot import name '(\w+)' from '([\w.]+)'", str(e))
            if not m or m.group(2).startswith("bfcl_eval"):
                raise
            full = f"{m.group(2)}.{m.group(1)}"
            stubbed.append(full)
            _stub(full)
            setattr(importlib.import_module(m.group(2)), m.group(1), sys.modules[full])
        for k in [k for k in sys.modules if k.startswith("bfcl_eval")]:
            del sys.modules[k]
    raise RuntimeError("could not import bfcl_eval.eval_checker.eval_runner")


def _clean_executor_state(tag: str) -> None:
    """bfcl's executor keeps tool instances in module globals keyed by model name; drop this episode's."""
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mu
    for k in [k for k in vars(mu) if tag in k]:
        delattr(mu, k)


def make_handler(QwenFCHandler, model: str):
    """bfcl-eval's Qwen3 FC handler for Qwen3; for a model bfcl has no handler for, the same class with its
    tool-call extraction replaced by the testbed's parser for that model's format."""
    from testbed import prompt_formats
    fmt = prompt_formats.for_model(model)
    if fmt is prompt_formats.QWEN3:
        return QwenFCHandler(model_name="Qwen/Qwen3", temperature=0.7, registry_name="fr-e2", is_fc_model=False), \
            "bfcl-eval QwenFCHandler"

    class FormatHandler(QwenFCHandler):
        @staticmethod
        def _extract_tool_calls(text):
            calls, _ = fmt.extract_tool_calls(fmt.split_think(text)[1])
            return [c for c in calls if isinstance(c, dict) and "name" in c and "arguments" in c]

    return FormatHandler(model_name="Qwen/Qwen3", temperature=0.7, registry_name="fr-e2", is_fc_model=False), \
        f"testbed prompt_formats.{fmt.name} parser (no bfcl-eval handler) + bfcl-eval scoring"


def check_bfcl(run_dir: str, eps: pd.DataFrame, steps: pd.DataFrame, official, model: str = "qwen3") -> dict:
    er, QwenFCHandler, load_dataset_entry, load_ground_truth_entry, _ = official
    handler, parser = make_handler(QwenFCHandler, model)
    eps = eps[eps.benchmark.str.startswith("bfcl_")]
    by_run = {rid: g.sort_values(["turn", "step"]) for rid, g in steps[steps.run_id.isin(set(eps.run_id))]
              .groupby("run_id")}
    prompts, answers = {}, {}
    out = {"n": 0, "agree": 0, "disagree": [], "by_benchmark": Counter(), "agree_by_benchmark": Counter(),
           "parser": parser}
    for e in eps.itertuples():
        cat = e.benchmark[len("bfcl_"):]
        if cat not in prompts:
            prompts[cat] = {p["id"]: p for p in load_dataset_entry(cat, include_prereq=False,
                                                                   include_language_specific_hint=False)}
            answers[cat] = {a["id"]: a["ground_truth"] for a in load_ground_truth_entry(cat)}
        g = by_run.get(e.run_id)
        turns: list[list[str]] = []
        if g is not None:
            for _, tg in g.groupby("turn", sort=True):
                turns.append([handler._parse_query_response_prompting(SimpleNamespace(
                    choices=[SimpleNamespace(text=t or "")], usage=SimpleNamespace(prompt_tokens=0,
                                                                                    completion_tokens=0)))
                    ["model_responses"] for t in tg.raw_text])
        tag = "fre2_" + re.sub(r"\W", "_", e.run_id)
        res = er._evaluate_single_multi_turn_entry(handler, e.task_id, turns, answers[cat][e.task_id],
                                                   copy.deepcopy(prompts[cat][e.task_id]), tag, cat)
        _clean_executor_state(tag)
        official_valid = bool(res["valid"])
        out["n"] += 1
        out["by_benchmark"][e.benchmark] += 1
        if official_valid == bool(e.valid):
            out["agree"] += 1
            out["agree_by_benchmark"][e.benchmark] += 1
        else:
            out["disagree"].append({"run_id": e.run_id, "task_id": e.task_id, "ours": bool(e.valid),
                                    "ours_error": e.error_type, "official": official_valid,
                                    "official_error": (res.get("error") or {}).get("error_type", "")})
    return out


def check_tau2(run_dir: str, eps: pd.DataFrame) -> dict:
    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.run import load_tasks
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    eps = eps[eps.benchmark.str.startswith("tau2_")]
    tasks: dict = {}
    out = {"n": 0, "agree": 0, "disagree": [], "no_simulation": Counter(), "by_benchmark": Counter(),
           "agree_by_benchmark": Counter()}
    for e in eps.itertuples():
        p = os.path.join(run_dir, "tau2_sims", f"{e.run_id}.json")
        sim = json.load(open(p)).get("simulation") if os.path.exists(p) else None
        if sim is None:
            out["no_simulation"][e.error_type or "missing file"] += 1
            continue
        domain = e.benchmark[len("tau2_verified_"):]
        if domain not in tasks:
            tasks[domain] = {t.id: t for t in load_tasks(domain, "base")}
        info = evaluate_simulation(simulation=SimulationRun.model_validate(sim), task=tasks[domain][e.task_id],
                                   evaluation_type=EvaluationType.ALL, solo_mode=False, domain=domain)
        official_valid = float(info.reward) == 1.0
        out["n"] += 1
        out["by_benchmark"][e.benchmark] += 1
        if official_valid == bool(e.valid):
            out["agree"] += 1
            out["agree_by_benchmark"][e.benchmark] += 1
        else:
            out["disagree"].append({"run_id": e.run_id, "task_id": e.task_id, "ours": bool(e.valid),
                                    "ours_error": e.error_type, "official": official_valid,
                                    "official_reward": float(info.reward)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-tau2", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes per run (a quick try)")
    a = ap.parse_args()
    official = official_bfcl()
    result = {"criterion": f"FR-E2: {TARGET:.0%} verdict agreement with the official evaluators on >= {MIN_N} "
                           "replayed trajectories", "stand_ins": official[-1], "models": {}}
    tot_n = tot_agree = 0
    for d in a.run_dirs:
        name = json.load(open(os.path.join(d, "manifest.json"))).get("experiment", d)
        eps = pd.read_parquet(os.path.join(d, "episodes.parquet"))
        specs = [json.loads(line) for line in open(os.path.join(d, "specs.jsonl")) if line.strip()]
        bench = {x["run_id"]: x["benchmark"] for x in specs}
        model = specs[0].get("model", "qwen3") if specs else "qwen3"
        eps["benchmark"] = eps.run_id.map(bench)
        if a.limit:
            eps = pd.concat([eps[eps.benchmark.str.startswith("bfcl_")].head(a.limit),
                             eps[eps.benchmark.str.startswith("tau2_")].head(a.limit)])
        steps = pd.read_parquet(os.path.join(d, "steps.parquet"), columns=["run_id", "turn", "step", "raw_text"])
        r = {"bfcl": check_bfcl(d, eps, steps, official, model)}
        if not a.skip_tau2:
            r["tau2"] = check_tau2(d, eps)
        for k, v in r.items():
            tot_n += v["n"]
            tot_agree += v["agree"]
            print(f"{name} {k}: {v['agree']}/{v['n']} agree, {len(v['disagree'])} disagree"
                  + (f"; no simulation: {dict(v['no_simulation'])}" if v.get("no_simulation") else ""), flush=True)
            for x in v["disagree"][:10]:
                print("   ", x)
            v["disagree_total"] = len(v["disagree"])
            v["disagree"] = v["disagree"][:50]
        result["models"][name] = r
    rate = tot_agree / tot_n if tot_n else 0.0
    result.update({"n": tot_n, "agree": tot_agree, "agreement": rate,
                   "pass": tot_n >= MIN_N and rate >= TARGET})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1, default=dict)
    print(f"\nall: {tot_agree}/{tot_n} = {rate:.4%}; {'PASS' if result['pass'] else 'FAIL'}; wrote {a.out}")


if __name__ == "__main__":
    main()
