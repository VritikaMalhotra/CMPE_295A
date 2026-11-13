#!/usr/bin/env python
"""Domain counts behind docs/agents_and_data.md (feedback point 3): BFCL multi-turn API classes in the study's
fixed-seed sample, and the tau2 domains, read from the installed bfcl-eval and tau2 data.

    python scripts/benchmark_domains.py --out results/m3/benchmark_domains.json   (needs tau2: cluster venv)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testbed import adapter  # noqa: E402

CATS = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"bfcl": {"sample_seed": adapter.SAMPLE_SEED, "per_category": adapter.SAMPLE_PER_CATEGORY, "categories": {},
                    "api_classes": {}}, "tau2": {}}
    per_class = collections.defaultdict(lambda: {c: 0 for c in CATS})
    n_classes = collections.Counter()
    for cat in CATS:
        full = adapter._jsonl(os.path.join(adapter.bfcl_data_dir(), f"BFCL_v4_{cat}.json"))
        tasks = adapter.load_bfcl_multi_turn(cat, ids=adapter.sample_task_ids(cat))
        turns = [len(t.turns) for t in tasks]
        out["bfcl"]["categories"][cat] = {"tasks_in_bfcl": len(full), "tasks_sampled": len(tasks),
                                          "turns_mean": round(sum(turns) / len(turns), 2),
                                          "turns_min": min(turns), "turns_max": max(turns)}
        for t in tasks:
            n_classes[len(t.involved_classes)] += 1
            for c in t.involved_classes:
                per_class[c][cat] += 1
    for c, counts in sorted(per_class.items(), key=lambda kv: -sum(kv[1].values())):
        out["bfcl"]["api_classes"][c] = {"functions": len(adapter._func_docs(c)), "tasks": counts,
                                         "tasks_total": sum(counts.values())}
    out["bfcl"]["tasks_by_number_of_classes"] = dict(sorted(n_classes.items()))
    try:
        from tau2.registry import registry
        from tau2.run import load_tasks
        from testbed.tau2_bench import domain_write_tools, load_tau2_verified
        _, excluded = load_tau2_verified()          # the write-task rule (decision 6), as the runs apply it
        for d in ("airline", "retail", "telecom"):
            env = registry.get_env_constructor(d)()
            try:
                user_tools = len(env.get_user_tools())
            except Exception:
                user_tools = 0
            base = len(load_tasks(d, "base"))
            out["tau2"][d] = {"base_tasks": base, "write_tasks": base - len(excluded[d]),
                              "agent_tools": len(env.get_tools()), "user_tools": user_tools,
                              "write_tools": len(domain_write_tools(d))}
    except ImportError as e:
        out["tau2"] = {"error": f"tau2 not installed: {e}"}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()
