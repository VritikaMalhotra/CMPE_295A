"""Recompute Workbook 2 Table 27 (power for a 1.5x failure-rate ratio) from the BF16 reference runs.

Registered test (Workbook 2, "Power"): given the failure events of one type in a paired FP/4-bit comparison,
the 4-bit share is 0.5 under the null and 0.6 at a 1.5x amplification (equal step counts). Events needed for
80% power at two-sided alpha = 0.05/10 come from the normal approximation (about 330), times the design
effect D for clustering. Here D, the base rate per step and the steps per episode are measured, not assumed:

* sampled BF16 trials only (temperature > 0), all tasks (decision 11: the step-level tests keep every task);
* p = failures of the type / steps; m = steps per episode;
* D = observed variance of per-task failure counts over the binomial variance, sum_t (y_t - p n_t)^2 /
  sum_t n_t p (1 - p): tasks are the clusters a paired FP/4-bit comparison shares, so this is the
  conservative choice. Pairing within a task removes much of that variance, so D per episode (steps within
  one episode) is reported too and the true D lies between the two;
* events per FP/4-bit episode pair = p m (1 + 1.5); episodes needed = events x D / events per pair.

    python scripts/power_table.py <ref_run_dir> [<ref_run_dir> ...] --out results/m2/power_table.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import NormalDist

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.classifier import FAILURES  # noqa: E402

ALPHA, POWER, RATIO, FAMILY = 0.05, 0.80, 1.5, 10


def events_needed(alpha: float = ALPHA / FAMILY, power: float = POWER, ratio: float = RATIO) -> float:
    p0, p1 = 0.5, ratio / (1 + ratio)
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    return ((za * (p0 * (1 - p0)) ** 0.5 + zb * (p1 * (1 - p1)) ** 0.5) / (p1 - p0)) ** 2


def design_effect(y: pd.Series, n: pd.Series, p: float) -> float:
    """Variance of cluster totals over the binomial variance (1 = no clustering)."""
    if p <= 0 or p >= 1:
        return float("nan")
    return float(((y - p * n) ** 2).sum() / (n * p * (1 - p)).sum())


def table(run_dir: str, n_events: float) -> dict:
    specs = [json.loads(line) for line in open(os.path.join(run_dir, "specs.jsonl")) if line.strip()]
    sampled = {s["run_id"] for s in specs if float(s.get("decoding", {}).get("temperature", 0)) > 0
               and s["variant"] == "bf16" and s["repair"] == "none"}
    steps = pd.read_parquet(os.path.join(run_dir, "steps.parquet"),
                            columns=["run_id", "task_id", "benchmark", "failure_class"])
    steps = steps[steps.run_id.isin(sampled)]
    eps = steps.groupby("run_id").size()
    n_eps, m = len(eps), float(eps.mean())
    tasks_per_cell = steps[["benchmark", "task_id"]].drop_duplicates().shape[0]
    rows = []
    types = sorted(set(steps.failure_class) & FAILURES) + ["any_failure"]
    for ft in types:
        hit = steps.failure_class.isin(FAILURES) if ft == "any_failure" else steps.failure_class == ft
        p = float(hit.mean())
        g = steps.assign(y=hit.astype(int))
        by_task = g.groupby(["benchmark", "task_id"]).agg(y=("y", "sum"), n=("y", "size"))
        by_ep = g.groupby("run_id").agg(y=("y", "sum"), n=("y", "size"))
        d_task, d_ep = design_effect(by_task.y, by_task.n, p), design_effect(by_ep.y, by_ep.n, p)
        per_pair = p * m * (1 + RATIO)
        need = n_events * d_task / per_pair if per_pair > 0 else float("inf")
        need_ep = n_events * d_ep / per_pair if per_pair > 0 else float("inf")
        rows.append({"failure_type": ft, "events_fp": int(hit.sum()), "rate_per_step": round(p, 4),
                     "events_per_pair": round(per_pair, 3), "D_task": round(d_task, 2), "D_episode": round(d_ep, 2),
                     "episodes_needed": round(need), "detectable_at_n_eps": bool(need <= n_eps),
                     "episodes_needed_D_episode": round(need_ep), "detectable_D_episode": bool(need_ep <= n_eps)})
    return {"run_dir": run_dir, "sampled_episodes": n_eps, "tasks": tasks_per_cell, "steps_per_episode": round(m, 2),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n_events = events_needed()
    out = {"test": f"binomial share of 4-bit events, 0.5 vs {RATIO / (1 + RATIO):.1f}; two-sided alpha "
                   f"{ALPHA}/{FAMILY}; power {POWER}", "events_needed_D1": round(n_events, 1),
           "models": {}}
    for d in a.run_dirs:
        m = json.load(open(os.path.join(d, "manifest.json"))).get("experiment", os.path.basename(os.path.dirname(d)))
        t = table(d, n_events)
        out["models"][m] = t
        print(f"\n{m}: {t['sampled_episodes']} sampled episodes over {t['tasks']} tasks, "
              f"{t['steps_per_episode']} steps/episode")
        print(f"  {'failure type':<28}{'events':>7}{'rate':>8}{'per pair':>9}{'D task':>8}{'D ep':>7}"
              f"{'eps (D task)':>13}{'eps (D ep)':>11}")
        for r in t["rows"]:
            print(f"  {r['failure_type']:<28}{r['events_fp']:>7}{r['rate_per_step']:>8.4f}{r['events_per_pair']:>9.3f}"
                  f"{r['D_task']:>8.2f}{r['D_episode']:>7.2f}"
                  f"{str(r['episodes_needed']) + ('' if r['detectable_at_n_eps'] else '*'):>13}"
                  f"{str(r['episodes_needed_D_episode']) + ('' if r['detectable_D_episode'] else '*'):>11}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  * more episodes than the reference cell has (not detectable at this size)")
    print(f"\nevents needed at D = 1: {n_events:.0f}; wrote {a.out}")


if __name__ == "__main__":
    main()
