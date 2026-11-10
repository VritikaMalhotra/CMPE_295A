#!/usr/bin/env python
"""Which tasks fail (feedback point 6, 2026-11-09): episode success and step failures of one model's BF16 reference
and its 4-bit cells, by benchmark and domain, by BFCL API class, and by failure class.

Reads the run directories directly (``episodes/*.json``, ``steps/*.jsonl`` of finished episodes), so a cell that
is still running contributes what it has finished. Every 4-bit cell is compared with BF16 **on the same
(task, seed) pairs**. Sampled trials only (seeds 1-4): every rate is the mean over the 4 seeds with ± 1 SD across
seeds (the registered dispersion convention); steps are relabelled with the current classifier (codebook v0).
Development data, preview.

    python analysis/which_tasks_fail.py --model qwen3-0.6b --ref BF16=$TESTBED_OUT/ref_qwen3_0p6b/cd314594fb \\
        --cell Q4_0=$TESTBED_OUT/h1_qwen3_0p6b_q4_0/c200505b35 [--cell ...] \\
        --json results/m3/which_tasks_fail_qwen3-0.6b.json --md docs/figs/which_tasks_fail_qwen3-0.6b.md \\
        --fig docs/figs/which_tasks_fail_qwen3-0.6b.png
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testbed.classifier import FAILURES, classify_step  # noqa: E402
from testbed.spec import StepRow  # noqa: E402

SEEDS = (1, 2, 3, 4)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # dataviz reference slots 1-4, validated (light)
BENCH_LABEL = {"bfcl_multi_turn_base": "BFCL base", "bfcl_multi_turn_miss_func": "BFCL miss_func",
               "bfcl_multi_turn_miss_param": "BFCL miss_param", "tau2_verified_airline": "τ² airline",
               "tau2_verified_retail": "τ² retail", "tau2_verified_telecom": "τ² telecom"}
FIELDS = set(StepRow.__dataclass_fields__)


def load(run_dir: str) -> tuple[dict, dict]:
    """(episodes by (benchmark, task, seed), steps by the same key) for finished episodes."""
    specs = {}
    with open(os.path.join(run_dir, "specs.jsonl")) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                specs[d["run_id"]] = d
    eps, steps = {}, {}
    for p in glob.glob(os.path.join(run_dir, "episodes", "*.json")):
        e = json.load(open(p))
        sp = specs.get(e["run_id"])
        if not sp or e.get("seed") not in SEEDS:
            continue
        key = (sp["benchmark"], str(e["task_id"]), int(e["seed"]))
        eps[key] = e
        sp_path = os.path.join(run_dir, "steps", e["run_id"] + ".jsonl")
        rows = [json.loads(l) for l in open(sp_path) if l.strip()] if os.path.exists(sp_path) else []
        labs = []
        for r in rows:
            r.setdefault("benchmark", sp["benchmark"])
            labs.append(classify_step(StepRow(**{k: r.get(k) for k in FIELDS if k in r})))
        steps[key] = labs
    return eps, steps


def api_classes() -> dict[str, list[str]]:
    from testbed import adapter
    out = {}
    for cat in ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param"):
        for e in adapter._jsonl(os.path.join(adapter.bfcl_data_dir(), f"BFCL_v4_{cat}.json")):
            out[e["id"]] = list(e["involved_classes"])
    return out


def seed_stats(vals_by_seed: dict[int, list[float]]) -> dict:
    per = [float(np.mean(v)) for s, v in sorted(vals_by_seed.items()) if v]
    return {"mean": float(np.mean(per)) if per else None, "sd": float(np.std(per, ddof=1)) if len(per) > 1 else None,
            "per_seed": per, "n": int(sum(len(v) for v in vals_by_seed.values()))}


def summarise(eps: dict, steps: dict, keys: set, groups: dict) -> dict:
    """Success rate and failure rate per step, per group, mean ± SD over seeds."""
    out = {}
    for gname, gkeys in groups.items():
        ks = [k for k in keys if k in gkeys]
        if not ks:
            continue
        succ = collections.defaultdict(list)
        anyf = collections.defaultdict(list)
        for k in ks:
            succ[k[2]].append(float(bool(eps[k].get("valid"))))
            labs = steps.get(k, [])
            if labs:
                anyf[k[2]].append(sum(l in FAILURES for l in labs) / len(labs))
        out[gname] = {"success": seed_stats(succ), "failure_per_step": seed_stats(anyf), "episodes": len(ks)}
    return out


def by_class(steps: dict, keys: set) -> dict:
    """Failures per 100 steps, by class, mean ± SD over seeds."""
    classes = sorted({l for k in keys for l in steps.get(k, []) if l in FAILURES})
    out = {}
    for c in classes:
        v = collections.defaultdict(list)
        for k in keys:
            labs = steps.get(k, [])
            if labs:
                v[k[2]].append(100.0 * sum(l == c for l in labs) / len(labs))
        out[c] = seed_stats(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True, help="LABEL=run_dir of the BF16 reference")
    ap.add_argument("--cell", action="append", default=[], help="LABEL=run_dir of a 4-bit cell (repeatable)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--fig")
    a = ap.parse_args()
    ref_label, ref_dir = a.ref.split("=", 1)
    ref_eps, ref_steps = load(ref_dir)
    api = api_classes()
    # groups: benchmark/domain, and BFCL API class (a two-class task counts once in each)
    groups: dict[str, set] = collections.defaultdict(set)
    for k in ref_eps:
        groups["bench:" + k[0]].add(k)
        if k[0].startswith("bfcl"):
            for c in api.get(k[1], []):
                groups["api:" + c].add(k)
        groups["all"].add(k)
    res = {"model": a.model, "what": "Episode success and step failures by benchmark/domain, BFCL API class and "
           "failure class; sampled seeds 1-4, mean ± SD over seeds; development data, preview",
           "reference": {"label": ref_label, "dir": ref_dir}, "configs": {}}
    res["configs"][ref_label] = {"dir": ref_dir, "vs": None, "groups": summarise(ref_eps, ref_steps, set(ref_eps), groups),
                                 "failure_classes": by_class(ref_steps, set(ref_eps))}
    for c in a.cell:
        lab, d = c.split("=", 1)
        eps, st = load(d)
        both = set(eps) & set(ref_eps)
        res["configs"][lab] = {"dir": d, "pairs_with_reference": len(both),
                               "groups": summarise(eps, st, both, groups),
                               "failure_classes": by_class(st, both),
                               "reference_on_same_pairs": {"groups": summarise(ref_eps, ref_steps, both, groups),
                                                           "failure_classes": by_class(ref_steps, both)}}
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    if a.md:
        write_md(res, a.md)
    if a.fig:
        write_fig(res, a.fig)
    print("wrote", a.json, a.md or "", a.fig or "")


def _fmt(s, pct=True):
    if not s or s["mean"] is None:
        return "–"
    m = 100 * s["mean"] if pct else s["mean"]
    sd = (100 * s["sd"] if pct else s["sd"]) if s["sd"] is not None else None
    return f"{m:.1f} ± {sd:.1f}" if sd is not None else f"{m:.1f}"


def write_md(res: dict, path: str) -> None:
    labs = list(res["configs"])
    rows = []
    gnames = ["all"] + [g for g in res["configs"][labs[0]]["groups"] if g.startswith("bench:")] + \
        sorted(g for g in res["configs"][labs[0]]["groups"] if g.startswith("api:"))
    head = "| Group | " + " | ".join(f"{l}: success % / failures per 100 steps" for l in labs) + " |"
    rows.append(head)
    rows.append("|" + "---|" * (len(labs) + 1))
    for g in gnames:
        name = "All tasks" if g == "all" else BENCH_LABEL.get(g[6:], g[6:]) if g.startswith("bench:") else "API " + g[4:]
        cells = []
        for l in labs:
            gg = res["configs"][l]["groups"].get(g)
            cells.append("–" if not gg else f"{_fmt(gg['success'])} / {_fmt(gg['failure_per_step'])} (n={gg['episodes']})")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("| Failure class (per 100 steps) | " + " | ".join(labs) + " |")
    rows.append("|" + "---|" * (len(labs) + 1))
    classes = sorted({c for l in labs for c in res["configs"][l]["failure_classes"]})
    for c in classes:
        rows.append(f"| `{c}` | " + " | ".join(_fmt(res["configs"][l]["failure_classes"].get(c), pct=False) for l in labs) + " |")
    pairs = {l: res["configs"][l].get("pairs_with_reference") for l in labs[1:]}
    note = (f"\nSampled trials (seeds 1–4); mean ± 1 SD over seeds; n = episodes. {res['reference']['label']} is shown "
            f"on all its episodes; each 4-bit cell only on (task, seed) pairs it shares with it {pairs} (the JSON also "
            f"has {res['reference']['label']} restricted to each cell's pairs). Development data, preview.\n")
    with open(path, "w") as f:
        f.write(f"# Which tasks fail: {res['model']}\n\n" + "\n".join(rows) + "\n" + note)


def write_fig(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    labs = list(res["configs"])[:4]
    gnames = [g for g in res["configs"][labs[0]]["groups"] if g.startswith("bench:")]
    gnames.sort(key=lambda g: list(BENCH_LABEL).index(g[6:]) if g[6:] in BENCH_LABEL else 99)
    classes = sorted({c for l in labs for c in res["configs"][l]["failure_classes"]},
                     key=lambda c: -(res["configs"][labs[0]]["failure_classes"].get(c, {}).get("mean") or 0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200, facecolor="white", gridspec_kw={"width_ratios": [1, 1.4]})
    w = 0.8 / len(labs)
    for ax, names, getter, ylabel, title in (
            (axes[0], gnames, lambda l, g: res["configs"][l]["groups"].get(g, {}).get("success"),
             "Episode success (%)", "Success by benchmark and domain"),
            (axes[1], classes, lambda l, c: res["configs"][l]["failure_classes"].get(c),
             "Failures per 100 steps", "Step failures by class (all tasks)")):
        x = np.arange(len(names))
        for i, l in enumerate(labs):
            vals, errs = [], []
            for n in names:
                s = getter(l, n)
                scale = 100 if ax is axes[0] else 1
                vals.append((s["mean"] or 0) * scale if s and s["mean"] is not None else 0)
                errs.append((s["sd"] or 0) * scale if s and s["sd"] is not None else 0)
            ax.bar(x + (i - (len(labs) - 1) / 2) * w, vals, w * 0.92, yerr=errs, color=COLORS[i], label=l,
                   error_kw={"elinewidth": 1, "capsize": 2, "ecolor": ink2}, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([BENCH_LABEL.get(n[6:], n) if n.startswith("bench:") else n for n in names],
                           rotation=30, ha="right", color=ink)
        ax.set_ylabel(ylabel, color=ink)
        ax.set_title(title, color=ink, loc="left", fontsize=11)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(0.01, 0.01, f"{res['model']}: sampled trials (seeds 1-4), bars = mean over seeds, whiskers = ± 1 SD. "
             "4-bit cells on the (task, seed) pairs they share with BF16. Development data, preview.",
             fontsize=8, color=ink2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white")


if __name__ == "__main__":
    main()
