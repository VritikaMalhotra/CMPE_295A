#!/usr/bin/env python
"""Step-level damage table (H1 preview): for each failure class, the rate per 100 steps under BF16 and under each
4-bit cell, the amplification ratio (4-bit rate / BF16 rate) and its 95% CI.

Pairing and dispersion follow pre-registration §6: only (task, seed) pairs that both configurations finished,
sampled seeds 1-4 (the greedy pass is separate); the CI is a paired cluster bootstrap over tasks (all trials of a
task resampled together, 10,000 resamples), on the ratio of pooled rates. Steps are labelled with the current
classifier (codebook v0). **Development data, preview**: the pre-registration is not filed, and these runs are
not confirmatory.

    python analysis/damage_table.py --model qwen3-0.6b --ref BF16=<dir> --cell Q4_0-pure=<dir> [--cell ...] \\
        --json results/m3/damage_preview_qwen3-0.6b.json --md docs/m3_damage_preview.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.which_tasks_fail import load  # noqa: E402
from testbed.classifier import FAILURES  # noqa: E402

N_BOOT = 10_000


def counts(steps: dict, keys: list, cls: str | None) -> tuple[np.ndarray, np.ndarray, list]:
    """Per task: failures of class ``cls`` (None = any failure) and steps, summed over the task's paired trials."""
    by_task = collections.defaultdict(lambda: [0, 0])
    for k in keys:
        labs = steps.get(k, [])
        t = (k[0], k[1])
        by_task[t][0] += sum((l == cls) if cls else (l in FAILURES) for l in labs)
        by_task[t][1] += len(labs)
    tasks = sorted(by_task)
    return (np.array([by_task[t][0] for t in tasks], float), np.array([by_task[t][1] for t in tasks], float), tasks)


def ratio_ci(f_ref, n_ref, f_q, n_q, rng) -> dict:
    r_ref, r_q = f_ref.sum() / n_ref.sum(), f_q.sum() / n_q.sum()
    out = {"rate_ref_per100": 100 * r_ref, "rate_4bit_per100": 100 * r_q, "events_ref": int(f_ref.sum()),
           "events_4bit": int(f_q.sum()), "steps_ref": int(n_ref.sum()), "steps_4bit": int(n_q.sum()),
           "ratio": (r_q / r_ref) if r_ref > 0 else None, "diff_per100": 100 * (r_q - r_ref)}
    idx = rng.integers(0, len(f_ref), size=(N_BOOT, len(f_ref)))
    br = f_ref[idx].sum(1) / n_ref[idx].sum(1)
    bq = f_q[idx].sum(1) / n_q[idx].sum(1)
    ok = br > 0
    if r_ref > 0 and ok.mean() > 0.99:
        lo, hi = np.quantile(bq[ok] / br[ok], [0.025, 0.975])
        out["ratio_ci95"] = [float(lo), float(hi)]
    else:
        out["ratio_ci95"] = None            # too few reference events for a ratio
    dlo, dhi = np.quantile(100 * (bq - br), [0.025, 0.975])
    out["diff_ci95_per100"] = [float(dlo), float(dhi)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--seed", type=int, default=295)
    a = ap.parse_args()
    rl, rd = a.ref.split("=", 1)
    ref_eps, ref_steps = load(rd)
    res = {"model": a.model, "label": "development data, preview", "reference": {"label": rl, "dir": rd},
           "method": "paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled "
                     "rates; classifier codebook v0", "cells": {}}
    for c in a.cell:
        cl, cd = c.split("=", 1)
        eps, steps = load(cd)
        keys = sorted(set(eps) & set(ref_eps))
        rng = np.random.default_rng(a.seed)
        classes = sorted({l for k in keys for l in ref_steps.get(k, []) + steps.get(k, []) if l in FAILURES})
        rows = {}
        for cls in [None] + classes:
            fr, nr, tasks = counts(ref_steps, keys, cls)
            fq, nq, tasks_q = counts(steps, keys, cls)
            assert tasks == tasks_q
            rows["any_failure" if cls is None else cls] = ratio_ci(fr, nr, fq, nq, rng)
        bench = collections.Counter(k[0] for k in keys)
        res["cells"][cl] = {"dir": cd, "pairs": len(keys), "tasks": len({(k[0], k[1]) for k in keys}),
                            "pairs_by_benchmark": dict(bench), "classes": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    if a.md:
        write_md(res, a.md, a.json)
    print("wrote", a.json, a.md or "")


def _r(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def write_md(res: dict, path: str, json_path: str) -> None:
    lines = [f"# H1 damage preview: {res['model']} (development data, preview)\n",
             "**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may "
             "still be running (pair counts below). Method: " + res["method"] + f". JSON: `{json_path}`.\n"]
    for cl, c in res["cells"].items():
        lines.append(f"## {cl} vs {res['reference']['label']}: {c['pairs']} paired episodes, {c['tasks']} tasks\n")
        lines.append(f"Pairs by benchmark: {c['pairs_by_benchmark']}\n")
        lines.append("| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | "
                     "Difference per 100 steps [95% CI] | Events BF16 / 4-bit |")
        lines.append("|---|---|---|---|---|---|---|")
        order = ["any_failure"] + sorted((k for k in c["classes"] if k != "any_failure"),
                                         key=lambda k: -c["classes"][k]["rate_4bit_per100"])
        for k in order:
            r = c["classes"][k]
            ci = "–" if not r["ratio_ci95"] else f"[{r['ratio_ci95'][0]:.2f}, {r['ratio_ci95'][1]:.2f}]"
            lines.append(f"| {'**any failure**' if k == 'any_failure' else '`' + k + '`'} | {_r(r['rate_ref_per100'])} | "
                         f"{_r(r['rate_4bit_per100'])} | {_r(r['ratio'])} | {ci} | {_r(r['diff_per100'])} "
                         f"[{_r(r['diff_ci95_per100'][0])}, {_r(r['diff_ci95_per100'][1])}] | "
                         f"{r['events_ref']} / {r['events_4bit']} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
