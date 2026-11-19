#!/usr/bin/env python
"""Damage map v1 (M3, Dec 4): one view of 4-bit damage across all six models, from the per-model previews.

Reads ``results/m3/damage_preview_<model>.json`` (analysis/damage_table.py: step-level amplification ratio of
pooled failure rates, paired (task, seed), seeds 1-4, cluster bootstrap over tasks) and
``results/m3/which_tasks_fail_<model>.json`` (analysis/which_tasks_fail.py: episode success, mean ± 1 SD over the
4 seeds, on the pairs every configuration finished). Writes a JSON summary, a Markdown table and a two-panel
figure:

* A - amplification ratio of the any-failure step rate (4-bit / BF16), model x 4-bit format, on a diverging
  log scale centred on 1 (blue: fewer failures than BF16, red: more), with the 95% CI in each cell;
* B - episode success rate per model and configuration, bars with ± 1 SD over seeds (advisor rule: every bar
  graph shows its SD).

Development data, preview: the pre-registration is not filed, and none of this is confirmatory.

    python analysis/damage_map.py --out-json results/m3/damage_map_v1.json --md docs/m3_damage_map_v1.md \\
        --fig docs/figs/damage_map_v1.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

MODELS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "smollm3-3b", "lfm2.5-1.2b", "lfm2.5-2.6b"]
LABEL = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B",
         "smollm3-3b": "SmolLM3-3B", "lfm2.5-1.2b": "LFM2.5-1.2B", "lfm2.5-2.6b": "LFM2.5-2.6B"}
CELLS = ["Q4_0-pure", "Q4_K_M", "IQ4_XS"]
CONFIGS = ["BF16"] + CELLS
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # dataviz categorical slots 1-4, validated (light)
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]          # dataviz diverging pair: blue <-> gray <-> red
INK, MUTED, SURFACE = "#1a1a19", "#6b6a66", "#fcfcfb"
SIG_CLASSES = 3                                        # per cell, the classes listed in the Markdown table


def _f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def collect(results_dir: str) -> dict:
    out = {}
    for m in MODELS:
        dp = os.path.join(results_dir, f"damage_preview_{m}.json")
        wt = os.path.join(results_dir, f"which_tasks_fail_{m}.json")
        if not (os.path.exists(dp) and os.path.exists(wt)):
            print(f"skip {m}: missing {dp if not os.path.exists(dp) else wt}")
            continue
        d, w = json.load(open(dp)), json.load(open(wt))
        row = {"reference_dir": d["reference"]["dir"], "cells": {}, "success": {}}
        for c in CONFIGS:
            g = w["configs"].get(c, {}).get("groups", {})
            row["success"][c] = {k: g[k]["success"] for k in g if k == "all" or k.startswith("bench:")}
        for c in CELLS:
            cell = d["cells"].get(c)
            if not cell:
                continue
            classes = cell["classes"]
            anyf = classes["any_failure"]
            sig = sorted(((k, v) for k, v in classes.items() if k != "any_failure" and v.get("ratio_ci95")
                          and (v["ratio_ci95"][0] > 1 or v["ratio_ci95"][1] < 1)),
                         key=lambda kv: -abs(kv[1]["diff_per100"]))
            row["cells"][c] = {"pairs": cell["pairs"], "tasks": cell["tasks"], "any_failure": anyf,
                               "ci_excludes_1": [k + (" ↑" if v["ratio"] > 1 else " ↓") for k, v in sig]}
        out[m] = row
    return out


def write_md(res: dict, path: str, fig: str | None) -> None:
    L = ["# Damage map v1: 4-bit damage across six models (development data, preview)", "",
         "**Development data, preview.** Not confirmatory: the pre-registration is not filed. H1 grid complete "
         "(18/18 cells, `results/m3/run_status.json`); exclusions follow the registered rule and decisions 15, 16 "
         "and 18. Method: paired (task, seed), seeds 1-4, cluster bootstrap over tasks (10,000 resamples), "
         "ratio of pooled per-step failure rates, classifier codebook v0; success is mean ± 1 SD over the 4 seeds. "
         "Per-model detail: `docs/damage_previews/`.", ""]
    if fig:
        L += [f"![Damage map v1]({os.path.relpath(fig, os.path.dirname(path))})", ""]
    L += ["## Any-failure amplification (4-bit / BF16 step failure rate)", "",
          "| Model | Format | Pairs | BF16 per 100 steps | 4-bit per 100 steps | Ratio [95% CI] | "
          "Difference per 100 [95% CI] | Classes with CI excluding 1 (↑ more / ↓ fewer than BF16, largest first) |",
          "|---|---|---|---|---|---|---|---|"]
    for m, row in res.items():
        for c, cell in row["cells"].items():
            a = cell["any_failure"]
            ci = a.get("ratio_ci95") or [None, None]
            dci = a["diff_ci95_per100"]
            L.append(f"| {LABEL[m]} | {c} | {cell['pairs']} | {_f(a['rate_ref_per100'])} | "
                     f"{_f(a['rate_4bit_per100'])} | {_f(a['ratio'])} [{_f(ci[0])}, {_f(ci[1])}] | "
                     f"{_f(a['diff_per100'])} [{_f(dci[0])}, {_f(dci[1])}] | "
                     f"{', '.join(f'`{k[:-2]}`{k[-2:]}' for k in cell['ci_excludes_1'][:SIG_CLASSES]) or '–'} |")
    L += ["", "## Episode success, % (mean ± 1 SD over seeds 1-4)", "",
          "| Model | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for m, row in res.items():
        cells = []
        for c in CONFIGS:
            s = row["success"].get(c, {}).get("all")
            cells.append("–" if not s else f"{100 * s['mean']:.1f} ± {100 * s['sd']:.1f}")
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(L) + "\n")


def plot(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    models = list(res)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 0.55 * len(models) + 2.4),
                                 gridspec_kw={"width_ratios": [1, 1.5]}, facecolor=SURFACE)
    # A: heatmap
    R = np.full((len(models), len(CELLS)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(CELLS):
            r = res[m]["cells"].get(c, {}).get("any_failure", {}).get("ratio")
            R[i, j] = np.nan if r is None else r
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    norm = LogNorm(vmin=0.5, vmax=2.0)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.clip(R, 0.5, 2.0), cmap=cmap, norm=norm, aspect="auto")
    for i, m in enumerate(models):
        for j, c in enumerate(CELLS):
            a = res[m]["cells"].get(c, {}).get("any_failure")
            if not a:
                continue
            ci = a.get("ratio_ci95") or [None, None]
            ax.text(j, i, f"{_f(a['ratio'])}\n[{_f(ci[0])}, {_f(ci[1])}]", ha="center", va="center",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(CELLS)), CELLS)
    ax.set_yticks(range(len(models)), [LABEL[m] for m in models])
    ax.set_xticks(np.arange(-.5, len(CELLS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(models)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A. Any-failure step rate, 4-bit / BF16\n(ratio [95% CI]; red = more failures)", loc="left",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[0.5, 0.75, 1, 1.5, 2])
    cb.ax.minorticks_off()
    cb.ax.set_yticks([0.5, 0.75, 1, 1.5, 2], ["≤0.5", "0.75", "1", "1.5", "≥2"])
    cb.outline.set_visible(False)
    # B: grouped bars with ± 1 SD
    bx.set_facecolor(SURFACE)
    w = 0.8 / len(CONFIGS)
    x = np.arange(len(models))
    for k, c in enumerate(CONFIGS):
        mu = [100 * (res[m]["success"].get(c, {}).get("all") or {"mean": np.nan})["mean"] for m in models]
        sd = [100 * (res[m]["success"].get(c, {}).get("all") or {"sd": np.nan})["sd"] for m in models]
        bx.bar(x + (k - 1.5) * w, mu, w * 0.9, yerr=sd, color=COLORS[k], label=c, edgecolor=SURFACE,
               linewidth=1, error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK})
    bx.set_xticks(x, [LABEL[m] for m in models])
    bx.set_ylabel("Success, % (mean ± 1 SD)")
    bx.grid(axis="y", color="#e4e3df", linewidth=0.6)
    bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    bx.set_title("B. Episode success by configuration\n", loc="left", fontsize=10)
    fig.text(0.01, 0.01, "Development data, preview. Paired (task, seed), seeds 1-4; BFCL multi-turn + τ² "
             "verified; cluster bootstrap over tasks.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/m3")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--fig")
    a = ap.parse_args()
    res = collect(a.results)
    json.dump({"what": "damage map v1; development data, preview", "models": res}, open(a.out_json, "w"), indent=1)
    if a.fig:
        plot(res, a.fig)
    if a.md:
        write_md(res, a.md, a.fig)
    print(f"damage map: {len(res)} models -> {a.out_json}")


if __name__ == "__main__":
    main()
