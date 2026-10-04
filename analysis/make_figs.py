"""Slide figures for the 2026-10-08 pilot (16:9, 1600x900 px at 200 dpi, white background).

Reads the pilot CSVs from the progress directory and writes PNGs to progress/figs/.
Pilot / development data, not study results.

    python make_figs.py --progress ../..
"""
import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FORMATS = ["BF16", "Q4_K_M", "Q4_0"]
FORMAT_LABEL = {"BF16": "BF16", "Q4_K_M": "Q4_K_M", "Q4_0": "Q4_0 (--pure)"}
# fixed categorical order (validated: adjacent CVD dE >= 9.2, values labeled on every bar)
FORMAT_COLOR = {"BF16": "#2a78d6", "Q4_K_M": "#eb6834", "Q4_0": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FOOT = "Pilot data (not study results), Colab A100, 2026-10-08"
BAR_W = 0.24  # inches, about 48 px at 200 dpi


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def base_axes():
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor="white")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def footnote(fig, extra=""):
    text = FOOT + ("\n" + extra if extra else "")
    fig.text(0.01, 0.012, text, fontsize=8.5, color=INK_2, ha="left", va="bottom", linespacing=1.3)


def bar_width_data(ax, fig):
    # convert a fixed physical bar width (inches) into data units on the x axis
    x0, x1 = ax.get_xlim()
    return BAR_W * (x1 - x0) / (ax.get_position().width * fig.get_figwidth())


def accuracy_fig(df, out):
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(FORMATS) - 0.4)
    w = bar_width_data(ax, fig) * 1.6
    for i, f in enumerate(FORMATS):
        d = df[df.format == f]
        k, n = int(d.correct.sum()), len(d)
        acc = 100 * k / n
        lo, hi = wilson(k, n)
        ax.bar(i, acc, width=w, color=FORMAT_COLOR[f], zorder=2)
        ax.plot([i, i], [100 * lo, 100 * hi], color=INK_2, linewidth=1.2, zorder=3)
        ax.text(i, 100 * hi + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
    ax.set_xticks(range(len(FORMATS)), [f"{FORMAT_LABEL[f]}\nn = {len(df[df.format == f])}" for f in FORMATS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("BFCL simple_python accuracy (%)")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.23)
    footnote(fig, "bfcl-eval 2026.3.23 AST checker · Qwen3-0.6B, non-thinking, greedy · whiskers: Wilson 95% CI")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def failure_fig(df, out):
    classes = [("no_call", "No tool call\n(prose answer)"), ("malformed", "Malformed\ncall"),
               ("hallucinated_tool", "Hallucinated\ntool"), ("schema_invalid", "Schema-invalid\nargs"),
               ("wrong_value_or_count", "Valid schema,\nwrong value/count")]
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(classes) - 0.4)
    w = bar_width_data(ax, fig)
    gap = w * 0.08
    top = 0
    for j, f in enumerate(FORMATS):
        d = df[df.format == f]
        for i, (c, _) in enumerate(classes):
            v = int((d.failure_class == c).sum())
            top = max(top, v)
            x = i + (j - 1) * (w + gap)
            ax.bar(x, v, width=w, color=FORMAT_COLOR[f], zorder=2, label=FORMAT_LABEL[f] if i == 0 else None)
            ax.text(x, v + 0.8, str(v), ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_xticks(range(len(classes)), [lab for _, lab in classes])
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel("Failed tasks (of 400 per format)")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.25)
    footnote(fig, "BFCL simple_python · classes are exclusive, checked in this order left to right")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def grammar_fig(g, out):
    # g: rows with format, condition (none/gbnf), failure_class, correct; temperature 0 runs only
    groups = [("Q4_K_M", "Q4_K_M"), ("Q4_0", "Q4_0 (--pure)"), ("BF16", "BF16")]
    conds = [("none", "Without grammar", "#8c8b85"), ("gbnf", "With GBNF grammar", "#4a3aa7")]
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(groups) - 0.4)
    w = bar_width_data(ax, fig) * 1.3
    gap = w * 0.08
    top = 0
    for j, (cond, lab, col) in enumerate(conds):
        for i, (f, _) in enumerate(groups):
            d = g[(g.format == f) & (g.condition == cond)]
            if d.empty:
                continue
            bad = int(d.failure_class.isin(["no_call", "malformed", "hallucinated_tool", "schema_invalid"]).sum())
            top = max(top, bad)
            x = i + (j - 0.5) * (w + gap)
            ax.bar(x, bad, width=w, color=col, zorder=2, label=lab if i == 0 else None)
            ax.text(x, bad + 0.8, str(bad), ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(len(groups)), [f"{lab}\nn = 400" for _, lab in groups])
    ax.set_ylim(0, max(top, 1) * 1.3)
    ax.set_ylabel("Tasks with a format failure (of 400)")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.23)
    footnote(fig, "BFCL simple_python, raw /completion, greedy · format failure = no call, malformed call, unknown tool, or schema-invalid args")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    args = ap.parse_args()
    figs = os.path.join(args.progress, "figs")
    os.makedirs(figs, exist_ok=True)
    df = pd.read_csv(os.path.join(args.progress, "bfcl_pilot.csv"))
    df = df[df.category == "simple_python"]
    accuracy_fig(df, os.path.join(figs, "accuracy_bf16_vs_q4.png"))
    failure_fig(df, os.path.join(figs, "failure_breakdown.png"))
    gpath = os.path.join(args.progress, "grammar_pilot.csv")
    if os.path.exists(gpath):
        g = pd.read_csv(gpath)
        grammar_fig(g[g.temperature == 0], os.path.join(figs, "grammar_effect.png"))


if __name__ == "__main__":
    main()
