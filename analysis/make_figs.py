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


def nonlive_fig(nl, out):
    cats = [("simple_python", "Simple\nPython"), ("simple_java", "Simple\nJava"), ("simple_javascript", "Simple\nJS"),
            ("multiple", "Multiple"), ("parallel", "Parallel"), ("parallel_multiple", "Parallel\nmultiple")]
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(cats) - 0.4)
    w = bar_width_data(ax, fig) * 0.85
    gap = w * 0.08
    for j, f in enumerate(FORMATS):
        d = nl[nl.format == f]
        for i, (c, _) in enumerate(cats):
            acc = 100 * d[d.category == c].correct.mean()
            ax.bar(i + (j - 1) * (w + gap), acc, width=w, color=FORMAT_COLOR[f], zorder=2,
                   label=FORMAT_LABEL[f] if i == 0 else None)
    ax.set_xticks(range(len(cats)), [f"{lab}\nn = {int((nl[(nl.format == 'BF16') & (nl.category == c)]).shape[0])}"
                                      for c, lab in cats], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("BFCL non-live accuracy (%)")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.27)
    footnote(fig, "bfcl-eval 2026.3.23 · Non-Live Overall: BF16 69.4%, Q4_K_M 67.4%, Q4_0 (--pure) 56.0%")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def energy_fig(nv, out):
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(FORMATS) - 0.4)
    w = bar_width_data(ax, fig) * 1.6
    top = 0
    for i, f in enumerate(FORMATS):
        d = nv[nv.format == f].mj_per_tok_counter
        m, sd = d.mean(), d.std()
        top = max(top, m + sd)
        ax.bar(i, m, width=w, color=FORMAT_COLOR[f], zorder=2)
        ax.plot([i, i], [m - sd, m + sd], color=INK_2, linewidth=1.2, zorder=3)
        ax.text(i, m + sd + top * 0.02, f"{m:.0f} mJ", ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
    ax.set_xticks(range(len(FORMATS)), [f"{FORMAT_LABEL[f]}\nn = {int((nv.format == f).sum())} runs" for f in FORMATS])
    ax.set_ylim(0, top * 1.2)
    ax.set_ylabel("GPU energy per decoded token (mJ)")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.23)
    footnote(fig, "NVML energy counter, idle power subtracted · 1,024-token greedy decode · whiskers: \u00b11 sd over 10 runs")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def v2_accuracy_fig(t, nl_summary, out, sd=None):
    """Greedy, grammar none: simple_python and non-live overall per format; optional sampled sd whiskers."""
    groups = [("simple_python", "BFCL simple_python\nn = 400"), ("nonlive", "BFCL non-live overall\nn = 1,390")]
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(groups) - 0.4)
    w = bar_width_data(ax, fig) * 1.1
    gap = w * 0.35
    for j, f in enumerate(FORMATS):
        for i, (g, _) in enumerate(groups):
            if g == "simple_python":
                d = t[(t.run == f"L1_{f}_nonlive_none") & (t.category == "simple_python")]
                v = 100 * d.correct.mean()
            else:
                v = float(nl_summary[f])
            x = i + (j - 1) * (w + gap)
            ax.bar(x, v, width=w, color=FORMAT_COLOR[f], zorder=2, label=FORMAT_LABEL[f] if i == 0 else None)
            if sd is not None and (f, g) in sd:
                ax.plot([x, x], [v - sd[(f, g)], v + sd[(f, g)]], color=INK_2, linewidth=1.2, zorder=3)
            ax.text(x, v + 1.5, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(groups)), [lab for _, lab in groups])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.23)
    note = "bfcl-eval 2026.3.23 · non-thinking, greedy, prompt cache off"
    footnote(fig, note + (" · whiskers: sd of 4 sampled runs (model-card settings)" if sd else ""))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def v2_failure_fig(t, out):
    classes = [("no_call", "No tool call\n(prose answer)"), ("tag_spacing", "Tag spacing\n(parser rejects)"),
               ("malformed", "Malformed\ncall"), ("hallucinated_tool", "Hallucinated\ntool"),
               ("schema_invalid", "Schema-invalid\nargs"), ("wrong_value_or_count", "Valid schema,\nwrong value/count")]
    fig, ax = base_axes()
    ax.set_xlim(-0.6, len(classes) - 0.4)
    w = bar_width_data(ax, fig) * 0.9
    gap = w * 0.08
    top = 0
    for j, f in enumerate(FORMATS):
        d = t[(t.run == f"L1_{f}_nonlive_none") & (t.category == "simple_python")]
        for i, (c, _) in enumerate(classes):
            v = int((d.failure_class == c).sum())
            top = max(top, v)
            x = i + (j - 1) * (w + gap)
            ax.bar(x, v, width=w, color=FORMAT_COLOR[f], zorder=2, label=FORMAT_LABEL[f] if i == 0 else None)
            ax.text(x, v + 0.8, str(v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(classes)), [lab for _, lab in classes], fontsize=10.5)
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel("Failed tasks (of 400 per format)")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.25)
    footnote(fig, "BFCL simple_python · non-thinking, greedy, prompt cache off · classes exclusive, checked left to right")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def v2_grammar_fig(t, out):
    """Two panels: simple_python (a call is always right) and irrelevance (no call is right)."""
    conds = [("none", "No grammar", "#8c8b85"), ("free", "R1 grammar (calls or free text)", "#4a3aa7"),
             ("forced", "Forced-call grammar (ablation)", "#e87ba4")]
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=200, facecolor="white", sharey=True)
    for ax, (cat, title) in zip(axes, [("simple_python", "simple_python (n = 400)"), ("irrelevance", "irrelevance (n = 240)")]):
        ax.set_facecolor("white")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(FORMATS) - 0.4)
        w = 0.22
        for j, (c, lab, col) in enumerate(conds):
            for i, f in enumerate(FORMATS):
                run = f"L1_{f}_forced" if c == "forced" else f"L1_{f}_nonlive_{c}"
                d = t[(t.run == run) & (t.category == cat)]
                if d.empty:
                    continue
                v = 100 * d.correct.mean()
                x = i + (j - 1) * (w + 0.02)
                ax.bar(x, v, width=w, color=col, zorder=2, label=lab if (i == 0 and cat == "simple_python") else None)
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(range(len(FORMATS)), [FORMAT_LABEL[f].replace(" (--pure)", "\n(--pure)") for f in FORMATS], fontsize=10.5)
        ax.text(0.5, 1.01, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, color=INK_2)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%)")
    fig.legend(frameon=False, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.21, wspace=0.08)
    footnote(fig, "bfcl-eval 2026.3.23 · non-thinking, greedy, prompt cache off · panel labels are task categories")
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
    npath = os.path.join(args.progress, "bfcl_nonlive_pilot.csv")
    if os.path.exists(npath):
        nonlive_fig(pd.read_csv(npath), os.path.join(figs, "accuracy_nonlive_by_category.png"))
    epath = os.path.join(args.progress, "nvml_summary.csv")
    if os.path.exists(epath):
        energy_fig(pd.read_csv(epath), os.path.join(figs, "energy_per_token.png"))
    v2path = os.path.join(args.progress, "bfcl_v2_pilot.csv")
    if os.path.exists(v2path):
        t = pd.read_csv(v2path)
        keys = dict(l.split(": ", 1) for l in open(os.path.join(args.progress, "RESULTS.md")).read().split("## Slide numbers", 1)[1].strip().splitlines() if ": " in l)
        nl = {f: keys.get(f"nonlive_acc_{k}_v2", "nan") for f, k in [("BF16", "bf16"), ("Q4_K_M", "q4km"), ("Q4_0", "q40")]}
        sd = {}
        for f, k in [("BF16", "bf16"), ("Q4_K_M", "q4km"), ("Q4_0", "q40")]:
            for g, kk in [("simple_python", f"sp_acc_{k}_sd_v2"), ("nonlive", f"nonlive_acc_{k}_sd_v2")]:
                try:
                    sd[(f, g)] = float(keys[kk])
                except (KeyError, ValueError):
                    pass
        if all(v.replace(".", "", 1).isdigit() for v in nl.values()):
            v2_accuracy_fig(t, nl, os.path.join(figs, "v2_accuracy.png"), None)
            v2_failure_fig(t, os.path.join(figs, "v2_failure_breakdown.png"))
            v2_grammar_fig(t, os.path.join(figs, "v2_grammar_effect.png"))
    gpath = os.path.join(args.progress, "grammar_pilot.csv")
    if os.path.exists(gpath):
        g = pd.read_csv(gpath)
        grammar_fig(g[g.temperature == 0], os.path.join(figs, "grammar_effect.png"))


if __name__ == "__main__":
    main()
