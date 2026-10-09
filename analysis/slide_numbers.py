"""Compute the "## Slide numbers" block for RESULTS.md from the pilot CSVs (no hand-copied values).

    python slide_numbers.py --progress ../.. --end "19:55"
"""
import argparse
import json
import os
from math import comb

import pandas as pd

FMT_FAIL = ["no_call", "malformed", "hallucinated_tool", "schema_invalid"]
KEY = {"BF16": "bf16", "Q4_K_M": "q4km", "Q4_0": "q40"}


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--end", default=None, help="end of run window, HH:MM PT")
    a = ap.parse_args()
    P = a.progress
    out = []

    p2 = pd.read_csv(os.path.join(P, "bfcl_pilot.csv"))
    p2 = p2[p2.category == "simple_python"]
    out.append(f"n_tasks: {p2[p2.format == 'BF16'].shape[0]}")
    for f, k in KEY.items():
        out.append(f"{k}_acc: {fmt(100 * p2[p2.format == f].correct.mean())}")

    g = pd.read_csv(os.path.join(P, "grammar_pilot.csv"))
    g0 = g[g.temperature == 0]
    for cond, suffix in [("none", "nogram"), ("gbnf", "grammar")]:
        for f, k in KEY.items():
            d = g0[(g0.format == f) & (g0.condition == cond)]
            out.append(f"{k}_acc_{suffix}: {fmt(100 * d.correct.mean())}")
    for f, k in KEY.items():
        for cond, suffix in [("none", "without"), ("gbnf", "with")]:
            d = g0[(g0.format == f) & (g0.condition == cond)]
            out.append(f"fmt_fail_{k}_{suffix}: {int(d.failure_class.isin(FMT_FAIL).sum())}")
    for f, k in KEY.items():
        out.append(f"nocall_{k}: {int((p2[p2.format == f].failure_class == 'no_call').sum())}")

    nl = pd.read_csv(os.path.join(P, "bfcl_nonlive_summary.csv"))
    nld = pd.read_csv(os.path.join(P, "bfcl_nonlive_pilot.csv"))
    out.append(f"nonlive_n: {(nld.format == 'BF16').sum()}")
    for f, k in KEY.items():
        v = nl[nl.run == f"{f}_nonlive"]["Non-Live Overall Acc"].iloc[0]
        out.append(f"nonlive_acc_{k}: {v.rstrip('%')}")

    w = p2.pivot(index="task_id", columns="format", values="correct")
    for f, k in [("Q4_K_M", "q4km"), ("Q4_0", "q40")]:
        b = int((w.BF16 & ~w[f]).sum())
        c = int((~w.BF16 & w[f]).sum())
        out.append(f"mcnemar_p_{k}: {mcnemar_exact(b, c):.3g}")

    q = pd.read_csv(os.path.join(P, "quant_files.csv")).set_index("format")
    for f, k in KEY.items():
        out.append(f"bpw_{k}: {fmt(q.loc[f, 'bpw_whole_file'])}")

    p1 = json.load(open(os.path.join(P, "colab_raw", "p1", "p1_toolcall.json")))["formats"]
    nv_path = os.path.join(P, "nvml_summary.csv")
    nv = pd.read_csv(nv_path) if os.path.exists(nv_path) else None
    for f, k in KEY.items():
        if nv is not None:
            out.append(f"decode_tps_{k}: {fmt(nv[nv.format == f].decode_tok_s.mean(), 1)}")
        else:
            s = sorted(p1[f]["decode_tps"])
            out.append(f"decode_tps_{k}: {fmt(s[len(s) // 2], 1)}")
    for f, k in KEY.items():
        out.append(f"peak_mem_mib_{k}: {p1[f]['gpu_mem_peak_mib']}")

    env = json.load(open(os.path.join(P, "env.json")))
    out.append(f"llama_cpp_tag: {env['llama_cpp_tag']}")
    out.append(f"gpu: {env['gpu'].split(',')[0].strip()}")
    out.append(f"run_window: 2026-10-08 18:54-{a.end or 'ongoing'} PT")

    for f, k in KEY.items():
        if nv is not None:
            out.append(f"energy_per_token_mj_{k}: {fmt(nv[nv.format == f].mj_per_tok_counter.mean(), 3)}")
        else:
            out.append(f"energy_per_token_mj_{k}: n/a: P4 (NVML) not finished")

    s7 = pd.read_csv(os.path.join(P, "s7_loss_compare.csv"))
    out.append(f"s7_resume_identical: {'yes' if (s7.abs_diff == 0).all() and len(s7) == 50 else 'no'}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
