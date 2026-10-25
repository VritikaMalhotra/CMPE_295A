"""M1 report: one Markdown file per experiment directory, built only from the logged episodes and steps.

Sections: accuracy per configuration, failure-class mix (C6), paired comparisons (C10), costs (C9),
the planted-effect check (C10), and the manifest (C11). Everything is marked development data.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from .classifier import CLASSES
from .logger import to_parquet
from .profiler import step_costs
from .stats import compare, planted_effect_check

# (label, reference filter, other filter); the M1 questions from the spec
COMPARISONS = [
    ("Q4_K_M vs BF16 (CUDA)", {"variant": "bf16", "backend": "cuda", "repair": "none"},
     {"variant": "q4_k_m", "backend": "cuda", "repair": "none"}),
    ("Q4_0 vs BF16 (CUDA)", {"variant": "bf16", "backend": "cuda", "repair": "none"},
     {"variant": "q4_0", "backend": "cuda", "repair": "none"}),
    ("Q4_0: CPU vs CUDA", {"variant": "q4_0", "backend": "cuda", "repair": "none"},
     {"variant": "q4_0", "backend": "cpu", "repair": "none"}),
    ("R1 grammar vs Q4_0 (CUDA)", {"variant": "q4_0", "backend": "cuda", "repair": "none"},
     {"repair": "r1"}),
    ("R2 Q8_0 vs Q4_0 (CUDA)", {"variant": "q4_0", "backend": "cuda", "repair": "none"},
     {"repair": "r2"}),
    ("R3 QAD Q4_0 vs Q4_0 (CUDA)", {"variant": "q4_0", "backend": "cuda", "repair": "none"},
     {"repair": "r3"}),
]


SHORT = {"bfcl_multi_turn_base": "bfcl", "tau2_verified_airline": "tau2"}


def _cfg_label(row) -> str:
    b = SHORT.get(row.get("benchmark", ""), row.get("benchmark", "")) if hasattr(row, "get") else ""
    return (f"{b}:" if b else "") + f"{row['variant']}/{row['backend']}" + (
        "" if row["repair"] == "none" else f" +{row['repair']}")


def _benchmarks(run_dir: str) -> dict:
    """config_key -> benchmark, from specs.jsonl (episode records predate the benchmark column)."""
    p = os.path.join(run_dir, "specs.jsonl")
    out = {}
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            out[d["config_key"]] = d.get("benchmark", "")
    return out


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join("" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, float) else str(v))
                                  for v in r.values) + " |\n"
    return out


def write_report(run_dir: str) -> str:
    to_parquet(run_dir)                       # always rebuild from the per-episode files (source of truth)
    eps = pd.read_parquet(os.path.join(run_dir, "episodes.parquet"))
    steps = pd.read_parquet(os.path.join(run_dir, "steps.parquet")) if os.path.exists(
        os.path.join(run_dir, "steps.parquet")) else pd.DataFrame()
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"))) if os.path.exists(
        os.path.join(run_dir, "manifest.json")) else {}
    planned = {k: v["episodes"] for k, v in manifest.get("configs", {}).items()}
    bmap = _benchmarks(run_dir)
    if not eps.empty:
        eps["benchmark"] = eps["config_key"].map(lambda k: bmap.get(k, ""))
    if not steps.empty:
        steps["benchmark"] = steps["config_key"].map(lambda k: bmap.get(k, ""))
    benches = sorted(set(bmap.values())) or [""]

    L = ["# M1 walking-skeleton report", "",
         "> **Development data.** Produced by the M1 walking skeleton to check the pipeline end to end. "
         "Not a study result; excluded from all M2+ analyses.", "",
         f"- Experiment: `{manifest.get('experiment')}` / `{manifest.get('experiment_hash')}`",
         f"- Episodes logged: {len(eps)} of {manifest.get('n_specs', '?')} planned; steps logged: {len(steps)}",
         f"- Tasks: {len(manifest.get('task_ids', []))} across {', '.join(manifest.get('benchmarks', ['bfcl_multi_turn_base']))}",
         f"- Decoding: `{json.dumps(manifest.get('decoding', {}))}`; thinking={manifest.get('thinking')}; "
         f"prompt cache={manifest.get('prompt_cache')}", ""]

    # 1. accuracy
    L += ["## 1. Accuracy per configuration", ""]
    if not eps.empty:
        if "context_overflow" not in eps:
            eps["context_overflow"] = False
        g = eps.groupby(["config_key", "benchmark", "variant", "backend", "repair"], sort=False)
        acc = g.agg(episodes=("valid", "size"), accuracy=("valid", "mean"), forced_quit=("forced_quit", "sum"),
                    context_overflow=("context_overflow", "sum"), mean_steps=("n_steps", "mean")).reset_index()
        acc.insert(0, "config", acc.apply(_cfg_label, axis=1))
        acc = acc.drop(columns=["benchmark"])
        acc["planned"] = acc["config_key"].map(planned)
        for seed, sub in eps.groupby("seed"):
            acc[f"acc_seed{seed}"] = acc["config_key"].map(sub.groupby("config_key")["valid"].mean())
        L.append(_md(acc.drop(columns=["variant", "backend", "repair"])))
        L += ["Error types (bfcl checker) among failed episodes:", ""]
        et = eps[~eps.valid].assign(config=lambda d: d.apply(_cfg_label, axis=1)) \
            .groupby(["config", "error_type"]).size().rename("n").reset_index()
        L.append(_md(et))

    # 2. failure mix
    L += ["## 2. Failure-class mix per configuration (C6, per step)", ""]
    if not steps.empty and "failure_class" in steps:
        fm = pd.crosstab(steps.apply(_cfg_label, axis=1), steps["failure_class"])
        fm = fm.reindex(columns=[c for c in CLASSES if c in fm.columns])
        L.append(_md(fm.reset_index().rename(columns={"index": "config", "row_0": "config"})))
        n_unclassified = int(steps["failure_class"].isna().sum())
        L += [f"Unclassified steps: {n_unclassified}", ""]

    # 3. comparisons
    L += ["## 3. Paired comparisons (C10)", "",
          "Pairs are (task, seed). McNemar exact two-sided; 95% bootstrap CI over tasks for "
          "acc(reference) − acc(other). With 20 tasks these intervals are wide by design: M1 checks the "
          "machinery, not the effect.", ""]
    rows = []
    for bench in benches:
        for label, ref, other in COMPARISONS:
            fb = {"benchmark": bench} if bench else {}
            c = compare(eps, {**ref, **fb}, {**other, **fb}) if not eps.empty else {"n_pairs": 0}
            name = (f"{SHORT.get(bench, bench)}: " if len(benches) > 1 else "") + label
            if c["n_pairs"]:
                rows.append({"comparison": name, "pairs": c["n_pairs"], "acc_ref": c["acc_a"],
                             "acc_other": c["acc_b"], "n10": c["n10"], "n01": c["n01"], "p": c["p"],
                             "diff": c["ci_diff"], "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"]})
            else:
                rows.append({"comparison": name, "pairs": 0})
    L.append(_md(pd.DataFrame(rows)))

    # 3b. CPU vs CUDA output parity (FR-E17 preview): same variant, same task, seed, turn, step
    L += ["## 3b. Output parity: CPU vs CUDA (Q4_0)", "",
          "Share of steps whose raw model text is identical on the two backends, for the same task, seed, turn "
          "and step. Paths diverge after the first differing step, so the first-step rate is the cleanest number.", ""]
    for bench in (benches if not steps.empty else []):
        base = steps[(steps.variant == "q4_0") & (steps.repair == "none") & (steps.benchmark == bench)]
        a = base[base.backend == "cuda"].set_index(["task_id", "seed", "turn", "step"])["raw_text"]
        b = base[base.backend == "cpu"].set_index(["task_id", "seed", "turn", "step"])["raw_text"]
        j = a.to_frame("cuda").join(b.to_frame("cpu"), how="inner")
        tag = f"{bench}: " if len(benches) > 1 else ""
        if len(j):
            first = j.reset_index()
            first = first[(first.turn == 0) & (first.step == 0)]
            L += [f"- {tag}Aligned steps: {len(j)}; identical: {int((j.cuda == j.cpu).sum())} "
                  f"({(j.cuda == j.cpu).mean():.1%})",
                  f"- {tag}First step of each episode: {int((first.cuda == first.cpu).sum())} of {len(first)} identical"]
        else:
            L += [f"- {tag}_(no CPU-backend episodes yet)_"]
    L.append("")

    # 3c. C4 simulated-user cache
    db = os.path.join(run_dir, "simuser_cache.sqlite")
    if os.path.exists(db):
        from .simuser_cache import report as cache_report
        st = cache_report(db)
        L += ["## 3c. Simulated-user cache (C4)", "",
              f"Simulator `{manifest.get('simuser', {}).get('model', '?')}` "
              f"(`{(manifest.get('simuser', {}).get('sha256') or '')[:16]}`), greedy, behind the cache. "
              f"Stored replies: {st['stored_replies']}; simulator time: {st['upstream_hours'] * 60:.1f} min.", ""]
        rows = [{"role": role, "outcome": oc, "requests": v["n"], "upstream_s": v["upstream_s"]}
                for role, d in st["by_role"].items() for oc, v in d.items()]
        L.append(_md(pd.DataFrame(rows)))

    # 4. costs
    L += ["## 4. Costs (C9)", "",
          "Shared Colab GPU, so these check the measurement path only; cost results need the exclusive-node "
          "protocol (M6).", ""]
    if not steps.empty:
        crow = []
        for key, sub in steps.groupby("config_key", sort=False):
            e = eps[eps.config_key == key]
            c = step_costs(sub)
            crow.append({"config": _cfg_label(sub.iloc[0]), **c,
                         "wall_s_per_episode": float(e.wall_s.mean()) if len(e) else None,
                         "energy_j_per_episode": float(e.energy_j.mean()) if len(e) and e.energy_j.notna().any() else None,
                         "peak_gpu_mib": float(e.peak_gpu_mib.max()) if len(e) and e.peak_gpu_mib.notna().any() else None})
        L.append(_md(pd.DataFrame(crow)))

    # 5. planted effect
    pe = planted_effect_check()
    L += ["## 5. Planted-effect check (C10)", "",
          f"Synthetic paired data, 200 tasks × 2 seeds, planted drop {pe['planted_drop']:.2f} "
          f"(realized {pe['realized_diff']:.3f}).", "",
          f"- McNemar p = {pe['mcnemar_p']:.2e}; CI [{pe['ci']['lo']:.3f}, {pe['ci']['hi']:.3f}] → "
          f"**detected: {pe['detected']}**, CI covers realized: {pe['ci_covers_realized']}",
          f"- No-effect control: p = {pe['null_p']:.2f} → false positive: {pe['null_false_positive']}", ""]

    # 6. manifest
    L += ["## 6. Manifest (C11)", ""]
    if manifest:
        mf = manifest.get("model_files", {})
        L.append(_md(pd.DataFrame([{"variant": v, "file": os.path.basename(e["path"]), "bytes": e.get("bytes"),
                                    "sha256": (e.get("sha256") or "")[:16],
                                    "matches_pilot": e.get("sha256_matches_pilot", "n/a")} for v, e in mf.items()])))
        L += [f"- llama-server: " + "; ".join(f"{k}: `{(v.get('version') or '').splitlines()[0] if v.get('version') else ''}`"
                                              for k, v in manifest.get("llama_server", {}).items()),
              f"- bfcl-eval {manifest.get('bfcl_eval', {}).get('version')}; testbed commit "
              f"`{manifest.get('testbed_git', {}).get('commit', '')[:10]}` (dirty={manifest.get('testbed_git', {}).get('dirty')})",
              f"- GPU: {manifest.get('gpu')}; host {manifest.get('host')}; Python {manifest.get('python')}", ""]
    others = sorted(glob.glob(os.path.join(run_dir, "manifests", "*.json")))
    for p in others:
        m = json.load(open(p))
        L.append(f"- Host record `{os.path.basename(p)}`: GPU {m.get('gpu') or 'none'}; CPU {m.get('cpu', {}).get('model', '?')} "
                 f"× {m.get('cpu', {}).get('logical_cpus', '?')} logical, threads flag {m.get('cpu', {}).get('threads_flag')}")
    L.append("")
    out = os.path.join(run_dir, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    return out
