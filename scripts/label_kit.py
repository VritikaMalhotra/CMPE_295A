#!/usr/bin/env python
"""H2 label kit (Workbook 2 FR-E8, Ch. 7 "Measurement Validity"): blank rater sheets, a hidden key, and kappa.

Two sheets, both drawn from logged steps with the classifier's label hidden:
- the **30-step rater sample** (M2 plan item 14: a second person checks codebook v0, including the two classes
  added by decisions 3a/3b), and
- the **300 stratified pilot steps** (M3 exit: two independent human raters, Cohen's kappa >= 0.80).
Steps are stratified by the classifier's class and by precision (16-bit or 4-bit), so rare classes and the 4-bit
steps the study is about are both covered. The rater sees what the logger recorded about the step (as in
``scripts/rater_check.py``); model, precision, run and the machine label are hidden. They go to a **key file
kept outside the repository**, which raters never see. This script never writes a label into a sheet.

    # export (cluster, from run directories that have steps.parquet or steps/*.jsonl):
    python scripts/label_kit.py export --runs <dir> [<dir> ...] --out-dir labels/h2_kit \\
        --key-dir $CMPE295_GROUP/labels/h2_kit_<date> [--n-sample 30 --n-pilot 300 --seed 295]
    # agreement between two filled copies of one sheet (and, with --key, each rater against the classifier):
    python scripts/label_kit.py kappa rater_a.csv rater_b.csv [--key key_pilot_300.csv] [--out kappa.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.classifier import CLASSES, FAILURES, classify_step  # noqa: E402
from testbed.spec import StepRow  # noqa: E402

SHOWN = ["benchmark", "turn", "step", "turn_has_ground_truth", "is_last_step_of_turn", "forced_quit",
         "stop_reason", "n_tool_call_blocks", "known_tools", "schema_errors", "decoded_calls",
         "execution_results", "raw_text"]
HIDDEN = ["run_dir", "run_id", "config_key", "task_id", "seed", "variant", "backend", "repair", "constraint",
          "precision", "machine_label", "logged_failure_class"]
SIXTEEN_BIT = {"bf16", "f16", "fp16", "f32"}


def precision_of(variant: str) -> str:
    v = str(variant).lower()
    return "16bit" if v in SIXTEEN_BIT else "8bit" if v.startswith("q8") else "4bit"
LABEL_COLS = ["label", "note"]


def _cell(v) -> str:
    if hasattr(v, "tolist"):
        v = v.tolist()
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, bool)) else ("" if v is None else str(v))


def load_run(run_dir: str) -> list[dict]:
    """Steps of finished episodes from ``steps/*.jsonl`` (the logger's source of truth; ``steps.parquet`` stores
    list columns as JSON text). Rows from runs older than the ``benchmark`` field get it from ``specs.jsonl``."""
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run_dir, "episodes", "*.json"))}
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "steps", "*.jsonl"))):
        if done and os.path.basename(p)[:-6] not in done:
            continue
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"no steps under {run_dir}")
    bench = {}
    sp = os.path.join(run_dir, "specs.jsonl")
    if os.path.exists(sp):
        with open(sp) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    bench[d["run_id"]] = d.get("benchmark", "")
    for r in rows:
        if not r.get("benchmark"):
            r["benchmark"] = bench.get(r.get("run_id"), "")
    fields = {f for f in StepRow.__dataclass_fields__}
    for r in rows:
        r["run_dir"] = run_dir
        r["logged_failure_class"] = r.get("failure_class", "")
        # relabel with the current classifier (codebook v0 with decisions 3a/3b), the version the raters check
        r["machine_label"] = classify_step(StepRow(**{k: r.get(k) for k in fields if k in r}))
        r["precision"] = precision_of(r.get("variant", ""))
    # the kit labels plain model output: repair runs (R1 grammar, R2 bits) are left out
    return [r for r in rows if r.get("repair", "none") == "none" and r["precision"] in ("16bit", "4bit")]


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Equal share per (machine label, precision) stratum, capped by what each holds; the remainder goes to the
    largest strata. Every stratum present gets at least one step when n allows."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["machine_label"], r["precision"])].append(r)
    for v in by.values():
        rng.shuffle(v)
    n = min(n, len(rows))
    take = {k: 0 for k in by}
    left = n
    open_ = sorted(by)
    while left and open_:
        share = max(1, left // len(open_))
        for k in list(open_):
            add = min(share, len(by[k]) - take[k], left)
            take[k] += add
            left -= add
            if take[k] >= len(by[k]):
                open_.remove(k)
            if not left:
                break
    out = [r for k in by for r in by[k][:take[k]]]
    rng.shuffle(out)
    return out


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_sheet(rows: list[dict], prefix: str, sheet: str, key: str) -> None:
    with open(sheet, "w", newline="", encoding="utf-8") as f, open(key, "w", newline="", encoding="utf-8") as k:
        w, kw = csv.writer(f), csv.writer(k)
        w.writerow(["item"] + SHOWN + LABEL_COLS)
        kw.writerow(["item"] + HIDDEN)
        for i, r in enumerate(rows, 1):
            item = f"{prefix}-{i:03d}"
            w.writerow([item] + [_cell(r.get(c, "")) for c in SHOWN] + ["" for _ in LABEL_COLS])
            kw.writerow([item] + [_cell(r.get(c, "")) for c in HIDDEN])


def cmd_export(a) -> int:
    rows = [r for d in a.runs for r in load_run(d)]
    rng = random.Random(a.seed)
    sample = stratified(rows, a.n_sample, rng)
    taken = {(r["run_id"], r["turn"], r["step"]) for r in sample}
    pilot = stratified([r for r in rows if (r["run_id"], r["turn"], r["step"]) not in taken], a.n_pilot, rng)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.key_dir, exist_ok=True)
    out = {"what": "H2 label kit: blank rater sheets (labels to be filled by human raters only)",
           "seed": a.seed, "sources": a.runs, "n_steps_in_sources": len(rows), "sheets": {},
           "key_dir": a.key_dir, "label_set": list(CLASSES)}
    for name, prefix, picked in (("rater_sample_30", "R", sample), ("pilot_300", "P", pilot)):
        sheet = os.path.join(a.out_dir, f"{name}.csv")
        key = os.path.join(a.key_dir, f"key_{name}.csv")
        write_sheet(picked, prefix, sheet, key)
        os.chmod(key, 0o640)
        out["sheets"][name] = {"items": len(picked), "sheet": sheet, "sheet_sha256": _sha(sheet),
                               "key_sha256": _sha(key)}
        print(f"{name}: {len(picked)} steps -> {sheet}; key (hidden) -> {key}")
    with open(os.path.join(a.key_dir, "strata.json"), "w") as f:      # stratum counts stay with the key
        json.dump({n: dict(collections.Counter(f"{r['machine_label']}|{r['precision']}" for r in p))
                   for n, p in (("rater_sample_30", sample), ("pilot_300", pilot))}, f, indent=1)
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return 0


# ---- agreement ---------------------------------------------------------------------------------------------

def cohen_kappa(x: list[str], y: list[str]) -> dict:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = collections.Counter(x), collections.Counter(y)
    pe = sum(cx[c] * cy[c] for c in set(x) | set(y)) / (n * n)
    return {"n": n, "observed_agreement": po, "expected_agreement": pe,
            "kappa": 1.0 if pe == 1 else (po - pe) / (1 - pe)}


def _labels(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item"]: (r.get("label") or "").strip() for r in csv.DictReader(f)}


def agreement(a: dict[str, str], b: dict[str, str]) -> dict:
    items = sorted(set(a) & set(b))
    missing = [i for i in items if not a[i] or not b[i]]
    invalid = [i for i in items if (a[i] and a[i] not in CLASSES) or (b[i] and b[i] not in CLASSES)]
    if missing or invalid:
        raise ValueError(f"unlabelled items {missing[:10]}; labels outside the codebook {invalid[:10]}")
    x, y = [a[i] for i in items], [b[i] for i in items]
    res = cohen_kappa(x, y)
    # failure vs non-failure (the H1 outcome), and per class: how often both raters agree when either uses it
    res["kappa_failure_binary"] = cohen_kappa([str(v in FAILURES) for v in x], [str(v in FAILURES) for v in y])["kappa"]
    per = {}
    for c in sorted(set(x) | set(y)):
        either = sum(1 for p, q in zip(x, y) if c in (p, q))
        both = sum(1 for p, q in zip(x, y) if p == q == c)
        per[c] = {"either": either, "both": both, "specific_agreement": 2 * both / (x.count(c) + y.count(c))}
    res["per_class"] = per
    conf = collections.Counter(zip(x, y))
    res["confusion"] = {f"{p} | {q}": n for (p, q), n in sorted(conf.items()) if p != q}
    res["meets_0.80"] = res["kappa"] >= 0.80
    return res


def cmd_kappa(a) -> int:
    ra, rb = _labels(a.rater_a), _labels(a.rater_b)
    out = {"rater_a": a.rater_a, "rater_b": a.rater_b, "a_vs_b": agreement(ra, rb)}
    if a.key:
        with open(a.key, newline="", encoding="utf-8") as f:
            mk = {r["item"]: r["machine_label"] for r in csv.DictReader(f)}
        out["a_vs_classifier"] = agreement(ra, {i: mk[i] for i in ra if i in mk})
        out["b_vs_classifier"] = agreement(rb, {i: mk[i] for i in rb if i in mk})
    r = out["a_vs_b"]
    print(f"{r['n']} items: agreement {r['observed_agreement']:.3f}, Cohen's kappa {r['kappa']:.3f} "
          f"(target >= 0.80: {'met' if r['meets_0.80'] else 'not met'}); failure/non-failure kappa "
          f"{r['kappa_failure_binary']:.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--key-dir", required=True)
    e.add_argument("--n-sample", type=int, default=30)
    e.add_argument("--n-pilot", type=int, default=300)
    e.add_argument("--seed", type=int, default=295)
    k = sub.add_parser("kappa")
    k.add_argument("rater_a")
    k.add_argument("rater_b")
    k.add_argument("--key")
    k.add_argument("--out")
    a = ap.parse_args()
    return cmd_export(a) if a.cmd == "export" else cmd_kappa(a)


if __name__ == "__main__":
    sys.exit(main())
