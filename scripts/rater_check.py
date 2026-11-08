"""C6 second-rater check (Workbook Ch. 7, "Measurement Validity"; M1 size: 30 steps).

A second team member labels a stratified sample of logged steps from the written codebook (the rule list in
testbed/classifier.py), blind to model, precision, repair and the classifier's own label; Cohen's kappa
then compares the two. The full check before the main runs uses about 300 steps and needs kappa >= 0.80.

    # 1. whoever has the run directory (e.g. Colab with Drive mounted) draws the sample:
    python scripts/rater_check.py sample <exp_dir> --n 30 --out rate_steps.csv --key rate_key.csv
    #    -> send rate_steps.csv to the rater; keep rate_key.csv (it holds the hidden labels)
    # 2. the rater fills the `label` column of rate_steps.csv with one class per row
    # 3. score it:
    python scripts/rater_check.py score rate_steps.csv rate_key.csv

<exp_dir> is one experiment directory, e.g. .../runs/skeleton/3e3a25e70f (it has steps/*.jsonl).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.classifier import CLASSES  # noqa: E402

# What the rater sees: the step as logged, plus the facts the logger recorded about it. Hidden: run id,
# model variant, backend, repair, constraint and the classifier's failure_class.
SHOWN = ["benchmark", "turn", "step", "turn_has_ground_truth", "is_last_step_of_turn", "forced_quit",
         "stop_reason", "n_tool_call_blocks", "known_tools", "schema_errors", "decoded_calls",
         "execution_results", "raw_text"]
HIDDEN = ["run_id", "config_key", "task_id", "seed", "variant", "backend", "repair", "constraint", "failure_class"]


def _cell(v) -> str:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, bool)) else str(v)


def load_steps(exp_dir: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(exp_dir, "steps", "*.jsonl"))):
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"no steps under {exp_dir}/steps")
    return rows


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """At least one step of every class present, then proportional to class size (largest remainder)."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("failure_class") or "?"].append(r)
    for v in by.values():
        rng.shuffle(v)
    n = min(n, len(rows))
    take = {c: 1 for c in by} if n >= len(by) else {}
    left = n - sum(take.values())
    total = sum(len(v) for v in by.values())
    quota = {c: left * len(v) / total for c, v in by.items()}
    for c in by:
        take[c] = take.get(c, 0) + int(quota[c])
    for c in sorted(by, key=lambda c: quota[c] - int(quota[c]), reverse=True)[: n - sum(take.values())]:
        take[c] += 1
    out = [r for c, v in by.items() for r in v[: min(take[c], len(v))]]
    rng.shuffle(out)
    return out


def cmd_sample(a) -> int:
    picked = stratified(load_steps(a.exp_dir), a.n, a.seed)
    with open(a.out, "w", newline="") as f, open(a.key, "w", newline="") as k:
        w = csv.writer(f)
        kw = csv.writer(k)
        w.writerow(["item"] + SHOWN + ["label", "note"])
        kw.writerow(["item"] + HIDDEN)
        for i, r in enumerate(picked, 1):
            w.writerow([i] + [_cell(r.get(c, "")) for c in SHOWN] + ["", ""])
            kw.writerow([i] + [_cell(r.get(c, "")) for c in HIDDEN])
    counts = collections.Counter(r.get("failure_class") for r in picked)
    print(f"wrote {a.out} ({len(picked)} steps, blind) and {a.key} (hidden labels; do not send)")
    print("strata:", dict(sorted(counts.items())))
    print("classes the rater may use:", ", ".join(CLASSES))
    return 0


def cohen_kappa(x: list[str], y: list[str]) -> tuple[float, float]:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = collections.Counter(x), collections.Counter(y)
    pe = sum(cx[c] * cy[c] for c in set(x) | set(y)) / (n * n)
    return po, (1.0 if pe == 1 else (po - pe) / (1 - pe))


def cmd_score(a) -> int:
    with open(a.rated) as f:
        rated = {r["item"]: r for r in csv.DictReader(f)}
    with open(a.key) as f:
        key = {r["item"]: r for r in csv.DictReader(f)}
    items = sorted(set(rated) & set(key), key=int)
    bad = [i for i in items if rated[i]["label"].strip() not in CLASSES]
    if bad:
        sys.exit(f"items without a valid label (use one of {CLASSES}): {bad}")
    rater = [rated[i]["label"].strip() for i in items]
    clf = [key[i]["failure_class"] for i in items]
    po, kappa = cohen_kappa(rater, clf)
    print(f"{len(items)} steps: raw agreement {po:.2f}, Cohen's kappa {kappa:.2f} (Workbook target >= 0.80)")
    print("disagreements (item: rater vs classifier):")
    for i, r, c in zip(items, rater, clf):
        if r != c:
            print(f"  {i}: {r} vs {c}  {rated[i].get('note', '')}")
    per = collections.defaultdict(lambda: [0, 0])
    for r, c in zip(rater, clf):
        per[c][0] += 1
        per[c][1] += r == c
    print("per classifier class (n, rater agreed):", {c: tuple(v) for c, v in sorted(per.items())})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("exp_dir")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="rate_steps.csv")
    s.add_argument("--key", default="rate_key.csv")
    c = sub.add_parser("score")
    c.add_argument("rated")
    c.add_argument("key")
    a = ap.parse_args()
    return cmd_sample(a) if a.cmd == "sample" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
