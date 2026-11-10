#!/usr/bin/env python
"""C10: record both planted-effect checks in one JSON (results/m2/planted_effect.json).

1. ``testbed.stats.planted_effect_check``: the M1 episode-success check (exact McNemar + task-cluster bootstrap)
   on synthetic paired data with a known drop, plus a no-effect copy that must not be flagged.
2. ``testbed.mixed_effects.planted_effect_check``: the registered lme4 models A/B/C (analysis/mixed_effects.R)
   fitted to simulated C5 step and episode tables with known 4-bit effects, plus a no-effect dataset.

Needs R 4.4 + lme4 (cluster: slurm/r_analysis.sbatch sets it up). Pass = both checks recover their planted
effects and neither flags the null data.

    python scripts/planted_effect.py --out results/m2/planted_effect.json [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testbed import mixed_effects, stats  # noqa: E402


def r_versions() -> dict:
    exe = mixed_effects.rscript()
    if not exe:
        return {}
    r = subprocess.run([exe, "-e", "cat(R.version.string, '|', as.character(packageVersion('lme4')))"],
                       capture_output=True, text=True)
    parts = [p.strip() for p in r.stdout.split("|")]
    return {"R": parts[0] if parts else None, "lme4": parts[1] if len(parts) > 1 else None}


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    t0 = time.time()
    out = {"what": "C10 planted-effect checks: synthetic data with known effects (development check, not study data)",
           "commit": git_head(), "python": platform.python_version(), **r_versions(), "seeds": a.seeds,
           "stats": [], "mixed_effects": []}
    for sd in a.seeds:
        st = stats.planted_effect_check(seed=sd)
        st["pass"] = bool(st["detected"] and st["ci_covers_realized"] and not st["null_false_positive"])
        out["stats"].append({"seed": sd, **st})
        me = mixed_effects.planted_effect_check(seed=sd)
        me["pass"] = bool(me["recovered"] and not me["null_false_positive"])
        out["mixed_effects"].append({"seed": sd, **me})
        print(f"seed {sd}: stats pass={st['pass']} (p={st['mcnemar_p']:.2g}, null p={st['null_p']:.2g}); "
              f"mixed effects pass={me['pass']} (A family {me['family_A']}, "
              f"status {me['status']})", flush=True)
    out["pass"] = all(x["pass"] for x in out["stats"] + out["mixed_effects"])
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
        f.write("\n")
    print(f"wrote {a.out}: pass={out['pass']}")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
