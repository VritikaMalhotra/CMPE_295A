"""M1 exit criterion 6: a second member reruns part of the skeleton from a clean clone.

    .venv/bin/python scripts/clean_clone_check.py --out /content/cc_runs [--limit 10] [--config dc08400d1b5b]

Checks, against the Oct 25 reference run (experiment skeleton/3e3a25e70f, report.md):
  1. the dry run expands to the same 253 specs: same experiment hash, same 14 configuration keys;
  2. the chosen configuration (default: BFCL Q4_0 on CUDA, 40 planned) runs ``--limit`` episodes end to end
     into a fresh output directory, every step logged and classified;
  3. the verdicts agree with the reference (that configuration scored 0 of 40) and the model file hashes are
     recorded (whether they equal the pilot's SHA-256 is reported, not required).
Writes <out>/clean_clone_check.json and prints PASS or FAIL. Token-level identity is not expected on a
different GPU (the Oct 25 CPU-vs-CUDA parity was 32%), so the check is on specs, completion and verdicts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = {
    "experiment_hash": "3e3a25e70f",
    "config_keys": {"8bae173bdd21", "ec9d23f8549d", "dc08400d1b5b", "7d5318997f9f", "b373660af0bb", "74260a9a88e3",
                    "80e76cc888af", "7da4e492b2e6", "bf40118a38c4", "889ab84b6aae", "8822b612b855", "d3bb4197f6ec",
                    "39fe50886824", "14d44cccecc4"},
    "accuracy": {"dc08400d1b5b": 0.0, "8bae173bdd21": 0.0, "ec9d23f8549d": 0.0},
    "mean_steps": {"dc08400d1b5b": 7.15, "8bae173bdd21": 7.5, "ec9d23f8549d": 7.22},
    "commit": "05ec975",
}


def run(args: list[str], env: dict) -> str:
    p = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), "-m", "testbed.run", "+experiment=skeleton", *args],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        print(out[-3000:])
        sys.exit(f"testbed.run {' '.join(args)} failed with exit code {p.returncode}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="fresh output directory (not the shared Drive results)")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--config", default="dc08400d1b5b", help="a BFCL configuration key (default Q4_0/CUDA)")
    a = ap.parse_args()
    env = {**os.environ, "TESTBED_OUT": a.out}
    res: dict = {"host": platform.node(), "config": a.config, "limit": a.limit}
    res["commit"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()

    # 1. same plan
    dry = run(["dry_run=true"], env)
    m = re.search(r"dir: .*/skeleton/([0-9a-f]{10})", dry)
    res["experiment_hash"] = m.group(1) if m else None
    keys = set(re.findall(r"^\s+([0-9a-f]{12})\s", dry, flags=re.M))
    res["specs_match"] = res["experiment_hash"] == REF["experiment_hash"] and keys == REF["config_keys"]
    res["n_specs"] = int(re.search(r"(\d+) specs", dry).group(1)) if re.search(r"(\d+) specs", dry) else None

    # 2. run part of one configuration
    run(["local=true", f"only_config={a.config}", f"limit={a.limit}", "allow_sha_mismatch=true"], env)
    d = os.path.join(a.out, "skeleton", res["experiment_hash"] or "")
    eps = [json.load(open(p)) for p in glob.glob(os.path.join(d, "episodes", "*.json"))]
    eps = [e for e in eps if e.get("config_key") == a.config]
    steps = [json.loads(l) for p in glob.glob(os.path.join(d, "steps", "*.jsonl")) for l in open(p) if l.strip()]
    res["episodes"] = len(eps)
    res["steps"] = len(steps)
    res["unclassified_steps"] = sum(1 for s in steps if not s.get("failure_class"))
    res["accuracy"] = (sum(bool(e["valid"]) for e in eps) / len(eps)) if eps else None
    res["mean_steps"] = (sum(e["n_steps"] for e in eps) / len(eps)) if eps else None
    res["reference"] = {"accuracy": REF["accuracy"].get(a.config), "mean_steps": REF["mean_steps"].get(a.config),
                        "experiment_hash": REF["experiment_hash"], "commit": REF["commit"]}

    # 3. model files recorded
    mans = [json.load(open(p)) for p in glob.glob(os.path.join(d, "manifests", "*.json"))]
    files = mans[-1]["model_files"] if mans else {}
    res["model_files"] = {v: {"sha256": (e.get("sha256") or "")[:16], "matches_pilot": e.get("sha256_matches_pilot")}
                          for v, e in files.items()}
    res["gpu"] = mans[-1].get("gpu") if mans else None

    ok = (res["specs_match"] and res["episodes"] >= a.limit and res["unclassified_steps"] == 0
          and (REF["accuracy"].get(a.config) is None or res["accuracy"] == REF["accuracy"][a.config]))
    res["pass"] = bool(ok)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "clean_clone_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
