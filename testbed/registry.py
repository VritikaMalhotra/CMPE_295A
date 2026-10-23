"""C11 Config & Registry: experiment config -> run specs, each once, keyed by content hash.

Owner: Yashashav Devalapalli Kamalraj.

* ``expand(cfg)``     - the experiment file becomes a list of RunSpecs (no duplicates allowed).
* ``Registry``        - the experiment directory: ``specs.jsonl`` (planned), ``episodes/<run_id>.json``
                        (done; written atomically by C5), ``manifest.json``. A spec is done iff its
                        episode file exists, so a disconnect or a Slurm walltime kill resumes cleanly.
* ``build_manifest``  - everything that can change an output: server flags, caching, thinking mode,
                        decoding, llama.cpp / bfcl-eval / testbed versions, model file SHA-256.
The experiment directory name is a hash of the spec list, so a changed config never mixes with old results.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from typing import Iterable

from .spec import RunSpec, canonical_json, sha256_text


def expand(exp: dict, task_ids: list[str]) -> list[RunSpec]:
    common = dict(experiment=exp["name"], model=exp["model"], benchmark=exp["benchmark"],
                  decoding=dict(exp["decoding"]), server=dict(exp["server"]), thinking=bool(exp["thinking"]),
                  max_steps_per_turn=int(exp["max_steps_per_turn"]))
    specs = []
    for b in exp["baselines"]:
        for seed in exp["baseline_seeds"]:
            for t in task_ids:
                specs.append(RunSpec(variant=b["variant"], backend=b["backend"], repair="none", constraint="none",
                                     task_id=t, seed=int(seed), **common))
    for r in exp.get("repairs", []):
        for seed in exp["repair_seeds"]:
            for t in task_ids:
                specs.append(RunSpec(variant=r["variant"], backend=r["backend"], repair=r["repair"],
                                     constraint=r.get("constraint", "none"), task_id=t, seed=int(seed), **common))
    dup = [k for k, n in Counter(s.run_id for s in specs).items() if n > 1]
    if dup:
        raise ValueError(f"duplicate run specs: {dup[:5]}")
    return specs


def experiment_hash(specs: Iterable[RunSpec]) -> str:
    return sha256_text(canonical_json(sorted(s.run_id for s in specs)))[:10]


def file_sha256(path: str, cache: bool = True) -> str:
    side = path + ".sha256"
    if cache and os.path.exists(side) and os.path.getmtime(side) >= os.path.getmtime(path):
        return open(side).read().split()[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache:
        try:
            with open(side, "w") as f:
                f.write(f"{digest}  {os.path.basename(path)}\n")
        except OSError:
            pass
    return digest


def _cmd(args: list[str]) -> str:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return (p.stdout.strip() + "\n" + p.stderr.strip()).strip()
    except Exception as e:
        return f"unavailable: {type(e).__name__}"


def build_manifest(specs: list[RunSpec], model_files: dict[str, str], binaries: dict[str, str],
                   expected_sha: dict | None = None, hash_files: bool = True) -> dict:
    try:
        import bfcl_eval
        from importlib.metadata import version
        bfcl_ver = version("bfcl-eval")
        bfcl_path = os.path.dirname(bfcl_eval.__file__)
    except Exception:
        bfcl_ver, bfcl_path = "unavailable", ""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = _cmd(["git", "-C", repo, "rev-parse", "HEAD"]).splitlines()[:1]
    dirty = bool(_cmd(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"]))
    files = {}
    for variant, path in sorted(model_files.items()):
        entry = {"path": path, "exists": os.path.exists(path)}
        if entry["exists"]:
            entry["bytes"] = os.path.getsize(path)
            if hash_files:
                entry["sha256"] = file_sha256(path)
                if expected_sha and variant in expected_sha:
                    entry["sha256_matches_pilot"] = entry["sha256"] == expected_sha[variant]
        files[variant] = entry
    configs = {}
    for s in specs:
        configs.setdefault(s.config_key, {"variant": s.variant, "backend": s.backend, "repair": s.repair,
                                          "constraint": s.constraint, "episodes": 0})
        configs[s.config_key]["episodes"] += 1
    return {
        "experiment": specs[0].experiment if specs else None,
        "experiment_hash": experiment_hash(specs),
        "n_specs": len(specs),
        "configs": configs,
        "task_ids": sorted({s.task_id for s in specs}, key=lambda t: int(t.rsplit("_", 1)[-1])),
        "decoding": specs[0].decoding if specs else {},
        "server": specs[0].server if specs else {},
        "thinking": specs[0].thinking if specs else None,
        "prompt_cache": (specs[0].decoding or {}).get("cache_prompt", False) if specs else None,
        "model_files": files,
        "llama_server": {k: {"path": v, "version": _cmd([v, "--version"])[:400] if os.path.exists(v) else "missing"}
                         for k, v in binaries.items()},
        "bfcl_eval": {"version": bfcl_ver, "path": bfcl_path},
        "testbed_git": {"commit": git[0] if git else "unknown", "dirty": dirty},
        "python": platform.python_version(),
        "host": platform.node(),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "label": "development data (M1 walking skeleton), excluded from the study",
    }


def atomic_json(path: str, obj) -> str:
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    return path


class Registry:
    def __init__(self, out_root: str, specs: list[RunSpec]):
        self.specs = specs
        self.dir = os.path.join(out_root, specs[0].experiment, experiment_hash(specs))
        os.makedirs(os.path.join(self.dir, "episodes"), exist_ok=True)

    def write_specs(self) -> str:
        p = os.path.join(self.dir, "specs.jsonl")
        with open(p + ".part", "w") as f:
            for s in self.specs:
                f.write(json.dumps({"run_id": s.run_id, "config_key": s.config_key, **s.content()}) + "\n")
        os.replace(p + ".part", p)
        return p

    def write_manifest(self, manifest: dict) -> str:
        return atomic_json(os.path.join(self.dir, "manifest.json"), manifest)

    def done_ids(self) -> set[str]:
        return {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(self.dir, "episodes", "*.json"))}

    def pending(self) -> list[RunSpec]:
        done = self.done_ids()
        return [s for s in self.specs if s.run_id not in done]

    def summary(self) -> dict:
        done = self.done_ids()
        by_cfg: dict = {}
        for s in self.specs:
            c = by_cfg.setdefault(s.config_key, {"variant": s.variant, "backend": s.backend, "repair": s.repair,
                                                  "planned": 0, "done": 0})
            c["planned"] += 1
            c["done"] += s.run_id in done
        return {"dir": self.dir, "planned": len(self.specs), "done": len(done & {s.run_id for s in self.specs}),
                "unique_run_ids": len({s.run_id for s in self.specs}), "configs": by_cfg}
