"""One command for the whole experiment (C11 entry point).

    python -m testbed.run +experiment=skeleton dry_run=true   # expand + check, run nothing
    python -m testbed.run +experiment=skeleton local=true     # run every pending episode here (Colab)
    python -m testbed.run +experiment=skeleton                # write a Slurm array job (one task per config)
    python -m testbed.run +experiment=skeleton report_only=true

Resumable: a spec is done iff ``episodes/<run_id>.json`` exists, so re-running the same command after a
disconnect continues where it stopped. One llama-server per configuration; episodes inside a configuration
run one at a time (``-np 1``) so timings are not shared between requests.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import OrderedDict

import hydra
from omegaconf import DictConfig, OmegaConf

from . import registry as reg
from .adapter import load_bfcl_multi_turn, skeleton_task_ids
from .backends import Backend, LlamaServer, ServerConfig
from .grammar import task_grammar, task_grammar_free
from .logger import StepLogger, to_parquet
from .profiler import GpuMeter
from .runner import run_episode
from .spec import RunSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAMMARS = {"none": None, "free": task_grammar_free, "forced": task_grammar}   # spec.Constraint values


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _plan(cfg: DictConfig):
    exp = OmegaConf.to_container(cfg.experiment, resolve=True)
    model = OmegaConf.to_container(cfg.models[exp["model"]], resolve=True)
    category = exp["benchmark"].removeprefix("bfcl_")
    task_ids = skeleton_task_ids(int(exp["n_tasks"]), category)
    specs = reg.expand(exp, task_ids)
    model_files = {v: os.path.join(_abs(cfg.paths.models_dir), f) for v, f in model["files"].items()
                   if any(s.variant == v for s in specs)}
    binaries = {"cuda": _abs(cfg.paths.llama_cuda), "cpu": _abs(cfg.paths.llama_cpu)}
    binaries = {k: v for k, v in binaries.items() if any(s.backend == k for s in specs)}
    return exp, model, category, task_ids, specs, model_files, binaries


def _group(specs: list[RunSpec]) -> "OrderedDict[str, list[RunSpec]]":
    g: OrderedDict[str, list[RunSpec]] = OrderedDict()
    for s in specs:
        g.setdefault(s.config_key, []).append(s)
    return g


def _preflight(specs, model_files, binaries, model) -> list[str]:
    problems = []
    for v, p in model_files.items():
        if not os.path.exists(p):
            problems.append(f"missing model file for {v}: {p}")
    for b, p in binaries.items():
        if not os.path.exists(p):
            problems.append(f"missing llama-server ({b}): {p}")
    return problems


def run_local(cfg, r: reg.Registry, specs, category, model_files, binaries) -> None:
    pending = r.pending()
    if cfg.only_config:
        pending = [s for s in pending if s.config_key == cfg.only_config]
    if cfg.backend:
        pending = [s for s in pending if s.backend == cfg.backend]
    if cfg.limit:
        pending = pending[: int(cfg.limit)]
    if not pending:
        print("nothing pending")
        return
    tasks = {t.id: t for t in load_bfcl_multi_turn(category, ids=sorted({s.task_id for s in pending}))}
    logs = os.path.join(r.dir, "server_logs")
    os.makedirs(logs, exist_ok=True)
    t_all = time.time()
    n_done = 0
    for key, group in _group(pending).items():
        lock = _try_lock(os.path.join(r.dir, "locks", f"{key}.lock"))
        if lock is None:
            print(f"[{key}] held by another running process; skipping", flush=True)
            continue
        done = r.done_ids()                             # recheck under the lock: another process may
        group = [s for s in group if s.run_id not in done]   # have finished some while we waited
        if not group:
            lock.close()
            continue
        s0 = group[0]
        if s0.constraint not in GRAMMARS:
            raise ValueError(f"unknown constraint {s0.constraint!r}")
        scfg = ServerConfig(binary=binaries[s0.backend], model_path=model_files[s0.variant], backend=s0.backend,
                            ctx=int(s0.server.get("ctx", 40960)),
                            threads=cfg.cpu_threads if s0.backend == "cpu" else None)
        gfn = GRAMMARS[s0.constraint]
        print(f"[{key}] {s0.variant}/{s0.backend} repair={s0.repair} constraint={s0.constraint}: "
              f"{len(group)} episodes", flush=True)
        try:
            srv = LlamaServer(scfg, log_path=os.path.join(logs, f"{key}.log")).__enter__()
        except Exception as e:                          # one bad config must not stop the others
            print(f"  server failed to start: {e}; skipping this configuration", flush=True)
            lock.close()
            continue
        try:
            be = Backend(srv, s0.decoding, thinking=s0.thinking)
            for s in group:
                if os.path.exists(os.path.join(r.dir, "episodes", f"{s.run_id}.json")):
                    continue
                log = StepLogger(r.dir, s.run_id)
                try:
                    with GpuMeter() if s.backend == "cuda" else _NullMeter() as m:
                        ep = run_episode(s, tasks[s.task_id], be, gfn, log)
                    ep.energy_j, ep.peak_gpu_mib = m.energy_j, m.peak_mib
                    log.finish(ep)
                    n_done += 1
                    print(f"  {s.task_id} seed={s.seed} valid={ep.valid} steps={ep.n_steps} "
                          f"{ep.wall_s:.1f}s  ({n_done}/{len(pending)})", flush=True)
                except KeyboardInterrupt:
                    log.abort()
                    raise
                except Exception:
                    log.abort()
                    print(f"  {s.task_id} seed={s.seed} FAILED (will retry on next run)\n"
                          + traceback.format_exc(limit=3), flush=True)
        finally:
            srv.__exit__(None, None, None)
            lock.close()                                # releases the flock
    print(f"finished {n_done} episodes in {time.time() - t_all:.0f}s")


def _cpu_info(cfg) -> dict:
    model = ""
    try:
        model = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    return {"threads_flag": cfg.cpu_threads, "logical_cpus": os.cpu_count(), "model": model}


def _try_lock(path: str):
    """Exclusive non-blocking flock on one configuration; None if another live process holds it.
    The OS drops the lock when its holder dies, so a killed run never leaves a stale lock."""
    import fcntl
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()} {platform_node()}\n")
    f.flush()
    return f


def platform_node() -> str:
    import platform
    return platform.node()


class _NullMeter:
    energy_j = peak_mib = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def write_slurm(cfg, r: reg.Registry, exp_name: str) -> str:
    keys = list(_group(r.pending()).keys())
    if not keys:
        print("nothing pending")
        return ""
    cfg_list = os.path.join(r.dir, "configs.txt")
    with open(cfg_list, "w") as f:
        f.write("\n".join(keys) + "\n")
    sb = cfg.slurm
    common = dict(job=f"tb-{exp_name}", partition=sb.partition_gpu, gres=sb.gres_gpu, time=sb.time,
                  account_line=f"#SBATCH -A {sb.account}" if sb.account else "",
                  logdir=os.path.join(r.dir, "slurm_logs"), repo=REPO, experiment=exp_name)
    os.makedirs(common["logdir"], exist_ok=True)
    out = os.path.join(r.dir, "array.sbatch")
    with open(out, "w") as f:
        f.write(open(os.path.join(REPO, "slurm", "array.sbatch.tmpl")).read().format(
            n=len(keys) - 1, max_conc=sb.max_concurrent, cfg_list=cfg_list, **common))
    rep = os.path.join(r.dir, "report.sbatch")
    with open(rep, "w") as f:
        f.write(open(os.path.join(REPO, "slurm", "report.sbatch.tmpl")).read().format(
            partition_cpu=sb.partition_cpu, **common))
    print(f"wrote {out} ({len(keys)} array tasks) and {rep} (runs after the array)")
    if cfg.submit:
        import subprocess
        jid = subprocess.run(["sbatch", "--parsable", out], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["sbatch", f"--dependency=afterany:{jid.split(';')[0]}", rep], check=True)
        print(f"submitted array {jid} + dependent report job")
    return out


def _check_model_hashes(r: reg.Registry, manifest: dict, allow: bool) -> None:
    """Results are keyed by variant name; refuse to resume into a directory whose episodes came from
    different model files (e.g. a re-exported R3 GGUF). The first run records the hashes."""
    path = os.path.join(r.dir, "model_hashes.json")
    now = {v: e.get("sha256") for v, e in manifest["model_files"].items() if e.get("sha256")}
    if not os.path.exists(path):
        reg.atomic_json(path, now)
        return
    before = json.load(open(path))
    changed = sorted(v for v in now if v in before and before[v] != now[v])
    if changed and not allow:
        sys.exit(f"model files changed since this experiment's episodes were recorded: {changed}. "
                 f"Use a new experiment name, or allow_model_change=true to mix (not recommended).")
    reg.atomic_json(path, {**before, **now})


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "experiment" not in cfg:
        sys.exit("choose an experiment: +experiment=skeleton")
    exp, model, category, task_ids, specs, model_files, binaries = _plan(cfg)
    r = reg.Registry(_abs(cfg.paths.out_root), specs)
    array_task = bool(cfg.only_config)              # one Slurm array task: touch only its own files
    if not array_task:
        r.write_specs()
    s = r.summary()
    print(f"experiment {exp['name']}: {s['planned']} specs ({s['unique_run_ids']} unique), {s['done']} done, "
          f"{len(task_ids)} tasks [{task_ids[0]} .. {task_ids[-1]}]\n  dir: {r.dir}")
    for k, c in s["configs"].items():
        print(f"  {k}  {c['variant']:<12} {c['backend']:<5} {c['repair']:<5} {c['done']:>3}/{c['planned']}")
    scope = [s for s in specs if (not cfg.backend or s.backend == cfg.backend)
             and (not cfg.only_config or s.config_key == cfg.only_config)]
    problems = _preflight(scope, {v: p for v, p in model_files.items() if any(x.variant == v for x in scope)},
                          {b: p for b, p in binaries.items() if any(x.backend == b for x in scope)}, model)
    for p in problems:
        print("  WARNING:", p)

    if cfg.report_only:
        from .report import write_report
        print("report:", write_report(r.dir))
        return
    if cfg.dry_run:
        return
    # file hashes are cached in <file>.sha256 sidecars, so re-checking on every run (and on each
    # compute node) costs a stat unless the file changed
    keep = {s.variant for s in scope}
    files = {v: p for v, p in model_files.items() if v in keep}
    bins = {b: p for b, p in binaries.items() if any(x.backend == b for x in scope)}
    manifest = reg.build_manifest(specs, files, bins, model.get("sha256"), hash_files=not cfg.skip_hash)
    manifest["cpu"] = _cpu_info(cfg)
    bad = [v for v, e in manifest["model_files"].items() if e.get("sha256_matches_pilot") is False]
    if bad and not cfg.allow_sha_mismatch:
        sys.exit(f"model files differ from the pilot's SHA-256: {bad} (allow_sha_mismatch=true to override)")
    _check_model_hashes(r, manifest, cfg.allow_model_change)
    # one record per machine; manifest.json is the full-run host's (or the first host's) record
    import platform
    reg.atomic_json(os.path.join(r.dir, "manifests", f"{cfg.backend or 'all'}-{platform.node()}.json"), manifest)
    if not array_task and (not cfg.backend or not os.path.exists(os.path.join(r.dir, "manifest.json"))):
        r.write_manifest(manifest)
    if not cfg.local:
        write_slurm(cfg, r, exp["name"])
        return
    if problems:
        sys.exit("cannot run locally:\n  " + "\n  ".join(problems))
    run_local(cfg, r, specs, category, model_files, binaries)
    if array_task:
        return                                      # the dependent report job writes the report once
    if not r.pending() or cfg.report_partial:
        from .report import write_report
        print("report:", write_report(r.dir))


if __name__ == "__main__":
    main()
