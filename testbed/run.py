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
import signal
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
    tasks = {exp["benchmark"]: task_ids}
    for extra in exp.get("extra_benchmarks", []) or []:
        tasks[extra["name"]] = [str(t) for t in extra["task_ids"]]
    specs = reg.expand_all(exp, tasks)
    model_files = {v: os.path.join(_abs(cfg.paths.models_dir), f) for v, f in model["files"].items()
                   if any(s.variant == v for s in specs)}
    binaries = {"cuda": _abs(cfg.paths.llama_cuda), "cpu": _abs(cfg.paths.llama_cpu)}
    need = {s.backend for s in specs} | ({"cuda"} if _simuser(specs) else set())   # the simulator runs on CUDA
    binaries = {k: v for k, v in binaries.items() if k in need}
    return exp, model, category, task_ids, specs, model_files, binaries


def _simuser(specs) -> dict:
    """The simulated-user settings of the experiment's tau2 specs ({} if none)."""
    for s in specs:
        if s.server.get("simuser"):
            return dict(s.server["simuser"])
    return {}


def _sim_file(cfg, sim: dict) -> str:
    return os.path.join(_abs(cfg.paths.models_dir), sim["file"])


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


class SimStack:
    """C4 for one run: the simulator's llama-server (CUDA) + the caching proxy in front of it.
    The SQLite store lives on local disk while running (SQLite on a Drive mount is unsafe) and is copied
    into the experiment directory after every episode."""

    def __init__(self, cfg, r: reg.Registry, sim: dict, binary: str, mode: str = "live"):
        from .simuser_cache import CacheProxy, SimUserCache
        self.dir_copy = os.path.join(r.dir, "simuser_cache.sqlite")
        local = cfg.paths.get("simuser_cache_local") or None
        self.db = local or self.dir_copy
        if local and not os.path.exists(local) and os.path.exists(self.dir_copy):
            import shutil
            shutil.copyfile(self.dir_copy, local)
        self.server = None
        upstream = None
        if mode == "live":
            scfg = ServerConfig(binary=binary, model_path=_sim_file(cfg, sim), backend="cuda",
                                ctx=int(sim.get("ctx", 32768)), seed=0, extra_args=("--jinja",))
            logs = os.path.join(r.dir, "server_logs")
            os.makedirs(logs, exist_ok=True)
            self.server = LlamaServer(scfg, log_path=os.path.join(logs, "simuser.log"), startup_timeout_s=900).__enter__()
            upstream = self.server.base
        identity = {k: sim.get(k) for k in ("model", "file", "sha256", "ctx", "temperature")}
        inject = {"chat_template_kwargs": sim["chat_template_kwargs"]} if sim.get("chat_template_kwargs") else {}
        self.cache = SimUserCache(self.db, identity, upstream=upstream, mode=mode, inject=inject,
                                  max_upstream_hours=cfg.get("simuser_max_hours"))
        self.proxy = CacheProxy(self.cache).__enter__()
        self.base_url = self.proxy.base_url

    def sync(self) -> None:
        if self.db != self.dir_copy:
            self.cache.backup_to(self.dir_copy)

    def close(self) -> None:
        self.sync()
        self.proxy.__exit__(None, None, None)
        self.cache.close()
        if self.server:
            self.server.__exit__(None, None, None)


def run_local(cfg, r: reg.Registry, specs, category, model_files, binaries) -> None:
    pending = r.pending()
    if cfg.only_config:
        pending = [s for s in pending if s.config_key == cfg.only_config]
    if cfg.backend:
        pending = [s for s in pending if s.backend == cfg.backend]
    if cfg.only_benchmark:
        pending = [s for s in pending if s.benchmark == cfg.only_benchmark]
    if cfg.limit:
        pending = pending[: int(cfg.limit)]
    if not pending:
        print("nothing pending")
        return
    tasks = {(t.benchmark, t.id): t for t in load_bfcl_multi_turn(
        category, ids=sorted({s.task_id for s in pending if s.benchmark.startswith("bfcl")}))}
    tau2_ids = sorted({s.task_id for s in pending if s.benchmark.startswith("tau2")}, key=int)
    if tau2_ids:
        from .tau2_bench import load_tau2_tasks
        t2 = next(s.server["tau2"] for s in pending if s.benchmark.startswith("tau2"))
        tasks.update({(t.benchmark, t.id): t for t in load_tau2_tasks(tau2_ids, t2.get("split", "base"),
                                                                       t2.get("write_tools"))})
    sim: SimStack | None = None
    sim_down = False
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
        is_tau2 = s0.benchmark.startswith("tau2")
        if is_tau2 and sim is None and not sim_down:
            try:
                sim = SimStack(cfg, r, dict(s0.server["simuser"]), binaries["cuda"], mode=cfg.simuser_mode)
                print(f"simulated user up ({cfg.simuser_mode}): {sim.base_url}", flush=True)
            except Exception as e:
                print(f"simulated user failed to start: {e}; tau2 episodes stay pending", flush=True)
                sim_down = True
        if is_tau2 and sim_down:
            lock.close()
            continue
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
                        if is_tau2:
                            from .tau2_bench import run_tau2_episode
                            ep = run_tau2_episode(s, tasks[(s.benchmark, s.task_id)], be, gfn, log,
                                                  api_base=sim.base_url, cache=sim.cache,
                                                  sim_dir=os.path.join(r.dir, "tau2_sims"))
                        else:
                            ep = run_episode(s, tasks[(s.benchmark, s.task_id)], be, gfn, log)
                    ep.energy_j, ep.peak_gpu_mib = m.energy_j, m.peak_mib
                    log.finish(ep)
                    if is_tau2:
                        sim.sync()
                    n_done += 1
                    print(f"  {s.task_id} seed={s.seed} valid={ep.valid} steps={ep.n_steps} "
                          f"{ep.wall_s:.1f}s  ({n_done}/{len(pending)})", flush=True)
                except KeyboardInterrupt:
                    log.abort()
                    raise
                except Exception as e:
                    if type(e).__name__ == "SimulatorUnavailable":
                        log.abort()
                        print(f"  {s.task_id} seed={s.seed}: simulated user unavailable ({e}); "
                              "stopping tau2 scheduling, episodes stay pending", flush=True)
                        sim_down = True
                        break
                    log.abort()
                    print(f"  {s.task_id} seed={s.seed} FAILED (will retry on next run)\n"
                          + traceback.format_exc(limit=3), flush=True)
        finally:
            srv.__exit__(None, None, None)
            lock.close()                                # releases the flock
    if sim is not None:
        print("simulated-user cache:", json.dumps(sim.cache.stats()["by_role"]), flush=True)
        sim.close()
    print(f"finished {n_done} episodes in {time.time() - t_all:.0f}s")


def replay_one(cfg, r: reg.Registry, specs, category, model_files, binaries, run_id: str) -> dict:
    """FR-E6 check: re-run one finished tau2 episode with the simulated user in replay mode (cache only,
    no simulator process). Passes if every simulator request is a cache hit and the reward matches."""
    import shutil
    from .simuser_cache import SimUserCache
    spec = next((s for s in specs if s.run_id == run_id), None)
    if spec is None or not spec.benchmark.startswith("tau2"):
        sys.exit(f"{run_id} is not a tau2 spec of this experiment")
    orig = os.path.join(r.dir, "episodes", f"{run_id}.json")
    if not os.path.exists(orig):
        sys.exit(f"{run_id} has not been run yet")
    scratch = reg.Registry(os.path.join(r.dir, "replays"), [spec])
    for d in ("episodes", "steps"):
        p = os.path.join(scratch.dir, d, f"{run_id}.json" + ("l" if d == "steps" else ""))
        if os.path.exists(p):
            os.remove(p)
    src_db = os.path.join(r.dir, "simuser_cache.sqlite")
    shutil.copyfile(src_db, os.path.join(scratch.dir, "simuser_cache.sqlite"))
    before = SimUserCache(os.path.join(scratch.dir, "simuser_cache.sqlite"), {}, mode="replay").stats()
    sub = OmegaConf.merge(cfg, {"simuser_mode": "replay", "replay": None, "only_config": None, "limit": None,
                                "backend": None, "only_benchmark": None,
                                "paths": {"simuser_cache_local": None}})
    run_local(sub, scratch, [spec], category, model_files, binaries)
    after = SimUserCache(os.path.join(scratch.dir, "simuser_cache.sqlite"), {}, mode="replay").stats()
    def n(st, outcome):
        return sum(v.get(outcome, {}).get("n", 0) for v in st["by_role"].values())
    out_p = os.path.join(scratch.dir, "episodes", f"{run_id}.json")
    res = {"run_id": run_id, "hits": n(after, "hit") - n(before, "hit"),
           "misses": n(after, "replay_miss") - n(before, "replay_miss") + n(after, "miss") - n(before, "miss"),
           "finished": os.path.exists(out_p)}
    if res["finished"]:
        a, b = json.load(open(orig)), json.load(open(out_p))
        res.update(reward_matches=a["valid"] == b["valid"], steps_match=a["n_steps"] == b["n_steps"])
    res["pass"] = bool(res["finished"] and res["misses"] == 0 and res.get("reward_matches"))
    reg.atomic_json(os.path.join(r.dir, "replays", f"{run_id}.result.json"), res)
    print("replay:", json.dumps(res))
    return res


def _tau2_version() -> dict:
    try:
        from importlib.metadata import version
        import tau2
        d = os.path.dirname(os.path.dirname(os.path.dirname(tau2.__file__)))
        import subprocess
        commit = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        return {"version": version("tau2"), "path": d, "commit": commit or "unknown"}
    except Exception as e:
        return {"version": f"unavailable: {type(e).__name__}"}


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
            n=len(keys) - 1, max_conc=sb.max_concurrent, cfg_list=cfg_list,
            max_requeue=sb.get("max_requeue", 5), **common))
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
    # Slurm stops a job with SIGTERM (walltime, requeue, scancel). Turn it into SystemExit so every context
    # manager runs: llama-server and the simulator are stopped and the C4 store is copied back. Finished
    # episodes are already on disk (atomic writes); the one in flight is redone on resume.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if "experiment" not in cfg:
        sys.exit("choose an experiment: +experiment=skeleton")
    exp, model, category, task_ids, specs, model_files, binaries = _plan(cfg)
    r = reg.Registry(_abs(cfg.paths.out_root), specs)
    array_task = bool(cfg.only_config)              # one Slurm array task: touch only its own files
    if not array_task:
        r.write_specs()
    s = r.summary()
    print(f"experiment {exp['name']}: {s['planned']} specs ({s['unique_run_ids']} unique), {s['done']} done, "
          f"{len({x.task_id for x in specs})} tasks over {', '.join(sorted({x.benchmark for x in specs}))}\n  dir: {r.dir}")
    for k, c in s["configs"].items():
        print(f"  {k}  {c.get('benchmark', ''):<22} {c['variant']:<12} {c['backend']:<5} {c['repair']:<5} "
              f"{c['done']:>3}/{c['planned']}")
    scope = [s for s in specs if (not cfg.backend or s.backend == cfg.backend)
             and (not cfg.only_config or s.config_key == cfg.only_config)]
    if cfg.only_benchmark:
        scope = [s for s in scope if s.benchmark == cfg.only_benchmark]
    sim = _simuser(scope)
    scope_bins = {b: p for b, p in binaries.items() if any(x.backend == b for x in scope) or (sim and b == "cuda")}
    problems = _preflight(scope, {v: p for v, p in model_files.items() if any(x.variant == v for x in scope)},
                          scope_bins, model)
    if sim and cfg.simuser_mode == "live" and not os.path.exists(_sim_file(cfg, sim)):
        problems.append(f"missing simulated-user model: {_sim_file(cfg, sim)}")
    for p in problems:
        print("  WARNING:", p)

    if cfg.import_from and not array_task:
        n = r.import_from(_abs(cfg.import_from))
        print(f"imported {n} finished episodes from {cfg.import_from}")
    if cfg.replay:
        replay_one(cfg, r, specs, category, model_files, binaries, str(cfg.replay))
        return
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
    expected = dict(model.get("sha256") or {})
    if sim and cfg.simuser_mode == "live":
        files["simuser"] = _sim_file(cfg, sim)
        expected["simuser"] = sim.get("sha256")
    bins = scope_bins
    manifest = reg.build_manifest(specs, files, bins, expected, hash_files=not cfg.skip_hash)
    if sim:
        manifest["simuser"] = sim
        manifest["tau2"] = _tau2_version()
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
