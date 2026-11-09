"""C4 simulated user as its own Slurm job (decision 4, Ekant 2026-11-08; advisor to confirm).

One GPU job (A100/H100) runs the simulator's llama-server for a whole experiment and announces itself in an
endpoint file on /scratch; the agent jobs read that file and send their cache misses there instead of starting
their own simulator. Each agent job still keeps its own C4 cache (testbed/simuser_cache.py), so replays never
need this job.

* ``serve``             - start llama-server (listening on all interfaces), write the endpoint file, keep it
                          alive; the file is removed when the job stops.
* ``wait_for_endpoint`` - agent side: wait for the file, check it serves the experiment's simulator (model
                          file, SHA-256, context), check /health, return the URL.

    python -m testbed.run +experiment=<exp> simuser_serve=true simuser_endpoint=<file>   # the simulator job
    TESTBED_SIMUSER_ENDPOINT=<file> python -m testbed.run +experiment=<exp> local=true  # an agent job
"""
from __future__ import annotations

import json
import os
import socket
import time
from typing import Optional

import requests

ENDPOINT_VERSION = 1
IDENTITY = ("model", "file", "sha256")


def endpoint_record(sim: dict, url: str, ctx: int, parallel: int) -> dict:
    return {"version": ENDPOINT_VERSION, "url": url, "ctx_per_slot": ctx, "parallel": parallel,
            **{k: sim.get(k) for k in IDENTITY},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def write_endpoint(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)


def remove_endpoint(path: str, pid: int) -> None:
    """Remove the file only if this process wrote it (a newer simulator job may have replaced it)."""
    try:
        if json.load(open(path)).get("pid") == pid:
            os.remove(path)
    except (OSError, ValueError):
        pass


def check_record(rec: dict, sim: dict) -> list[str]:
    """Reasons the endpoint does not serve this experiment's simulator ([] = it does)."""
    bad = [f"{k}: endpoint {rec.get(k)!r}, experiment {sim.get(k)!r}" for k in IDENTITY if rec.get(k) != sim.get(k)]
    if int(rec.get("ctx_per_slot") or 0) != int(sim.get("ctx", 32768)):
        bad.append(f"ctx: endpoint {rec.get('ctx_per_slot')}, experiment {sim.get('ctx', 32768)}")
    if rec.get("version") != ENDPOINT_VERSION:
        bad.append(f"endpoint file version {rec.get('version')}")
    return bad


def wait_for_endpoint(path: str, sim: dict, timeout_s: float = 1800, poll_s: float = 10) -> str:
    """URL of a live simulator that serves ``sim``. Raises RuntimeError after ``timeout_s`` (the runner then
    leaves the tau2 episodes pending) or at once if the file names a different simulator."""
    deadline = time.time() + timeout_s
    last = "endpoint file not written yet"
    while True:
        rec: Optional[dict] = None
        try:
            rec = json.load(open(path))
        except (OSError, ValueError):
            pass
        if rec is not None:
            bad = check_record(rec, sim)
            if bad:
                raise RuntimeError(f"{path} serves a different simulator: " + "; ".join(bad))
            try:
                if requests.get(rec["url"] + "/health", timeout=5).status_code == 200:
                    return rec["url"]
                last = f"{rec['url']} not healthy yet"
            except requests.RequestException as e:
                last = f"{rec['url']} unreachable ({type(e).__name__})"
        if time.time() >= deadline:
            raise RuntimeError(f"no simulated user at {path} after {timeout_s:.0f}s: {last}")
        time.sleep(poll_s)


def serve(sim: dict, model_path: str, binary: str, endpoint: str, log_path: str, parallel: int = 4,
          check_s: float = 30) -> None:
    """Run until the job is stopped (SIGTERM -> SystemExit in testbed.run.main) or llama-server dies."""
    from .backends import LlamaServer, ServerConfig
    ctx = int(sim.get("ctx", 32768))
    scfg = ServerConfig(binary=binary, model_path=model_path, backend="cuda", ctx=ctx * parallel, seed=0,
                        host="0.0.0.0", parallel=parallel, extra_args=("--jinja",))
    with LlamaServer(scfg, log_path=log_path, startup_timeout_s=900) as server:
        rec = endpoint_record(sim, f"http://{socket.gethostname()}:{server.port}", ctx, parallel)
        write_endpoint(endpoint, rec)
        print(f"simulated user serving at {rec['url']} ({parallel} slots x {ctx} ctx); endpoint {endpoint}",
              flush=True)
        try:
            while server.proc.poll() is None:
                time.sleep(check_s)
            raise RuntimeError(f"llama-server exited with {server.proc.returncode}; see {log_path}")
        finally:
            remove_endpoint(endpoint, rec["pid"])
