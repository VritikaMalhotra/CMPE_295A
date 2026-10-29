"""C9 Cost Profiler: per-episode GPU energy and peak memory (NVML), plus TTFT and tokens/s from
llama-server's timings (already in each StepRow).

M1 measures one episode per configuration on a shared Colab GPU, so the numbers are a smoke test of
the protocol, not cost results (those need the exclusive-node protocol of M6). Energy uses NVML's
total-energy counter delta, the estimator the S6 spike validated against sampled power.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


def _visible_handle(pynvml, index: int):
    """NVML ignores CUDA_VISIBLE_DEVICES; map the process's visible GPU to its physical handle (Slurm sets it)."""
    import os
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if index < len(vis):
        v = vis[index]
        if v.startswith(("GPU-", "MIG-")):
            return pynvml.nvmlDeviceGetHandleByUUID(v)
        return pynvml.nvmlDeviceGetHandleByIndex(int(v))
    return pynvml.nvmlDeviceGetHandleByIndex(index)


class GpuMeter:
    """Context manager: ``with GpuMeter() as m: ...; m.energy_j, m.peak_mib`` (None when no NVML/GPU)."""

    def __init__(self, index: int = 0, period_s: float = 0.2):
        self.index, self.period_s = index, period_s
        self.energy_j: Optional[float] = None
        self.peak_mib: Optional[float] = None
        self._nvml = None
        self._h = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def __enter__(self) -> "GpuMeter":
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = _visible_handle(pynvml, self.index)
            self._e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)   # mJ
            self.peak_mib = 0.0
            self._t = threading.Thread(target=self._sample, daemon=True)
            self._t.start()
        except Exception:
            self._nvml = None
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                used = self._nvml.nvmlDeviceGetMemoryInfo(self._h).used / 2**20
                self.peak_mib = max(self.peak_mib or 0.0, used)
            except Exception:
                pass
            time.sleep(self.period_s)

    def __exit__(self, *exc) -> None:
        if not self._nvml:
            return
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        try:
            self.energy_j = (self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h) - self._e0) / 1000.0
        except Exception:
            self.energy_j = None
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


def step_costs(steps) -> dict:
    """TTFT and decode tokens/s summarised over a DataFrame of StepRows."""
    s = steps.dropna(subset=["decode_ms"])
    s = s[s["decode_ms"] > 0]
    tok_s = (s["completion_tokens"] / (s["decode_ms"] / 1000.0)) if len(s) else s["decode_ms"]
    return {
        "steps": int(len(steps)),
        "ttft_ms_median": float(steps["ttft_ms"].median()) if len(steps) else None,
        "decode_tok_s_median": float(tok_s.median()) if len(s) else None,
        "completion_tokens_total": int(steps["completion_tokens"].sum()),
    }
