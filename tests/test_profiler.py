"""C9 profiler: NVML device mapping under Slurm, missing NVML, energy units (design spec 7.2).

A fake ``pynvml`` module is put in ``sys.modules``, so no GPU is needed.
"""
import sys
import time
import types

import pandas as pd
import pytest

from testbed.profiler import GpuMeter, _visible_handle, step_costs


def _fake_nvml(energy=(1000, 6000), used_mib=512):
    m = types.ModuleType("pynvml")
    m.calls = []
    e = iter(energy)
    m.nvmlInit = lambda: m.calls.append("init")
    m.nvmlShutdown = lambda: m.calls.append("shutdown")
    m.nvmlDeviceGetHandleByUUID = lambda u: m.calls.append(("uuid", u)) or ("h", u)
    m.nvmlDeviceGetHandleByIndex = lambda i: m.calls.append(("index", i)) or ("h", i)
    m.nvmlDeviceGetTotalEnergyConsumption = lambda h: next(e)
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=used_mib * 2**20)
    return m


def test_visible_device_mapping(monkeypatch):
    nv = _fake_nvml()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234-abcd,3")
    assert _visible_handle(nv, 0) == ("h", "GPU-1234-abcd")
    assert _visible_handle(nv, 1) == ("h", 3)                     # physical index 3, not 1
    assert nv.calls == [("uuid", "GPU-1234-abcd"), ("index", 3)]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert _visible_handle(nv, 2) == ("h", 2)                     # no Slurm: index as given


def test_no_nvml_gives_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)              # import pynvml -> ImportError
    with GpuMeter() as m:
        pass
    assert m.energy_j is None and m.peak_mib is None


def test_energy_delta_units(monkeypatch):
    nv = _fake_nvml(energy=(1000, 6000), used_mib=512)
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with GpuMeter(period_s=0.01) as m:
        time.sleep(0.1)                                           # let the sampler thread take a reading
    assert m.energy_j == pytest.approx(5.0)                       # mJ counter -> J
    assert m.peak_mib == pytest.approx(512.0)
    assert nv.calls[-1] == "shutdown"


def test_step_costs():
    s = pd.DataFrame({"ttft_ms": [10.0, 30.0, 20.0], "decode_ms": [1000.0, 500.0, None],
                      "completion_tokens": [100, 100, 7]})
    c = step_costs(s)
    assert c["steps"] == 3 and c["ttft_ms_median"] == 20.0
    assert c["decode_tok_s_median"] == pytest.approx(150.0)      # median of 100 and 200 tok/s
    assert c["completion_tokens_total"] == 207
