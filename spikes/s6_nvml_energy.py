"""Spike S6: NVML power, memory and SM clock sampled every 200 ms around one generation,
10 repeats per format, idle baseline subtracted, energy per decoded token.

Pilot run: Google Colab A100-SXM4-40GB (shared VM, no exclusive allocation), llama.cpp v0.6.0,
2026-10-08 PT. Pilot / development data, not study results.

Workload: the S1 tool-call prompt (205 token IDs) and 1,024 decoded tokens (ignore_eos), greedy.
Two estimators per repeat:
  counter: (total-energy counter delta) - idle_power * duration, divided by tokens
  sampled: (mean sampled power - idle_power) * duration, divided by tokens

    python s6_nvml_energy.py --gguf models/Qwen3-0.6B-Q4_K_M.gguf --prompt-ids prompt.json --out nvml
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import threading
import time

import pandas as pd
import pynvml
import requests

LLAMA = "/content/work/llama.cpp"
PORT = 18080


class Sampler:
    def __init__(self, handle, phase, rep, dt=0.2):
        self.h, self.phase, self.rep, self.dt = handle, phase, rep, dt
        self.rows, self._stop = [], False

    def _loop(self):
        while not self._stop:
            self.rows.append({
                "t": time.time(), "phase": self.phase, "rep": self.rep,
                "power_mw": pynvml.nvmlDeviceGetPowerUsage(self.h),
                "energy_mj": pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h),
                "mem_used_mib": pynvml.nvmlDeviceGetMemoryInfo(self.h).used // 2**20,
                "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
            })
            time.sleep(self.dt)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join()


def serve(gguf):
    p = subprocess.Popen(f"{LLAMA}/build-cuda/bin/llama-server -m {gguf} -ngl 99 -c 8192 -np 1 "
                         f"--host 127.0.0.1 --port {PORT} --seed 0 --no-webui", shell=True,
                         preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                return p
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not start")


def measure(gguf, prompt_ids, reps=10, idle_s=5.0):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    body = {"prompt": prompt_ids, "n_predict": 1024, "ignore_eos": True, "temperature": 0.0,
            "top_k": 1, "seed": 0, "cache_prompt": False}
    p = serve(gguf)
    requests.post(f"http://127.0.0.1:{PORT}/completion", json={**body, "n_predict": 64})  # warm-up
    time.sleep(3)
    samples, summary = [], []
    for rep in range(reps):
        with Sampler(h, "idle", rep) as idle:
            time.sleep(idle_s)
        e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.time()
        with Sampler(h, "gen", rep) as gen:
            r = requests.post(f"http://127.0.0.1:{PORT}/completion", json=body).json()
        t1, e1 = time.time(), pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        samples += idle.rows + gen.rows
        idle_w = statistics.mean(x["power_mw"] for x in idle.rows) / 1000
        gen_w = statistics.mean(x["power_mw"] for x in gen.rows) / 1000
        dur, ntok = t1 - t0, r["timings"]["predicted_n"]
        summary.append({"rep": rep, "tokens": ntok, "gen_s": dur, "decode_tok_s": r["timings"]["predicted_per_second"],
                        "idle_w": idle_w, "gen_w_mean": gen_w,
                        "mj_per_tok_counter": ((e1 - e0) - idle_w * dur * 1000) / ntok,
                        "mj_per_tok_sampled": (gen_w - idle_w) * dur * 1000 / ntok,
                        "sm_clock_min": min(x["sm_clock_mhz"] for x in gen.rows),
                        "sm_clock_max": max(x["sm_clock_mhz"] for x in gen.rows)})
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    p.wait()
    return pd.DataFrame(samples), pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-ids", required=True, help="JSON list of prompt token IDs")
    ap.add_argument("--out", default="nvml")
    a = ap.parse_args()
    samples, summary = measure(a.gguf, json.load(open(a.prompt_ids)))
    samples.to_csv(f"{a.out}_samples.csv", index=False)
    summary.to_csv(f"{a.out}_summary.csv", index=False)
    print(summary[["mj_per_tok_counter", "mj_per_tok_sampled", "decode_tok_s"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
