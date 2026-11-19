#!/usr/bin/env python3
"""Write the confirmatory holdout experiment configs (docs/preregistration_osf.md §3, stages C1 and C2).

Model keys, Hugging Face repositories and GGUF file names are read from the development configs
(configs/experiment/ref_<key>.yaml and h1_<key>_<variant>.yaml), so the holdout runs use byte-identical model files.
Everything else follows the registration: the 300-task BFCL holdout (results/m3/holdout_tasks_bfcl.json, checked by
SHA-256 in testbed.run.load_task_ids_file), sampled seeds 5-8 (T 0.7, top_p 0.8, top_k 20) plus a greedy pass with
seed 0, ctx 40,960 / 4,096-token cap, no tau2 and no simulated user.

* C1: hold_ref_<key> (BF16, CUDA) and hold_h1_<key> (Q4_0-pure, Q4_K_M, IQ4_XS, CUDA) for all six models;
* C2: hold_cpu_<key> for Qwen3-0.6B and 1.7B: BF16 and the three formats on the CPU backend, greedy only,
  multi_turn_base holdout (decision 13).

    python3 scripts/make_holdout_configs.py          # writes configs/experiment/hold_*.yaml
"""
from __future__ import annotations

import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "configs", "experiment")
KEYS = ["qwen3_0p6b", "qwen3_1p7b", "qwen3_4b", "smollm3_3b", "lfm2p5_1p2b", "lfm2p5_2p6b"]
FORMATS = ["q4_0", "q4_k_m", "iq4_xs"]
CPU_KEYS = ["qwen3_0p6b", "qwen3_1p7b"]
TASKS = "results/m3/holdout_tasks_bfcl.json"
SEEDS = [5, 6, 7, 8]
DECODING = "{temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, n_predict: 4096, cache_prompt: false}"
GREEDY = "{temperature: 0.0, top_k: 1}"


def model_block(key: str, variants: list[str]) -> tuple[str, str]:
    ref = yaml.safe_load(open(os.path.join(EXP, f"ref_{key}.yaml")))
    name = ref["experiment"]["model"]
    files = dict(ref["models"][name]["files"])
    for v in variants:
        if v != "bf16":
            files.update(yaml.safe_load(open(os.path.join(EXP, f"h1_{key}_{v}.yaml")))["models"][name]["files"])
    files = {v: files[v] for v in variants}
    body = (f"models:\n  {name}:\n    hf_repo: {ref['models'][name]['hf_repo']}\n"
            f"    files: {{{', '.join(f'{v}: {f}' for v, f in files.items())}}}\n")
    return name, body


def write(path: str, header: str, name: str, model: str, models: str, baselines: list[str], extras: bool,
          time: str) -> None:
    extra = ("  extra_benchmarks:\n    - {name: bfcl_multi_turn_miss_func}\n    - {name: bfcl_multi_turn_miss_param}\n"
             if extras else "  extra_benchmarks: []\n")
    text = (f"# @package _global_\n{header}experiment:\n  name: {name}\n  model: {model}\n"
            f"  benchmark: bfcl_multi_turn_base\n  task_ids_file: {TASKS}   # registered holdout, SHA-256 checked\n"
            f"  n_tasks: 100\n  thinking: false\n  max_steps_per_turn: 20\n  baselines:\n"
            + "".join(f"    - {b}\n" for b in baselines)
            + f"  baseline_seeds: {SEEDS}\n  repairs: []\n  repair_seeds: [5]\n  decoding: {DECODING}\n"
            f"  server: {{ctx: 40960}}\n{extra}slurm:\n  time: \"{time}\"\n  max_requeue: 3\n{models}")
    open(path, "w").write(text)
    print("wrote", os.path.relpath(path, REPO))


def main() -> None:
    reg = "# Confirmatory (docs/preregistration_osf.md). Do not submit before the registration is filed (§9).\n"
    for key in KEYS:
        name, models = model_block(key, ["bf16"])
        write(os.path.join(EXP, f"hold_ref_{key}.yaml"),
              reg + f"# Stage C1: BF16 reference for {name} on the 300-task BFCL holdout, CUDA; seeds 5-8 + greedy (seed 0).\n"
              "# 1,500 specs. Inclusion set for the secondary success analysis afterwards (§5).\n",
              f"hold_ref_{key}", name, models,
              ["{variant: bf16, backend: cuda}",
               f"{{variant: bf16, backend: cuda, decoding: {GREEDY}, seeds: [0]}}   # greedy pass"], True, "1-00:00:00")
        name, models = model_block(key, FORMATS)
        base = []
        for v in FORMATS:
            base += [f"{{variant: {v}, backend: cuda}}",
                     f"{{variant: {v}, backend: cuda, decoding: {GREEDY}, seeds: [0]}}   # greedy pass"]
        write(os.path.join(EXP, f"hold_h1_{key}.yaml"),
              reg + f"# Stage C1: H1 damage cells for {name}: Q4_0-pure, Q4_K_M, IQ4_XS on the BFCL holdout, CUDA.\n"
              f"# Pairs with hold_ref_{key} by (task, seed). 4,500 specs.\n",
              f"hold_h1_{key}", name, models, base, True, "1-00:00:00")
    for key in CPU_KEYS:
        variants = ["bf16"] + FORMATS
        name, models = model_block(key, variants)
        base = [f"{{variant: {v}, backend: cpu, decoding: {GREEDY}, seeds: [0]}}" for v in variants]
        write(os.path.join(EXP, f"hold_cpu_{key}.yaml"),
              reg + f"# Stage C2 (decision 13): {name} on the CPU backend, BF16 and three 4-bit formats, greedy only,\n"
              "# multi_turn_base holdout (100 tasks). 400 specs. Run on cpuqs (about 11 min per episode on 4 cores).\n",
              f"hold_cpu_{key}", name, models, base, False, "5-00:00:00")


if __name__ == "__main__":
    main()
