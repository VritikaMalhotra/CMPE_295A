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
* C3: hold_c3_<key> for the three core models: the repairs of pre-registration §2 from Q4_0, each with seeds 5-8
  plus a greedy pass, CUDA: R1 (free GBNF grammar on Q4_0), R2a Q5_0 / Q6_K / Q8_0, R2b (layer-selective 5.5 bpw),
  R2c (Q4_0 and Q8_0 resident, tool-call steps routed to Q8_0 by the first generated token; testbed.backends.
  RoutingBackend), R3 QAD-KL, QAD-KL+CE and the train-longer control. Pairs with hold_ref_<key> and hold_h1_<key>
  (Q4_0) by (task, seed).

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
CORE_KEYS = ["qwen3_0p6b", "qwen3_1p7b", "lfm2p5_1p2b"]
# C3 repairs: (repair label, variant key, constraint, GGUF suffix after "<Model>-")
C3 = [("r1", "q4_0", "free", None),                       # file from the h1 config
      ("r2", "q5_0", "none", "Q5_0-pure.gguf"),
      ("r2", "q6_k", "none", "Q6_K-pure.gguf"),
      ("r2", "q8_0", "none", "Q8_0.gguf"),
      ("r2", "r2b", "none", "R2b-LS5.5.gguf"),
      ("r2", "r2c_tool", "none", "Q4_0-pure.gguf"),       # the low copy; the route names the high copy
      ("r3", "r3_qad_kl", "none", "R3-QAD-KL-Q4_0-pure.gguf"),
      ("r3", "r3_qad_klce", "none", "R3-QAD-KLCE-Q4_0-pure.gguf"),
      ("r3", "r3_ctrl", "none", "R3-CTRL-Q4_0-pure.gguf")]
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


def write_c3(path: str, header: str, name: str, model: str, models: str, repairs: list[str]) -> None:
    text = (f"# @package _global_\n{header}experiment:\n  name: {name}\n  model: {model}\n"
            f"  benchmark: bfcl_multi_turn_base\n  task_ids_file: {TASKS}   # registered holdout, SHA-256 checked\n"
            f"  n_tasks: 100\n  thinking: false\n  max_steps_per_turn: 20\n  baselines: []\n"
            f"  baseline_seeds: {SEEDS}\n  repairs:\n" + "".join(f"    - {r}\n" for r in repairs)
            + f"  repair_seeds: {SEEDS}\n  decoding: {DECODING}\n  server: {{ctx: 40960}}\n"
            "  extra_benchmarks:\n    - {name: bfcl_multi_turn_miss_func}\n    - {name: bfcl_multi_turn_miss_param}\n"
            f"slurm:\n  time: \"1-00:00:00\"\n  max_requeue: 3\n{models}")
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
    for key in CORE_KEYS:
        name, models = model_block(key, ["q4_0"])
        q4 = yaml.safe_load(models)["models"][name]["files"]["q4_0"]
        stem = q4[: -len("-Q4_0-pure.gguf")]
        files = {"q4_0": q4} | {v: f"{stem}-{suf}" for _, v, _, suf in C3 if suf}
        ref = yaml.safe_load(open(os.path.join(EXP, f"ref_{key}.yaml")))
        models = (f"models:\n  {name}:\n    hf_repo: {ref['models'][name]['hf_repo']}\n"
                  f"    files: {{{', '.join(f'{v}: {f}' for v, f in files.items())}}}\n")
        reps = []
        for rep, v, con, _ in C3:
            srv = ", server: {route: {policy: tool_call, high: q8_0}}" if v == "r2c_tool" else ""
            reps += [f"{{repair: {rep}, variant: {v}, backend: cuda, constraint: {con}{srv}}}",
                     f"{{repair: {rep}, variant: {v}, backend: cuda, constraint: {con}{srv}, decoding: {GREEDY}, seeds: [0]}}   # greedy pass"]
        write_c3(os.path.join(EXP, f"hold_c3_{key}.yaml"),
                 reg + f"# Stage C3: repairs for {name} on the 300-task BFCL holdout, CUDA, from Q4_0 (pre-registration §2, §6).\n"
                 f"# 9 repaired configurations x (4 sampled + 1 greedy) x 300 tasks = 13,500 specs.\n"
                 f"# Pairs with hold_ref_{key} (BF16) and hold_h1_{key} (Q4_0) by (task, seed).\n",
                 f"hold_c3_{key}", name, models, reps)
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
