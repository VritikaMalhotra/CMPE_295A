# Architecture: software, hardware, backends, and why each matters

This page answers feedback point 2 of 2026-11-09 (Gao: the architecture, and the significance of the software and
the hardware). It complements Workbook 2 Fig. 1 (software), Fig. 1a (infrastructure) and Fig. 3 (one agent step),
and puts our own measured numbers next to each part. The agent itself is described in `docs/agents_and_data.md`.
All numbers below are development data.

## Software: eleven components (C1–C11)

| Layer | Component | What it does | Why it matters for the study |
|---|---|---|---|
| Workload | C1 Benchmark Adapter | Loads BFCL multi-turn and τ²-bench-verified tasks and tools in one format | The same 501 tasks for every model, format and backend |
| Workload | C4 Simulated-User Service | Serves τ²'s simulated user (gpt-oss-20b) from a cluster job; caches each user turn by prompt hash | The user's side stays fixed while the agent changes; replays need no simulator |
| Execution | C2 Episode Runner | The agent loop (generate → parse → execute → observe), BFCL's 20-step turn limit | Every step is one comparable unit (the primary outcome is per step) |
| Execution | C3 Inference Backend | One `generate(messages, tools, constraint)` over llama-server, raw `/completion`, prompt cache off; the model's own prompt format | The real 4-bit kernels users run, on CUDA and CPU, from the same GGUF file |
| Execution | C7 Repairs | R1 grammar (GBNF, JSON and Pythonic), R2 extra bits (Q8_0 uniform, layer-selective R2b, step-selective switch), R3 QAD checkpoints | Each repair goes through the same loop as the damage it is meant to fix |
| Execution | C8 QAD Trainer | Quantization-aware distillation and a train-longer control, exported to GGUF | R3 is trained offline and evaluated on llama.cpp's real 4-bit path |
| Measurement | C5 Step Logger | One JSONL record per step (prompt and token hashes, raw output, calls, results, timings), Parquet after the run | Any classifier change relabels frozen logs; no rerun is needed |
| Measurement | C6 Failure Classifier | Codebook v0: 13 classes, one per step, first matching rule wins | Says *how* a step failed, not only whether the episode passed |
| Measurement | C9 Cost Profiler | TTFT, tokens/s, peak GPU memory, NVML energy | RQ4: what each repair costs to deploy |
| Analysis | C10 Analysis & Stats | Paired permutation tests, task-cluster bootstrap, lme4 mixed models; planted-effect checks | The planted checks pass (`results/m2/planted_effect.json`) |
| Control | C11 Config & Registry | Hydra configs; every episode is a run spec whose hash is its run id; Slurm array runner with requeue | Reproducible and resumable: S10 104/104 episodes with no duplicates, S7 identical losses after a cut |

## Hardware: the SJSU CoE HPC cluster (Slurm)

| Resource | What we use it for | Measured |
|---|---|---|
| P100 PCIe 12 GB (15 nodes, 1 GPU each) | Small agent runs, KL checks, S10 | S1 decode 144 tok/s (Qwen3-0.6B Q4_K_M); 182 episodes/h on one GPU (NFR-5 pilot). Qwen3-4B BF16 at ctx 40,960 does not fit (8 GB weights + 5.8 GB KV cache) |
| A100 PCIe 40 GB (4 nodes × 4 GPUs; cs001 and cs002 faulty) | Reference and damage runs; the simulated user (one GPU); QAD tests | S1 231 tok/s; a 0.6B BFCL episode in about 2.4 s (7× a P100); 1,578 episodes/h with 4 GPUs |
| H100 PCIe 80 GB (3 nodes) | QAD training (R3), the largest BF16 models | S1 350 tok/s |
| CPU nodes (`cpuqs`, up to 128 cores; `defq`, 2 CPUs per job) | The CPU backend arm, KL on CPU, analysis in R, builds | One 0.6B CPU episode takes about 11 min on 4 cores |
| Slurm, account `coe`, QoS `gpu3nodes` | Array jobs with walltime requeue (USR1 3 minutes before the limit) | **At most 2 GPU nodes per user**, so NFR-5 (≥ 8 concurrent jobs) is not met: plan 3–7 concurrent A100 jobs, or a share of the grid per member |

**Why the hardware matters.** The models are meant for laptops and phones, but the study needs about 2,500
episodes per model, format and backend: about 15,000 BF16 reference episodes for six models, before any 4-bit
cell. The A100s give the throughput. The P100s are the commodity-GPU end. The CPU nodes run the CPU kernels
users without a GPU would run.

## Backends: CUDA and CPU, same file

FR-E17 parity: 200 fixed prompts per model, replayed greedy on both backends from the same GGUF file. The prompt
token IDs were identical in 100% of prompts.

| Model | BF16: identical output / same calls | Q4_0-pure: identical output / same calls |
|---|---|---|
| Qwen3-0.6B | 90.5% / 97.5% | 53.0% / 92.0% |
| Qwen3-1.7B | 92.0% / 99.5% | 66.5% / 93.0% |
| Qwen3-4B | 96.0% / 100% | 67.0% / 95.0% |

**Why it matters.** No model reaches the 99% needed to pool, so the backend is a factor in every RQ1 comparison.
4-bit weights roughly double how often the two backends diverge. KL on CPU is also higher than on CUDA for
Q6_K-pure and Q4_K_M on 4 of 6 models (`docs/m3_results.md`). A damage map measured on one backend does not
transfer to the other.

## Memory and bits per weight per format (`results/m3/models_manifest.json`)

GGUF file size / whole-file bits per weight:

| Model | Weights | BF16 | Q8_0 | Q6_K-pure | Q5_0-pure | Q4_K_M | IQ4_XS | Q4_0-pure |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.75B | 1.41 GiB / 16.06 | 0.75 / 8.57 | 0.58 / 6.63 | 0.49 / 5.57 | 0.45 / 5.15 | 0.42 / 4.79 | 0.40 / 4.57 |
| Qwen3-1.7B | 2.03B | 3.79 / 16.02 | 2.02 / 8.52 | 1.56 / 6.59 | 1.31 / 5.53 | 1.19 / 5.05 | 1.09 / 4.63 | 1.07 / 4.53 |
| Qwen3-4B | 4.02B | 7.50 / 16.01 | 3.99 / 8.51 | 3.08 / 6.58 | 2.58 / 5.51 | 2.33 / 4.97 | 2.11 / 4.52 | 2.11 / 4.51 |
| LFM2.5-1.2B-Instruct | 1.17B | 2.18 / 16.02 | 1.16 / 8.52 | 0.90 / 6.58 | 0.75 / 5.52 | 0.68 / 5.00 | 0.62 / 4.53 | 0.62 / 4.52 |
| SmolLM3-3B | 3.08B | 5.74 / 16.02 | 3.05 / 8.52 | 2.36 / 6.58 | 1.98 / 5.52 | 1.78 / 4.98 | 1.61 / 4.48 | 1.62 / 4.52 |
| LFM2.5-2.6B | 2.70B | 5.03 / 16.03 | 2.68 / 8.53 | 2.07 / 6.59 | 1.74 / 5.53 | 1.56 / 4.97 | 1.41 / 4.50 | 1.42 / 4.53 |

(GiB / bits per weight; the first cell gives the units.) **Why it matters.** 4-bit cuts weight memory about 3.5×
against BF16. A 4B model then fits in about 2 GiB, which is phone-sized. Q8_0 is about 1.9× Q4_0, the price of
the R2 uniform repair. The R2b layer-selective variant targets 5.5 bits per weight (FR-E10).
