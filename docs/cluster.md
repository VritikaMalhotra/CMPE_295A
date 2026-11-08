# CoE HPC (coe-hpc2): limits and runbook

Measured 2026-10-26 under account `coe` (onboarding step 1, Workbook Table 37b). Every member checks their own
login with the smoke test below and adds a line to "Checked accounts".

## Facts
| | |
|---|---|
| Login | `coe-hpc2.sjsu.edu`, SJSU VPN or campus network only; SSH key login (see each member's `~/.ssh/config`) |
| Slurm account | `coe` (`-A coe`), QOS `normal`, no per-user job caps |
| GPU partitions | `gpuqs` (max 2 d 1 h), `gpuqm` (max 7 d 1 h); same GPU nodes |
| GPUs | 15 x P100 PCIe 12 GB (`gpu:p100:1`), 4 nodes x 4 A100 PCIe 40 GB (`gpu:a100:N`), 3 x H100 PCIe 80 GB (`gpu:h100:1`); drivers differ by node (575.57 on P100 node g2, 590.44 on an H100 node), so every job logs it |
| CPU partitions | `defq` (6 h), `cpuqs` (5 d 1 h), `cpuqm` (14 d 1 h), `cpuql` (21 d 1 h); 24 nodes |
| Default walltime | none: every job sets `-t` |
| Modules | `nvhpc-hpcx-cuda12/24.11` (NVIDIA HPC SDK 24.11: nvcc from CUDA 12.6, targets sm_50 to sm_90; it also sets CC/CXX to nvc/nvc++, which `scripts/hpc_env.sh` undoes), `nvhpc-hpcx-cuda11/24.11`, `cmake/3.24.2`, `python3/3.12.12`; system gcc 11.5, git, apptainer. Load modules in batch jobs via `scripts/hpc_env.sh` (non-login shells have no `module`) |
| Network | login node: Hugging Face, GitHub and PyPI reachable. **Compute nodes: no outbound access** (all three time out) |
| Storage | home is on a file system at 100%: config only. Group folder `/scratch/cmpe295-bertin` (12 TB free, group `cmpe295-bertin`, setgid): `models/ envs/ src/ runs/ cache/ logs/`. `/data/cmpe295-bertin` (3.8 TB free) for long-lived copies |
| GPU isolation | Slurm sets `CUDA_VISIBLE_DEVICES`, but `nvidia-smi` in a 1-GPU A100 job lists all four GPUs: use only the assigned device, never explicit IDs; every job logs it (`log_gpu`) |

| Known bad nodes | **cs001** (A100, 2026-11-03): GPUs 1–3 fail CUDA initialisation, GPU 0 works. **cs002** (A100): CUDA context creation fails (`CUDA_ERROR_UNKNOWN` from PyTorch, "device busy or unavailable" from llama.cpp) on 2026-10-26 while nvidia-smi looks normal. Add `--exclude=cs001,cs002` to A100 jobs until the administrator fixes them |
| CPU pinning | none either: llama.cpp started 28 threads in a 4-CPU job. `hpc_env.sh` sets `TESTBED_CPU_THREADS=$SLURM_CPUS_PER_TASK`, which `cpu_threads` in configs/config.yaml reads |

Consequences for the plan:
- Downloads (models, Python packages, llama.cpp and tau2 source) run on the login node; builds, conversions
  and runs run in jobs. `scripts/setup_env.sh` does this split by itself on the cluster.
- The τ² simulated user cannot be reached on an outside VM from a compute node (DEP-10): it runs on the
  cluster (an A100 or H100 job), or the τ² runs stay on the GCP/Colab host. Decision for the 2026-10-30 review.
- One llama.cpp CUDA build serves all three GPU types (`CUDA_ARCHS="60;80;90"`); the CPU code is built
  without `-march=native` so it runs on every node type.

## Runbook (from nothing to spike S1, about 1 hour, mostly waiting)
On the login node, in your own clone under the group folder:
```bash
mkdir -p /scratch/cmpe295-bertin/src/$USER && cd /scratch/cmpe295-bertin/src/$USER
git clone -b main https://github.com/VritikaMalhotra/CMPE_295A.git && cd CMPE_295A   # M2 work: -b m2
source scripts/hpc_env.sh                          # paths, caches, modules (every new shell)
bash scripts/setup_env.sh --llama --cuda           # login node: Python env + llama.cpp source, no compiling
uv pip install -q --python .venv/bin/python -e ".[train]"   # converter deps (torch, transformers)
DOWNLOAD_ONLY=1 bash scripts/make_probe_models.sh Qwen/Qwen3-0.6B $TESTBED_MODELS
```
Then the jobs (each waits for the one before):
```bash
B=$(sbatch --parsable slurm/build_llama.sbatch)                      # step 2: CPU + CUDA builds (~30 min)
M=$(sbatch --parsable -d afterok:$B slurm/make_models.sbatch)         # BF16, Q4_K_M, Q4_0-pure, Q8_0
sbatch -d afterok:$M slurm/smoke_s1.sbatch                            # step 4: one tool call on a P100
squeue -u $USER;  tail -f /scratch/cmpe295-bertin/logs/smoke-s1_*.out
```
The llama.cpp build and the model files are shared: once they exist, other members skip those two jobs.
S1 on other GPUs: `sbatch --gres=gpu:a100:1 --exclude=cs001,cs002 slurm/smoke_s1.sbatch` (or `gpu:h100:1`).

Experiments: `python -m testbed.run +experiment=<name>` writes `array.sbatch` + `report.sbatch` under
`$TESTBED_OUT` (`submit=true` also submits them). Each array task requeues itself 3 minutes before its
walltime and resumes (spike S10); finished episodes are never rerun.

## M2 runs (branch `m2`)
BF16 references with KL checks for the six models of Workbook 2 Table 13, spike S7 under a 10-minute limit and
spike S10. First check that a job can requeue itself (S7 and S10 depend on it):
```bash
sbatch slurm/requeue_probe.sbatch            # pass: its log shows "restart 1" and PASS
scontrol show config | grep -i requeue
```
Login node (network): llama.cpp perplexity tool, checkpoints, eval text, S7 data.
```bash
git fetch origin && git checkout m2 && source scripts/hpc_env.sh
uv pip install -q --python .venv/bin/python -e ".[train,dev]"
for r in LiquidAI/LFM2.5-1.2B-Instruct HuggingFaceTB/SmolLM3-3B LiquidAI/LFM2.5-2.6B; do
  DOWNLOAD_ONLY=1 bash scripts/make_probe_models.sh $r $TESTBED_MODELS; done
bash scripts/get_eval_text.sh                                   # WikiText-2 raw test + SHA-256
```
Jobs:
```bash
B=$(sbatch --parsable slurm/build_llama.sbatch)                # adds llama-perplexity, llama-imatrix
for r in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B LiquidAI/LFM2.5-1.2B-Instruct \
         HuggingFaceTB/SmolLM3-3B LiquidAI/LFM2.5-2.6B; do
  M=$(sbatch --parsable -d afterok:$B -t 02:00:00 --mem=48G slurm/make_models.sbatch $r)  # skips existing files
  sbatch -d afterok:$M slurm/kl_check.sbatch ${r#*/}           # $TESTBED_OUT/kl/<model>/kl_<model>.json
done
RUN=A S7_STEPS=50 sbatch -t 01:00:00 slurm/s7_qad.sbatch        # seconds per step, then choose S7_STEPS
python -m testbed.run +experiment=s10_requeue submit=true 'slurm.time="00:06:00"'   # time from episode speed
```
Checks: `scripts/kl_report.py` (inside kl_check), `train/compare_loss.py` (S7), `scripts/check_s10.py` (S10).
Environment notes (2026-11-03): `train/make_qad_data.py` needs `datasets`, whose pyarrow wants NumPy 2 while the
testbed pins NumPy 1.26, so build S7 data in a separate env (`$CMPE295_GROUP/envs/qad-data`, with `UV_LINK_MODE=copy`
because uv's default link mode fails on this NFS). The report job needs `fastparquet` in the testbed venv.
KL checks use WikiText-2 test text only. For M3, add logged agent prompts as a second corpus, plus IQ4_XS
variants with an imatrix whose calibration text shares no 13-gram with any benchmark task.

## M2 results (2026-11-03, development data; JSON in `results/m2/`)
- **Requeue probe** (job 101628): `JobRequeue=1`; USR1 arrived 60 s before the 2-minute limit, the job requeued
  itself and restarted with `SLURM_RESTART_COUNT=1`. PASS.
- **llama.cpp** d812350: `llama-perplexity` and `llama-imatrix` added to the shared build (job 101629); the other
  binaries were not rebuilt.
- **BF16 conversions** of the three new models at the pinned converter (jobs 101634, 101636, 101638), with Q4_K_M,
  pure Q4_0 and Q8_0 from the same BF16 file. Revisions, SHA-256, sizes and bits per weight:
  `results/m2/models_manifest.json`.
- **KL checks** against BF16 (jobs 101630–101632, 101635, 101637, 101639): `llama-perplexity --kl-divergence`,
  WikiText-2 raw test (SHA-256 173c87a5…), ctx 512, 50 chunks, CUDA on a P100. Mean KLD / same top token /
  whole-file bits per weight. Q8_0 is closer to BF16 than both 4-bit variants for every model.

| Model | BF16 PPL | Q4_0-pure | Q4_K_M | Q8_0 |
|---|---|---|---|---|
| LFM2.5-1.2B-Instruct | 17.45 | 0.1840 / 78.5% / 4.52 | 0.1020 / 83.3% / 5.00 | 0.0030 / 96.2% / 8.52 |
| LFM2.5-2.6B | 42.09 | 0.2475 / 77.4% / 4.53 | 0.1648 / 81.1% / 4.97 | 0.0056 / 95.6% / 8.53 |
| Qwen3-0.6B | 20.40 | 0.2455 / 75.3% / 4.57 | 0.1083 / 83.3% / 5.15 | 0.0033 / 95.8% / 8.57 |
| Qwen3-1.7B | 15.70 | 0.2184 / 81.2% / 4.53 | 0.1069 / 87.0% / 5.05 | 0.0054 / 95.9% / 8.52 |
| Qwen3-4B | 13.31 | 0.1250 / 85.7% / 4.51 | 0.0780 / 88.9% / 4.97 | 0.0118 / 94.9% / 8.51 |
| SmolLM3-3B | 9.16 | 0.2400 / 74.7% / 4.52 | 0.0448 / 88.3% / 4.98 | 0.0060 / 94.8% / 8.52 |

- **S7** (Workbook 2 Table 35, onboarding step 6). 50 steps took 94 s on an A100 (1.33 s/step), so a 10-minute
  limit would never cut them; the run was lengthened to 580 steps (data: 2336 x 512 tokens). A: uninterrupted,
  job 101642 on cs001. B: `-t 00:10:00`, job 101653 on cs004; USR1 at step ~368, requeued, resumed from the
  step-365 checkpoint and finished. **All 580 per-step losses identical to A (max abs diff 0.0), across two
  nodes.** Peak GPU memory 18.8 GB (A) / 19.9 GB (B) of 40 GB. `results/m2/s7_compare.json`. PASS.
- **S10** (Workbook 2 Table 35, onboarding step 5): `configs/experiment/s10_requeue.yaml`, 8 configurations x 13
  BFCL tasks = 104 episodes on P100s, `slurm.time=00:05:00` (USR1 at 2:00; chosen from the Oct 28 timings, 34–150 s
  of episodes per configuration). Array 101640: every task was cut and resumed; R1 with the forced grammar on Q4_0
  (no prior timing) progressed on each attempt but used all 5 requeues with 3 episodes left. Resubmitting
  (array 101658) ran only those 3. **104/104 episodes, no missing, no duplicate episodes or step rows.**
  `results/m2/s10_check.json` (first submission: `s10_check_first_submission.json`). PASS after one resubmission.
  For the main runs, size `slurm.time` from the slowest configuration, or raise `max_requeue`.
- **Bad node: cs001** (A100). On 2026-11-03 GPUs 1, 2 and 3 failed CUDA initialisation ("No CUDA GPUs are
  available") in three jobs; GPU 0 works. Until fixed, A100 jobs use `--exclude=cs001,cs002`.

## First results (2026-10-26, development data)
- Onboarding step 2: llama.cpp d812350 built in job 96279 on g2 (P100 node) in 17 min, CPU build plus CUDA 12.6 for sm 60, 80 and 90.
- The rebuilt GGUF files (job 96280) are byte-identical to the Oct 8 Colab pilot: SHA-256 of BF16, Q4_K_M and
  pure Q4_0 equal configs/experiment/skeleton.yaml. Q8_0 (new for R2): `5c9654ed…cfa28e`.
- Onboarding step 4 / S1, Qwen3-0.6B Q4_K_M, greedy, 3 repeats each, same `get_weather(city="San Jose",
  unit="celsius")` call every time:

| GPU | Node | Job | Decode tok/s | Prompt (205 tok) ms | GPU memory |
|---|---|---|---|---|---|
| P100 12 GB | g2 | 96387 | 144 | 89 | 1.6 GB |
| A100 40 GB | cs001 | 96390 | 231 | 53 | 1.8 GB |
| H100 80 GB | g19 | 96389 | 350 | 18 | 1.8 GB |

## Checked accounts
| Member | Date | Smoke test |
|---|---|---|
| Yashashav (017856371) | 2026-10-26 | srun on gpuqs: P100, A100, H100 |
| Ekant (015945761) | 2026-11-08 | srun on gpuqs: P100 (g2, job 109435); SSH key login, group cmpe295-bertin, Slurm account coe |
| Vritika | | |
