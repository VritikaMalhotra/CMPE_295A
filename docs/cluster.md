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

| Known bad node | **cs002** (A100): CUDA context creation fails (`CUDA_ERROR_UNKNOWN` from PyTorch, "device busy or unavailable" from llama.cpp) on 2026-10-26 while nvidia-smi looks normal. Add `--exclude=cs002` to A100 jobs until the administrator fixes it |
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
git clone -b m1/cluster https://github.com/VritikaMalhotra/CMPE_295A.git && cd CMPE_295A
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
S1 on other GPUs: `sbatch --gres=gpu:a100:1 --exclude=cs002 slurm/smoke_s1.sbatch` (or `gpu:h100:1`).

Experiments: `python -m testbed.run +experiment=<name>` writes `array.sbatch` + `report.sbatch` under
`$TESTBED_OUT` (`submit=true` also submits them). Each array task requeues itself 3 minutes before its
walltime and resumes (spike S10); finished episodes are never rerun.

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
| Ekant | | |
| Vritika | | |
