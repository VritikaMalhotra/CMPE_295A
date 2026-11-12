# CoE HPC (coe-hpc2): limits and runbook

Measured 2026-10-26 under account `coe` (onboarding step 1, Workbook Table 37b). Every member checks their own
login with the smoke test below and adds a line to "Checked accounts".

## Facts
| | |
|---|---|
| Login | `coe-hpc2.sjsu.edu`, SJSU VPN or campus network only; SSH key login (see each member's `~/.ssh/config`) |
| Slurm account | `coe` (`-A coe`). GPU partitions use QoS `gpu3nodes`: **at most 2 nodes per user** (a third GPU job waits with `QOSMaxNodePerUserLimit`), so one account runs at most 2 P100 jobs, or up to 8 single-GPU jobs on two A100 nodes (measured 2026-11-08). `defq` allows at most 2 CPUs per job; `cpuqs` up to 128 |
| GPU partitions | `gpuqs` (max 2 d 1 h), `gpuqm` (max 7 d 1 h); same GPU nodes |
| GPUs | 15 x P100 PCIe 12 GB (Qwen3-4B BF16 at ctx 40,960 does not fit: 8 GB weights + 5.8 GB KV cache; use A100/H100) (`gpu:p100:1`), 4 nodes x 4 A100 PCIe 40 GB (`gpu:a100:N`), 3 x H100 PCIe 80 GB (`gpu:h100:1`); drivers differ by node (575.57 on P100 node g2, 590.44 on an H100 node), so every job logs it |
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

### τ² runs: the simulated user as its own job (decision 4)
One A100 (or H100) job per experiment runs the simulator (gpt-oss-20b, 4 slots of 32,768 tokens) and writes
`$TESTBED_OUT/simuser/<experiment>.endpoint.json` (URL, model file, SHA-256, context, job id). Each agent
array task waits for that file (up to `simuser_wait_s`), checks that it names the experiment's simulator, and
sends its C4 cache misses there; it keeps its own store, `simuser_cache.<config>[.shard].sqlite`, so concurrent
jobs never overwrite each other's copy. Replays (`replay=<run_id>`) merge the stores, the episode's own first.
```bash
bash scripts/get_simuser_model.sh $TESTBED_MODELS          # login node, once: 12 GB + SHA-256 check
.venv/bin/python -c "import tau2"                        # setup_env.sh --tau2 installs it (login node)
bash scripts/tau2_submit.sh ref_qwen3_0p6b                 # simulator, agent array (after it starts), report, stop
```
Several experiments with the same simulator settings share one simulator:
`bash scripts/tau2_submit.sh ref_qwen3_0p6b ref_qwen3_1p7b ref_qwen3_4b`.
QoS: the simulator holds one of the 2 GPU nodes a user may use, so agent tasks default to A100s
(`AGENT_GRES`): 3 next to the simulator plus 4 on a second node. The simulator's walltime is 2 d 1 h
(`gpuqs` maximum); if it ends first, the agent jobs leave their tau2 episodes pending and a resubmission
finishes them.

**Smoke run, 2026-11-08** (`configs/experiment/tau2_smoke.yaml`, 1 BFCL + 2 airline tasks, Qwen3-0.6B BF16
greedy): simulator job 109543 on cs003, agent array 109544 on cs004, so **an agent on another node reaches the
simulator**. 11 simulator requests (9 user, 2 judge), all cache misses forwarded upstream; both tau2 episodes
finished (reward 0, as expected for 0.6B); the endpoint file was removed when the stop job cancelled the
simulator. **Replay** (job 109563, A100): both episodes 0 misses (7 and 4 hits), same reward and step count.
PASS.

**BF16 references launched 2026-11-08** (2505 specs each: BFCL 3 x 100 tasks + 201 tau2 write tasks, 4 sampled
seeds + 1 greedy): one simulator (job 109565) for all three; agent arrays 109566 (`ref_qwen3_0p6b/cd314594fb`),
109568 (`ref_qwen3_1p7b/c456da2aef`), 109570 (`ref_qwen3_4b/1c7b26d4e7`); reports 109567, 109569, 109571;
simulator stop 109572. Inclusion sets afterwards with `python -m testbed.inclusion` (see each config).

**What the first pass hit, and the rules now in the code (2026-11-09):**
- *Telecom needs `mcp`*: telecom's simulated user has tools, and litellm then imports `mcp`, which tau2 does
  not declare. Every telecom episode failed until it was installed; `setup_env.sh --tau2` now installs it.
  Telecom was rerun (0.6B array 109651, 1.7B 109657).
- *Simulated user out of context* (a 32,768-token slot): one episode ends, scored as a failure (reward 0),
  `error_type = tau2:user_context_overflow`; the overflow is stored in the C4 cache, so a replay repeats it.
  Before, it stopped every tau2 episode of that agent job. 1 episode so far (0.6B).
- *Simulated user empty after every escalation step*: **excluded, not scored** (the fault is the simulator's,
  not the agent's), and at once: the empty reply is cached, so a retry fails the same way. Recorded as a
  `permanent` line in `retries/<run_id>.jsonl`; each report lists them. 7 of 7,515 episodes: 0.6B 3
  (airline 8 s3, airline 42 s4, retail 29 s3), 1.7B 3 (airline 30 s1, airline 29 s3, retail 108 s3), 4B 1
  (retail 20 s4).
- *Config keys that look like numbers*: `only_config=2674909851e8` reached Hydra as the float 2.67e17, so the
  4B airline seed-0 array task ran nothing (26 episodes). The array template now quotes the key; 4B rerun
  with its own simulator (110093, agents 110094, report 110095).

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
- **S10 rerun on the integration branch** (2026-11-10, `week-2026-11-13` at `5bcf8d2`, after the array template
  started quoting `only_config`; review follow-up 1): same experiment hash `4375dbd0be`, fresh output root
  `runs/s10_rerun_20261110/`.
  - Array 111842, 5-minute walltime, one task at a time. Every one of the 8 tasks was cut and resumed.
  - The slowest configuration (R1 forced on Q4_0) used its 5 requeues with 10 episodes left (on 11-03 it was 3).
  - A 5-minute resubmission (112015) made no progress: what is left of a cycle after server start-up is too short
    for one of its episodes.
  - The last 10 ran with `slurm.time=00:20:00` (112024).
  - **104/104 episodes, no missing, unplanned or duplicate episodes or step rows.**
    `results/m2/s10_check_rerun.json`. PASS.
- **Bad node: cs001** (A100). On 2026-11-03 GPUs 1, 2 and 3 failed CUDA initialisation ("No CUDA GPUs are
  available") in three jobs; GPU 0 works. Until fixed, A100 jobs use `--exclude=cs001,cs002`.

## M2 results (2026-11-08, development data; JSON in `results/m2/`)
- **Clean-clone check, Ekant's account** (job 109482, P100 g2, commit 893fd70): fresh clone, `setup_env.sh`,
  10 skeleton episodes; experiment hash 3e3a25e70f and all 253 specs equal Yashashav's run, model SHA-256 matches
  the pilot, 0 unclassified steps. `results/m2/clean_clone_ekant.json`. **PASS** (the M1 criterion moved to M2).
- **FR-E17 CUDA vs CPU parity** (`scripts/parity_submit.sh`): 200 fixed prompts per model (Oct 28 BF16 seed-1
  episodes replayed), greedy, ctx 40,960, cache off, same GGUF file; CUDA on P100 (4B BF16 on H100: out of
  memory on the P100), CPU on `cpuqs` (32 CPUs).
  Prompt token IDs identical on both backends in 100% of prompts. Token-identical outputs / same call-or-reply
  decision / same decoded calls / median first divergent token:

  | Model | BF16 | Q4_0-pure |
  |---|---|---|
  | Qwen3-0.6B | 90.5% / 100% / 97.5% / 28 | 53.0% / 97.0% / 92.0% / 15 |
  | Qwen3-1.7B | 92.0% / 100% / 99.5% / 39.5 | 66.5% / 98.5% / 93.0% / 22 |
  | Qwen3-4B | 96.0% / 100% / 100% / 46 | 67.0% / 97.5% / 95.0% / 21 |

  **No model reaches the 99% pooling rule: backend stays a factor in every RQ1 analysis (decision 7).**
  The CUDA and CPU kernels round differently; 4-bit weights amplify it (outputs diverge 4–8× as often: 9.5% / 8% / 4% at BF16, 47% / 33.5% / 33% at Q4_0).
  `results/m2/parity_qwen3-{0.6b,1.7b,4b}.json`.
- **NFR-5 throughput pilot** (`scripts/throughput_submit.sh`, Qwen3-0.6B BF16 BFCL, 24 episodes per job):

  | Level | Ran at once | Episodes/h | Median episode | Queue wait |
  |---|---|---|---|---|
  | P100 x1 | 1 | 182 | 15.5 s | 54 min |
  | P100 x4 | 2 | 363 | 16.5 s | 65 min |
  | A100 x4 | 2 | 1,578 | 2.4 s | 167 min |
  | A100 x8 | 2 | 1,178 | 2.3 s | 175 min |

  **NFR-5 (≥ 8 concurrent jobs) is not met: QoS `gpu3nodes` allows 2 nodes per user, and the A100 nodes are
  shared, so free GPUs on our two nodes decide concurrency** (the A100 jobs lasted ~1 min, so they barely
  overlapped; in the BF16 reference runs 3 agent tasks plus the simulator ran at once on cs003/cs004).
  Episodes/h passes ≥ 120 easily: an A100 runs a 0.6B BFCL episode in ~2 s, 7 x faster than a P100. For the
  main grid, plan on 3–7 concurrent A100 jobs, or ask the administrator for a higher node limit.
  `results/m2/throughput.json`.
- **LFM2.5-2.6B Q4_K_M perplexity ratio** rerun with 200 chunks (job 109483): 0.862 ± 0.003 (50 chunks: 0.871).
  BF16 file SHA-256 equals the manifest. Mean KLD keeps the expected order (Q8_0 0.003 < Q4_K_M 0.166 <
  Q4_0-pure 0.242; same top token 97.3% / 81.0% / 77.1%), so the low ratio is not noise or a wrong file:
  Q4_K_M moves this model's predictions away from BF16 in a way that happens to lower WikiText perplexity.
  Cite KLD, not the perplexity ratio, as the damage measure. `results/m2/kl_LFM2.5-2.6B_c200.json`.

## M2 results (2026-11-09): BF16 references (development data; JSON in `results/m2/`)

Runs `ref_qwen3_0p6b/cd314594fb`, `ref_qwen3_1p7b/c456da2aef`, `ref_qwen3_4b/1c7b26d4e7` (2,505 specs each:
501 tasks = BFCL 3 x 100 + tau2 airline 26 / retail 104 / telecom 71, 4 sampled seeds at T 0.7 + 1 greedy).
Finished: 2,502 / 2,502 / 2,504; the rest are the excluded empty-simulated-user episodes (above).

- **Episode success, sampled trials** (greedy in brackets where it differs by more than 1 point):

  | Model | BFCL base | miss_func | miss_param | tau2 airline | retail | telecom |
  |---|---|---|---|---|---|---|
  | Qwen3-0.6B | 4.5% (6.0) | 2.0% | 1.2% | 0% | 1.9% | 0% |
  | Qwen3-1.7B | 6.2% | 2.2% | 8.5% | 0% | 3.6% (1.9) | 0% |
  | Qwen3-4B | 12.2% | 9.5% | 11.0% | 3.8% (0) | 8.2% (10.6) | 1.1% |

- **FR-E2 evaluator agreement: PASS, 7,506 / 7,506 verdicts (100%)** (`scripts/evaluator_agreement.py`, CPU job
  110190, `results/m2/evaluator_agreement.json`). Every finished episode of the three references, re-scored
  from its logged trajectory: BFCL (4,500) through bfcl-eval's own path for a local Qwen3 FC model (the
  handler's response parsing and `decode_execute`, `eval_runner._evaluate_single_multi_turn_entry`, prompts
  and ground truth from bfcl's own loaders); tau2 (3,006) through tau2's `evaluate_simulation` on the saved
  simulations. 2 tau2 episodes have no simulation to re-score (one agent and one simulated-user context
  overflow, scored without the evaluator). bfcl's eval_runner imports API clients that are not installed;
  the script replaces them with inert stand-ins (listed in the JSON), which multi-turn scoring never calls.
- **Inclusion sets** (`python -m testbed.inclusion`, `results/m2/inclusion_qwen3-{0.6b,1.7b,4b}.json`): tasks
  the reference solves in at least one of 4 sampled trials (a trial excluded by the infrastructure rule does
  not count; the task is judged on the rest). Multi-turn tasks included: **0.6B 17, 1.7B 33, 4B 73 of 501**,
  all under the registered 120, as decision 11 expected. So, per decision 11: the step-level tests keep all
  501 tasks; these sets apply only to the episode-success analysis.
- **Power, Table 27 recomputed** (`scripts/power_table.py`, `results/m2/power_table.json`): measured base rate
  per step, steps per episode (0.6B 9.1, 1.7B 11.2, 4B 11.8) and design effect D, for the registered test
  (4-bit share of events 0.5 vs 0.6, alpha 0.05/10, 80% power: **330 events at D = 1**, as Workbook 2 says).
  D by task (the clusters a paired comparison shares) is 3-21; D by episode 1-6. Pairing within a task removes
  much of the task-level variance, so the true D lies between. Episodes needed against the 2,000 sampled
  episodes a configuration gets (4 x 501):

  | Failure type (rate/step 0.6B / 1.7B / 4B) | Episodes needed, D task | D episode | Verdict at 2,000 |
  |---|---|---|---|
  | any failure (30% / 27% / 23%) | 212 / 214 / 250 | 106 / 92 / 99 | detectable, all models |
  | exec_error (7.2% / 10.4% / 6.6%) | 1,336 / 1,220 / 1,820 | 553 / 419 / 667 | detectable |
  | tool_error_payload (9.4% / 8.0% / 6.2%) | 1,347 / 1,673 / 1,584 | 458 / 488 / 437 | detectable |
  | unknown_tool (7.0% / 5.5% / 8.5%) | 3,156 / 2,933 / 2,719 | 1,107 / 1,114 / 746 | only if D is near the episode value |
  | missing_call (1.0% / 1.4% / 0.9%) | 5,109 / 3,126 / 5,600 | 2,149 / 1,088 / 1,819 | borderline |
  | missing_call_direct_answer (2.9% / 0.5% / 0.6%) | 2,439 / 6,999 / 5,428 | 850 / 2,712 / 2,025 | 0.6B only, at low D |
  | invalid_args (2.7% / 0.9% / 0.2%) | 4,040 / 12,406 / 24,845 | 2,569 / 5,043 / 8,922 | no |
  | malformed, forced_quit, truncated (< 0.3%) | > 13,000 | > 13,000 | no: pooled or exploratory |

  So the confirmatory step-level tests are powered for any failure, exec_error and tool_error_payload;
  unknown_tool and missing_call depend on D (estimate it from the Stage 1 pairs); the rest are pooled or
  exploratory, as registered.

## A1 smoke runs (2026-11-10, development data)

One BFCL episode (`multi_turn_base_55`) and one τ² episode (airline 7), seed 0, per new prompt handler
(`slurm/a1_smoke.sbatch`, `ef37df0`); the two LFM2.5 handlers ran again after `5ebfd2b`, which asks llama-server to
keep the call markers and think tags (`runs/a1_smoke_v2/`). Counts read from the run directories by
`scripts/a1_smoke_summary.py` (`results/m2/a1_smoke_summary.json`):

| Model | Code | BFCL base_55 | τ² airline 7 |
|---|---|---|---|
| SmolLM3-3B | `ef37df0` | ran end to end, not solved (9 steps, `empty_turn_model_response`) | ran end to end, reward 0 (6 steps) |
| LFM2.5-1.2B | `ef37df0` | ran, not solved (5 steps, `empty_turn_model_response`) | ran, reward 0 (3 steps) |
| LFM2.5-2.6B | `ef37df0` | ran, not solved (5 steps, `empty_turn_model_response`) | ran, reward 0 (6 steps) |
| LFM2.5-1.2B | `5ebfd2b` | ran, not solved (12 steps, `instance_state_mismatch`) | ran, reward 0 (14 steps) |
| LFM2.5-2.6B | `5ebfd2b` | ran, **solved** (15 steps) | ran, reward 0 (11 steps) |

Every handler ran one BFCL and one τ² episode end to end, scored by the official evaluators. As a task-success
test, only LFM2.5-2.6B's BFCL episode passed; no τ² smoke episode was solved (the BF16 references of SmolLM3,
LFM2.5-1.2B and Qwen3-0.6B later solved airline in 0 of 122, 128 and 128 episodes). No new smoke runs were made.

## A2 results (2026-11-12): BF16 references for SmolLM3-3B and LFM2.5 (development data; JSON in `results/m2/`)

Runs `ref_smollm3_3b/5053915afb`, `ref_lfm2p5_1p2b/3018b3dc15`, `ref_lfm2p5_2p6b/86ef1012fd` (2,505 specs each,
the Qwen3 references' design). A2's bar: at least 99% of specs finished; an excluded episode is not finished
(ruling, 2026-11-11). How each run got there, with every log finding, is in the next section.

| Model | Jobs (array; reruns; report; A2 check) | Finished | Excluded | 99% bar | FR-E2 | Multi-turn tasks included |
|---|---|---|---|---|---|---|
| SmolLM3-3B | 111883; 112924_10, telecom loop 113054; 111884; 113184 | 2,458 (98.1%) | 47: 46 empty reply, 1 simulator HTTP 500 × 4 | not met | 2,455 / 2,456, **FAIL** (decision 17) | 10, or 11 with bfcl's verdict (not committed) |
| LFM2.5-1.2B | 111885; 112937; 112992; 113030 | 2,495 (99.6%) | 10, empty reply | met | 2,494 / 2,494, PASS | 34 (single-turn-only) |
| LFM2.5-2.6B | 111887; 113083, retry 113224; 111888, 113225; 113236 | 2,504 (99.96%) | 1, empty reply | met | 2,504 / 2,504, PASS | 360 (full multi-turn) |

- JSON: `results/m2/evaluator_agreement_{smollm3-3b,lfm2.5-1.2b,lfm2.5-2.6b}.json`,
  `results/m2/inclusion_lfm2.5-{1.2b,2.6b}.json`. SmolLM3's inclusion set stays on scratch
  (`runs/week_outputs/a2_smollm3_113184/inclusion_smollm3-3b.json`) because it depends on the disputed verdict.
  Counts: report jobs and `results/m3/run_status.json` (09:54).
- **Episode success, sampled trials, of finished episodes** (greedy in brackets where it differs by more than 1
  point; episodes parquet built in each A2 check's shadow directory, computed 2026-11-12 by 09:58):

  | Model | BFCL base | miss_func | miss_param | τ² airline | retail | telecom |
  |---|---|---|---|---|---|---|
  | SmolLM3-3B | 2.2% | 0.5% | 0.2% | 0.0% | 1.8% | 0.0% |
  | LFM2.5-1.2B | 8.8% (13.0) | 0.5% | 8.8% (12.0) | 0.0% | 2.2% | 0.0% |
  | LFM2.5-2.6B | 68.2% (67.0) | 49.0% (46.0) | 39.2% (35.0) | 35.9% (30.8) | 38.9% (33.7) | 63.7% (56.3) |

  LFM2.5-2.6B (whose template always opens a reasoning block) is far ahead of every other reference, Qwen3-4B
  included (12.2% BFCL base, 1.1% telecom). FR-E2 agrees on all its episodes, so our scoring matches the
  official evaluators, but the size of the gap (telecom 64% at 2.6B) has not been checked against any outside
  result: **unverified, notable**. Its successful telecom episodes take 10–39 steps (median 18).
- **Power table for all 6 models: pending.** Due only if SmolLM3 and LFM2.5-2.6B both complete A2; the SmolLM3
  item stopped at FR-E2 (decision 17).

## A2 in progress (2026-11-11): BF16 references for SmolLM3-3B and LFM2.5 (development data)

Not finished. Counts are in `results/m3/run_status.json` (refreshed for the deck); this section records the job IDs
and what the logs showed.
- **Arrays.**
  - SmolLM3-3B: 111883, plus 112924_10, which resubmits task 10.
  - LFM2.5-1.2B-Instruct: 111885.
  - LFM2.5-2.6B: 111887. Task 0 started 2026-11-11 20:43; task 1 finished 100/100 at 21:42.
  - All on A100s with cs001 excluded, sharing one simulated user (`gpusafe/simuser_safe.sbatch`).
  - Code `6be0ff7`, checkout `CMPE_295A-a2`.
- **FR-E2 preview**, on the episodes finished by about 8:20 PM: **3,432 / 3,432 verdicts agree**
  (`results/m2/evaluator_agreement_preview_20261111.json`, job 112923).
  - SmolLM3: BFCL 604/604, τ² 749/749.
  - LFM2.5-1.2B: BFCL 1,489/1,489, τ² 590/590.
  - The parquet files were built in a shadow directory of symlinks, never in the live runs.
  - The models have no bfcl-eval handler, so the BFCL part runs our own parser for each model before bfcl-eval's
    scoring. It checks the scoring path, not the parser.
  - One SmolLM3 τ² episode had no simulation to re-score (simulated-user context overflow).
- **What the task logs showed** (`runs/ref_*/*/slurm_logs/`). A task that exits COMPLETED is not always finished:
  - **SmolLM3 task 10 (telecom): 76 of 284 episodes, then stopped.** At 2026-11-11 16:29 the simulated user
    returned HTTP 500 ("The model produced output that does not match the expected peg-native format"). The runner
    treats a simulator failure as "unavailable": the episode stays pending, τ² scheduling stops, and the task
    exits 0. Resubmitted as 112924_10. B1's Q4_0 telecom shard stopped the same way and completed on retry.
    - **112924_10 failed the same way** (2026-11-11 22:48–22:49, cs002): the same episode
      (`[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_on`, seed 2) got the same
      HTTP 500, and the task stopped with 0 new episodes. A 500 is not cached, so each attempt asks the simulator
      again: the ledger of this configuration's cache holds 14 `upstream_500` rows, all for one request key
      (14 of 14 attempts failed). B1's telecom shard recovered from a 500 on retry, but that was a different
      request. Not resubmitted again (the spec's stop rule: the same job failing twice); while this request
      fails, every resubmission stops here and leaves the other 207 episodes pending.
    - **Runner fix (wk11-ekant `9d8b13f`, week `8417963`, 2026-11-11 23:20).** The runner never applied the
      registered infrastructure rule (pre-registration §5: retried up to three times with the same seed, then
      excluded and listed) to simulator errors: any simulator failure stopped τ² scheduling and recorded nothing.
      Now an HTTP 500 that the simulator answers (it is up; one request failed) is an attempt of that episode,
      recorded in `retries/`, and the other episodes go on; after four attempts the existing rule excludes it.
      Unreachable, 502/503/504 and the budget ceiling are still an outage, and three such episodes in a row are
      treated as one. Tests: local 254 passed, 13 skipped; cluster suite 113052 on `8417963`: 266 passed, 1
      skipped.
    - **Run: job 113054** (A100, config `486579e948da` only; replaces 113053, cancelled before it started, whose
      fixed five passes could end with episodes still pending once the three-in-a-row guard cut the later passes
      short). It repeats the configuration until no pending episode has fewer than four recorded attempts (at
      most 15 passes; finished episodes are never rerun), and stops at once on a real outage or if pass 1 trips
      the guard. It runs from checkout `CMPE_295A-wk` at `8417963`, so this configuration's remaining episodes
      run on `8417963`; the other SmolLM3 configurations ran or run on `CMPE_295A-a2`. Between `6be0ff7` (the
      first 76 telecom episodes) and `8417963`, the run path differs in `testbed/backends.py` (`parse_special`,
      the server default), `testbed/lfm2_format.py` (not used by SmolLM3), and `testbed/run.py`,
      `testbed/simuser_cache.py`, `testbed/tau2_bench.py` (this fix). Per-host manifests copied first to
      `runs/week_outputs/manifest_snapshot_smollm3_20261111_2328/`. The SmolLM3 report job 111884 now waits for
      111883 and 113054. `CMPE_295A-wk` stays frozen until 113054 ends (each pass is a new process).
    - **First run (2026-11-12 03:30, cs004, commit `8d94524`):** pass 1 had 208 episodes to try. The failing episode
      (`f86be84f4255`) got the same HTTP 500 and was recorded as attempt 1 of 4, a non-permanent
      `SimulatorRequestFailed` line in `retries/`, and the next episodes ran (5 finished within 3 minutes). It used
      the shared simulator (no new `server_logs/simuser.log`). The new runner branch works as designed.
    - **Judge-500 rule, first run (113083_7, 2026-11-12 03:33–03:36, `8d94524`):** LFM2.5-2.6B airline task 44
      seed 0 (the episode 111887_7 left pending after 7 judge 500s) finished and was scored: 18 steps, reward 0.
      This time the judge answered (`nl_judge` has a reward), so the new judge-error branch was not exercised. No
      `retries/` line; shared simulator. LFM2.5-2.6B task 7 is complete (26/26); report 111888 waits for it.
    - **Not covered:** every τ² task that starts from `CMPE_295A-a2` (`8c605a1`, no fix) can stall the same way
      on a simulator 500: LFM2.5-2.6B tasks 6–11 (6 running, 7–11 pending) and the τ² tasks (6–11) of the six
      pending H1 arrays (111933, 111935, 111937, 111939, 111941, 111943). SmolLM3 tasks 0, 3 and 4 are BFCL and
      never call the simulator. Moving that checkout is a separate decision (Yashashav's).
  - **SmolLM3 τ² exclusions** (empty simulated-user reply, the registered rule): 46 by 22:00 (airline 7 + 1,
    retail 7 + 31; `excluded` lines in the task logs).
  - **Ruling on A2's ≥ 99% bar (Yashashav, 2026-11-11 22:56): an excluded episode does not count as finished.**
    This matches the 11-09 Qwen3 section below ("Finished: 2,502 / 2,502 / 2,504; the rest are the excluded …")
    and `scripts/run_status.py`, which reports `finished` and `excluded` separately. Counts at 22:53
    (`run_status.py`):
    - SmolLM3-3B: 1,355 / 2,505 finished, 46 excluded. At most 2,459 / 2,505 = 98.2%: **A2 cannot pass for
      SmolLM3** under the registered rules.
    - LFM2.5-1.2B: 2,495 / 2,505 = 99.6% finished, 10 excluded: **passes**.
    - LFM2.5-2.6B: 785 / 2,505 finished, 0 excluded, running.
    - Running the excluded episodes again gives the same result: the empty reply is stored in the simulated-user
      cache and replayed (`testbed/simuser_cache.py`), the simulator runs at temperature 0, and a `permanent`
      retry line makes `retry_count` return the limit (`testbed/logger.py:69`), so the runner skips the episode.
      Finishing them would need a protocol change (a different simulator setting for those episodes, with the
      cache rows and exclusion lines set aside), which overrides pre-registration §5. Not done: it would be a
      post-hoc change for one model, and gpt-oss already failed at low, medium and high reasoning effort.
    - **Decision (2026-11-11, after a review of the evidence): keep the 46 exclusions under the registered rule;
      SmolLM3 does not meet A2's 99%.** One reading of pre-registration §5 would change this and is for the
      advisor: 34 of the 46 empty replies end with gpt-oss's 32,768-token slot full (`finish_reason = length`,
      `total_tokens` = 32,768 in every one), and §5 scores "simulated user runs out of context (its 32,768-token
      slot)" as a failure, not an exclusion. Read that way, SmolLM3 would have 12 exclusions and could reach
      2,493 / 2,505 = 99.5%. Because that reading turns a fail into a pass, it is not applied here. (Final,
      2026-11-12: the finished reference also has 1 telecom exclusion after 4 simulator HTTP 500s, so under that
      reading it would have 13 exclusions and 2,492 / 2,505 = 99.5%.) Slot-full /
      stopped-empty split of the exclusions in the other references: LFM2.5-1.2B 7 / 3, Qwen3-0.6B 1 / 2,
      Qwen3-1.7B 1 / 2, Qwen3-4B 1 / 0 (`empty_split.out`).
    - **Exclusions under quantization (Qwen3-0.6B, the only finished 4-bit cells with τ²; `run_status.py`):**
      BF16 3, Q4_0 4, Q4_K_M 1, IQ4_XS 7, all slot-full. Small and mixed, but if 4-bit models trigger more
      exclusions, the rule removes episodes the agent caused and can hide damage. For the advisor.
    - **Qwen3-1.7B Q4_0 (array 111933, still running; 2026-11-12 08:34, development data): 59 empty-reply
      exclusions so far, against 3 for the BF16 1.7B reference.** By configuration (`SimulatorEmptyReply` lines in
      `retries/`): retail greedy `1ab77a0bf7f3` 24 of 104 (task 9 ended 08:32, 80 finished, nothing pending),
      retail sampled `844cbb5cf2b0` 25 (task 8 running), airline sampled `35cd8bf82d5a` 9, airline greedy
      `95ea1c0a0fba` 1. Split (`empty_split.py`, output `empty_split_1p7b_20261112.out`, 08:34 in the same folder):
      37 slot-full (`finish_reason = length`, 32,768 tokens), 22 stopped empty. Q4_K_M (111935; its τ² tasks
      started 07:49, running): 1 so far. Why the 4-bit agent triggers so many more is not checked.
    - **Read-only checks of that comparison (2026-11-12 08:34–08:38; scripts and output in the same folder):**
      - Simulator settings are the same in both manifests (gpt-oss-20b MXFP4, same sha256, ctx 32,768,
        temperature 0, reasoning effort low; τ² `864350a`; llama.cpp `d812350`). The testbed commit differs: the
        BF16 reference ran on `c6e83a2` (dirty), the Q4_0 τ² tasks on `8d94524`, each with its own simulator job.
      - The agent writes tool calls as text far more often at Q4_0: 59 of the 60 empty replies (one more arrived
        during the scan) follow such a message (`textcall_scan.py`, `textcall_1p7b_q4_0_20261112.out`, 08:37); over
        all τ² reply steps, 412 of 4,814 (8.6%) look like a written tool call, against 33 of 6,945 (0.5%) for BF16.
      - BF16 1.7B's 3 empty-reply exclusions: retail sampled 1, airline sampled 2; retail greedy 0 of 104 (Q4_0: 24).
      - **The simulated user's opening message is not reproducible across runs.** BFCL first prompts are
        identical in all 1,500 matched episodes (BF16 vs Q4_0, same task and seed), so the code renders the same
        prompt; but 479 of 624 matched τ² episodes have a different first agent prompt, because the simulator's
        first message differs (e.g. retail task 17 seed 1: "Hi, … Could you help me with that?" vs "Hello, …
        Could you confirm the change?"), even at temperature 0 with the same model file. **Checked on the request
        itself (by 09:18, read-only on both runs' simulator caches):** 421 first-turn simulator requests are
        byte-identical in the two runs (same messages, `seed`, `temperature` 0, `reasoning_effort` low), and only
        113 of them got the same reply; 308 got a different one. So the input is the same and the simulator's
        output is not reproducible across runs. Cause not checked (candidates: the simulator server runs 4 parallel
        slots, so batching differs; different GPU and host). Within one run the cache makes the reply to a repeated
        request identical. The analysis pairs by task, averaging trials, so this adds noise to τ² pairs rather
        than breaking them; for the advisor.
  - **Why SmolLM3 has 46 exclusions when the three Qwen3 references had 7 in all** (read-only scan of the
    simulated-user caches and steps, 2026-11-11 22:40; scripts and their output in
    `runs/week_outputs/equiv_scan_20261111_2209/`):
    - 33 of the 46 empty replies follow an agent message with a tool call written as text (`"name": …` JSON,
      a ```json block, "Tool Call: …"); among non-empty replies the share is 1,026 / 8,230 (12%).
    - 34 of the 46 end with `finish_reason = length`: gpt-oss reasons until its 32,768-token slot is full
      (e.g. "We should not mention we are the user …" repeated).
    - SmolLM3 writes a `<tool_call>` block in 21 of 5,872 τ² steps (0.4%), against 781 of 3,404 BFCL steps
      (23%); 5,736 τ² steps (98%) are plain replies. LFM2.5-1.2B and the Qwen3 models reply in 49–61% of τ² steps.
    - **Tool shape differs by benchmark** (`render_smol.py` in the same folder, the testbed's own loaders and
      SmolLM3 handler, 2026-11-11 23:10). τ² passes each tool in OpenAI's wrapped form, so SmolLM3's prompt shows
      `{'type': 'function', 'function': {'name': 'calculate', …}}`; BFCL passes bare docs, shown as
      `{'name': 'authenticate_twitter', …}`. Both prompts carry the `<tool_call>` instructions, and the τ² one
      carries the policy. The same holds for every model (τ² tools are wrapped for Qwen3 and LFM2.5 too).
    - The τ² prompt is a faithful rendering: SmolLM3's chat template prints each tool exactly as given
      (`tool | string`, no unwrapping), and the
      golden cases, rendered from the template (sha256 `3e0e275a…`), use the wrapped shape. **Untested:** only
      the model's sensitivity to the tool shape (wrapped in τ², bare in BFCL); a small τ² run with bare tools
      would show it.
    - SmolLM3's τ² successes so far: retail 9 of 482 episodes (2 tasks), airline 0 of 122, telecom 0 of 147;
      Qwen3-0.6B: retail 10 of 519 (2 tasks), 0 elsewhere; LFM2.5-1.2B: retail 12 of 512 (4 tasks).
    - Either way, these exclusions follow what the agent wrote, not only the simulator. Pre-registration §5 says
      "the fault is the simulator's, not the agent's". For the team and the advisor.
  - **SmolLM3 still to run** (pending for an A100): BFCL tasks 0 (0/400), 3 (4/100) and 4 (0/400), which were
    requeued off P100s on 11-10 and 11-11. Telecom task 10 is stopped at 76/284 (above).
  - **LFM2.5-1.2B: finished** (as far as it can). All 12 tasks ended at 21:48; the 11 failed episodes ran again
    on the fixed code as 112937 (22:49–22:51, all finished), and report job 112992 (22:52) rebuilt the report:
    2,495 / 2,505 finished, 10 excluded. The SmolLM3 report job 111884 now also waits for 112924.
    - **A2 outputs for LFM2.5-1.2B** (job 113030, cpuqs, checkout `CMPE_295A-wk` at week `b8bed21`, which has
      the surrogate fix; 2026-11-11 23:03–23:06; parquet built in a shadow directory):
      - FR-E2 on every finished episode: **2,494 / 2,494 agree (100%), PASS**; BFCL 1,500/1,500, τ² 994/994;
        one τ² episode has no simulation to re-score (simulated-user context overflow)
        (`results/m2/evaluator_agreement_lfm2.5-1.2b.json`; its `stand_ins` list equals the one in
        `evaluator_agreement.json` and is replaced by that pointer).
      - Inclusion set: 34 multi-turn tasks included (BFCL base 15, miss_func 1, miss_param 14; τ² retail 4,
        airline 0, telecom 0), below the 120 needed, so single-turn-only confirmatory analysis, as for the three
        Qwen3 models (`results/m2/inclusion_lfm2.5-1.2b.json`).
      - Still to do for A2: the power table for all 6 models, which needs the SmolLM3 and LFM2.5-2.6B references.
      - Its 100% did not exercise the rule behind the SmolLM3 disagreement below: LFM2.5-1.2B has 0 BFCL
        episodes scored `multi_turn:inference_error` (SmolLM3: 33).
  - **SmolLM3-3B: finished** (as far as it can; 2026-11-12). Telecom job 113054 (03:30–06:43, cs004, `8d94524`,
    shared simulator) ended after 4 passes with nothing left to try: 207 of the 208 pending episodes finished;
    `f86be84f4255` (`[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_on`, seed 2)
    got the simulator's HTTP 500 ("The model produced output that does not match the expected peg-native format")
    in passes 1–4, 4 recorded attempts in `retries/`, so it is excluded (pre-registration §5). 0 FAILED, 0
    "simulated user unavailable". Report 111884 (06:43–06:44): **2,458 / 2,505 finished (98.1%), 47 excluded**;
    BFCL 1,500/1,500; airline 97/104 + 25/26, retail 385/416 + 97/104, telecom 283/284 + 71/71.
    - **A2 outputs for SmolLM3** (job 113184, cpuqs, checkout `CMPE_295A-ekant` at `8d94524`, 2026-11-12
      06:47–06:51; parquet built in a shadow directory):
      - **FR-E2: 2,455 / 2,456 agree (99.96%), FAIL.** τ² 956/956 (two episodes have no simulation to re-score,
        simulated-user context overflow); BFCL 1,499/1,500 (`results/m2/evaluator_agreement_smollm3-3b.json`,
        `stand_ins` replaced by the pointer as above). **This stops the SmolLM3 A2 item** (Stop and report). No
        code was changed and nothing was rerun.
      - **The disagreement:** `multi_turn_base_139` seed 3 (`17fa839b04e1`, config `4313e23b16b8`). In turn 1
        SmolLM3 wrote two `<tool_call>` blocks: `{"name": "get_watchlist", "arguments": {}}` and
        `{"watchlist": ["NVDA", "ZETA"]}` (the earlier tool result, copied as a call). The runner scores a block
        that is not a `{name, arguments}` object as `multi_turn:inference_error` at once (`testbed/runner.py:123`,
        the logged deviation: bfcl keeps it in history and crashes on the next prompt). The replay's
        `FormatHandler` (`scripts/evaluator_agreement.py`, used for models without a bfcl-eval handler) drops such
        blocks and keeps the rest, so bfcl's scoring sees `get_watchlist()` and passes the episode. 33 SmolLM3
        BFCL episodes are scored `inference_error`; 32 of them also fail bfcl's scoring.
      - **Probe** (2026-11-12, read-only, login node): the same raw text through bfcl-eval's unmodified
        `QwenFCHandler`: `_parse_query_response_prompting` keeps both blocks in the chat-history message;
        `decode_execute` and the next prompt build (`_add_assistant_message_prompting` + `_format_prompt`) both
        raise `KeyError: 'name'`. That supports the runner's verdict, but SmolLM3 has no bfcl-eval handler, so
        which side is "official" is **unverified**. Fixing either side is a research-design call (for the
        advisor), not made here.
      - **Inclusion set** (not committed; scratch `runs/week_outputs/a2_smollm3_113184/inclusion_smollm3-3b.json`):
        10 multi-turn tasks included (BFCL base 6, miss_func 1, miss_param 1; τ² retail 2, airline 0, telecom 0),
        single-turn-only confirmatory analysis; 39 sampled trials excluded by the infrastructure rule. It depends
        on the disputed verdict: `multi_turn_base_139` is excluded (all four sampled trials fail by our scoring);
        with bfcl's verdict for seed 3 it would be included (11 tasks; single-turn-only either way).
      - The power table for all 6 models is pending: the SmolLM3 A2 item is stopped.
  - **LFM2.5-2.6B: finished** (2026-11-12). Telecom task 111887_10 (01:37–08:41, cs002, `8d94524`) ended 281/284:
    3 episodes got the simulator's HTTP 500 ("does not match the expected peg-native format"), attempt 1 of 4
    each. Retry loop 113224 (09:41–09:45, cs004, `8d94524`, shared simulator) finished all 3 in pass 1 (the 500
    did not recur); 0 FAILED, 0 "unavailable". Report 113225 (09:45): **2,504 / 2,505 finished (99.96%), 1
    excluded** (airline task 12, empty simulated-user reply); A2's 99% bar is met.
    - **A2 outputs for LFM2.5-2.6B** (job 113236, cpuqs, checkout `CMPE_295A-ekant` at `8d94524`, 2026-11-12
      09:49–09:53; parquet built in a shadow directory):
      - FR-E2 on every finished episode: **2,504 / 2,504 agree (100%), PASS**; BFCL 1,500/1,500, τ² 1,004/1,004
        (`results/m2/evaluator_agreement_lfm2.5-2.6b.json`, `stand_ins` replaced by the pointer). LFM2.5-2.6B has
        0 BFCL episodes scored `multi_turn:inference_error`, so the decision-17 difference cannot arise here.
      - Inclusion set: **360 multi-turn tasks included** (BFCL base 85, miss_func 68, miss_param 59; τ² airline
        16, retail 67, telecom 65), above the 120 needed, so LFM2.5-2.6B is the first model with full multi-turn
        confirmatory analysis; 1 sampled trial excluded by the infrastructure rule
        (`results/m2/inclusion_lfm2.5-2.6b.json`).
      - The power table for all 6 models stays pending: the SmolLM3 A2 item is stopped (FR-E2, decision 17).
  - **Report header.** Every report and manifest says "M1 walking skeleton … excluded from the study", including
    these references and the Qwen3 ones. The text is hard-coded (`testbed/report.py:82`,
    `testbed/registry.py:160`), written whatever the experiment; no analysis code reads it.
  - **LFM2.5-1.2B: 11 episodes failed** (by 20:45), all with lone surrogates in the prompt:
    - 10 with `UnicodeEncodeError: surrogates not allowed` while hashing the prompt;
    - 1 with HTTP 400 from llama-server's `/tokenize`, which rejects an unpaired surrogate.
    - Affected: BFCL base_188 s0/s2/s3, miss_func_58 s2, miss_func_86 s1, miss_func_170 s1, miss_param_5 s4,
      miss_param_68 s1/s2, miss_param_183 s2/s3.
    - **Cause confirmed by the retries** (112937, on `8c605a1`): base_188's raw output holds
      `post_tweet(content="Excited for my upcoming adventure! \ud83c\udf89 \u2026")`, an escaped pair, which the
      fixed parser reads as one emoji; seeds 0, 2 and 3 now finish. miss_param_5 seed 4 (the HTTP 400) holds an
      unpaired `\ud83d` followed by a real emoji; that step is now logged as `malformed` and the episode goes on.
    - Before the retries: the cause, fixed in `ffe8873`, was that the Pythonic parser read `\ud83d\ude80`-style
      escapes as lone surrogates. The failing episodes' steps are not kept. What supported it: each
      of the 10 encode errors is at two adjacent positions (e.g. 34238\u201334239), which is what one escaped pair
      gives. The 400 is probably an unpaired surrogate, which the fix turns into a call that does not parse
      (logged as malformed, like any other unparseable call). The retries, run on the fixed code, will show.
- **Code change during the run (2026-11-11 20:46).** `CMPE_295A-a2`, the checkout every agent array runs from,
  moved from `6be0ff7` to week `8c605a1`.
  - On the run path that adds only three things:
    - `backends.py` asks `/tokenize` for `parse_special` explicitly, as llama-server already does by default
      (llama.cpp d812350, `tools/server/server-context.cpp:5446`: `json_value(body, "parse_special", true)`).
      SmolLM3 sends token ids too, so this is the only change that reaches SmolLM3, and it changes nothing;
    - the NUL fix (47d41f7);
    - the surrogate fix (ffe8873).
  - **Old against new parser on every logged step** (`scripts/parser_equivalence.py`, 22:10,
    `results/m2/parser_equivalence_lfm2p5_20261111.json`). Every tool-call block in `raw_text` was parsed with
    `lfm2_format.py` from both commits (each file checked byte-equal to `git show <commit>:testbed/lfm2_format.py`):
    - LFM2.5-1.2B, all 29,527 steps of the 2,484 logged episodes: 14,624 blocks, **0 parse differently**;
    - LFM2.5-2.6B, 5,695 steps so far (tasks 0–6): 3,766 blocks, 0 differ;
    - no step has a NUL character, and none has a surrogate escape (`\ud800`–`\udfff` written as text).
    - A test with one escaped pair and one NUL in a call block finds both differences, so the zero is not a
      blind check.
    - So no logged step would have parsed differently on `8c605a1`.
    - **Rescan after 111887_0 ended** (2026-11-12 00:21, 400/400, 0 FAILED; it was the last task on `6be0ff7`):
      LFM2.5-2.6B tasks 0 and 1, every step that ran the old code: 5,169 steps, 3,310 blocks, **0 differ**, no NUL,
      no surrogate escape; all 2.6B steps logged by 00:27: 15,832 steps, 9,918 blocks, 0 differ
      (`results/m2/parser_equivalence_lfm2p5_2p6b_20261112.json`). The code-move evidence is complete. (Replaces an earlier scan of 24,294 steps at about 20:46, which
      missed the steps logged after it.)
  - The move is at 20:46:43 (`git reflog` in the checkout). Each array task is one Python process, and
    `testbed/run.py` imports `backends` and `prompt_formats` (which imports `lfm2_format`) when it starts, so a task
    runs the commit it found at start:
    - `6be0ff7`: every LFM2.5-1.2B task (111885_0–11), LFM2.5-2.6B tasks 0 and 1 (task 1's manifest was written at
      20:46:34 with `6be0ff7`), and every SmolLM3 task that has run so far;
    - `8c605a1`: LFM2.5-2.6B tasks 2–6, which started between 20:53 and 21:48 (`sacct`; the manifests of tasks
      5 and 6 record `8c605a1`, the earlier ones were overwritten), tasks 7–11 when they start, and SmolLM3
      tasks 0, 3, 4, 112924_10 and the retries 112937 when they start.
  - The per-host manifests (which hold only the latest commit) were copied first, to
    `runs/week_outputs/manifest_snapshot_6be0ff7_20261111_2046/`.
  - The 11 episodes were resubmitted as 112937_[0,1,2,4].

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
