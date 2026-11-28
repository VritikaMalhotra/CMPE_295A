# C4 cost runs (H4 / RQ4): plan

Draft 2026-11-27, for the team and the advisor. Nothing has run. The registration requires the list of
"no overhead" claims (family RQ4) to be in this plan **before** the cost runs, so §4 must be agreed and committed,
and filed on OSF with any scope change, before the first C4 job.

## 1. What is registered

- **Stage C4** (prereg §3): every C1/C3 configuration on both backends, single-stream blocks on an exclusive node,
  at least 10 interleaved repetitions of a fixed 20-task cost set.
- **Costs** (§4): weight memory (file and resident), time to first token, decode tokens/s, extra tokens
  (re-prefill for R2c, grammar-forced tokens for R1), GPU energy per step (NVML, CUDA only), all on an
  exclusively allocated node. R2c also records which copy served each step, plus the probe's tokens and time.
- **H4**: R1 adds no weight memory and at most 5% to median per-step decode latency on CUDA. Any overhead falls on
  TTFT (grammar setup) and is larger on CPU. Q8_0 is about 1.9× Q4_0 weight memory. R2c is about 2.9× plus
  re-prefill tokens. R3 adds no inference cost.
- **Analysis** (§6): medians with bootstrap CIs; TOST with a ±5% margin for every "no overhead" claim (R1
  decode latency on CUDA; R3 latency vs Q4_0); Holm within family RQ4; a Pareto frontier of failure reduction
  (from H2) against each cost.

## 2. Grid and budget

51 configurations per backend: C1 has 6 models × 4 formats (24), and C3 has 3 core models × 9 repairs (27).
Each configuration × backend runs 20 tasks × 10 repetitions = 200 episodes, so the full design is
102 × 200 = 20,400 episodes.

Episode times measured on the holdout (mean wall seconds per episode; CPU used 4 cores, not an exclusive node):

| Model | CUDA, C1 (4 formats) | CUDA, C3 (9 repairs) | CPU, holdout C2 |
|---|---|---|---|
| Qwen3-0.6B | 2.8–3.5 | 2.8–4.5 | 410–610 |
| Qwen3-1.7B | 4.8–7.3 | 5.0–8.3 | 940–1,540 |
| LFM2.5-1.2B | 3.2–3.8 | 3.1–5.3 | not run |
| Qwen3-4B | 8.9–10.6 | – | not run |
| SmolLM3-3B | 12–23 (long tail) | – | not run |
| LFM2.5-2.6B | 30–46 | – | not run |

Estimates:

- **CUDA, full scope:** about 23 h of episodes, plus about 3 h for server loads (510 blocks). Run as one job
  per model, so every contrast stays on one node; 2 nodes at a time (QoS) makes about 13 h of wall time.
  **Feasible.**
- **CPU, full scope, at the measured 4-core speed:** C1 about 130 node-days and C3 about 46 node-days.
  An exclusive 32-core node will be faster, but by how much is unknown (prefill should scale with cores,
  decode much less). Even at 4× it is about 45 node-days. **Not feasible as registered** (§5).

### Two cluster accounts (decided 2026-11-27)

The C4 jobs are split between Ekant's and Vritika's accounts. Node limits are per user (`gpu2nodes`: 2 nodes;
`cpu200`: 200 cores, which is 3 exclusive 56-core `cpuqs` nodes), so two accounts double what can run at once.
Both accounts are in `cmpe295-bertin`, and the code, venv and `runs/` are group-readable and writable.

- **Each person submits their own jobs from their own login.** No one runs jobs under the other's account.
- **Same code for both.** Both deploy the same commit with `git archive` and check the driver's SHA-256 before
  submitting. Both write to `/scratch/cmpe295-bertin/runs/c4_*`, and each run logs the user, job id and node.
- **The split is by model, so each model's contrasts stay on one node.**
  - CUDA, 6 jobs on 4 nodes at once. Ekant: LFM2.5-2.6B, Qwen3-0.6B, LFM2.5-1.2B. Vritika: SmolLM3-3B,
    Qwen3-4B, Qwen3-1.7B. The longest single job (LFM2.5-2.6B, about 9 h) sets the wall time, so about 9 h
    instead of 13 h.
  - CPU (option C), 3 jobs, one per core model. Ekant: Qwen3-1.7B and Qwen3-0.6B. Vritika: LFM2.5-1.2B. Three
    nodes already cover this, so the second account mostly frees capacity for reruns. Splitting one model's
    repetitions across two nodes would be faster, but node would then enter the contrast, so it isn't planned.
- **The analysis pools both accounts' run directories.** It checks that every block ran the same code hash
  and cost-set hash.

## 3. Design choices to fix before the runs

1. **Cost set.** 20 tasks drawn from the BFCL holdout with seed 295, stratified 7 / 7 / 6 over
   `multi_turn_base`, `miss_func` and `miss_param`, without looking at any outcome. Stored as
   `results/m3/c4_cost_set.json` with its SHA-256; the runner refuses any other list (as for the holdout).
2. **Decoding.** Greedy (seed 0). Repetitions are then the same workload, so their spread is measurement noise,
   which is what the medians and TOST need. The other settings are as C1–C3 (context 40,960, at most 4,096 new
   tokens, prompt cache off, one slot).
3. **Block.** One configuration runs the 20 tasks once in a fixed order: start the server or servers, send one
   warm-up request (discarded), run the 20 episodes, record 30 s of idle power, then stop. R2c runs both
   resident copies.
4. **Interleaving.** One job per model and backend. In each repetition r = 1…10, the model's configurations run
   in a random order (seed 295 + r), logged with the repetition and block index.
5. **Exclusive nodes.** CUDA: `--exclusive` on an A100 node, using one GPU (the other three stay idle, so NVML
   energy is not shared). CPU: `--exclusive` on a `cpuqs` node, with `TESTBED_CPU_THREADS` set to the physical
   core count; the node name is logged and reused across repetitions if possible.
6. **Metric definitions.**

   | Metric | Definition |
   |---|---|
   | Decode latency | ms per generated token per step (`predicted_ms / completion_tokens`); "median per-step" = median over steps |
   | TTFT | `latency_ms − decode_ms`, request wall time minus decode, so it includes grammar setup. `prompt_ms` alone does not. |
   | Weight memory, file | GGUF bytes, from `results/m3/models_manifest.json`; R2c counts both files |
   | Weight memory, resident | Peak GPU memory of the llama-server processes (NVML per process). On CPU: peak RSS of the server (`VmHWM`) |
   | Energy | NVML counter delta per episode, divided by its steps; the block's idle power is reported, not subtracted |
   | Extra tokens, R2c | probe prompt tokens + Q8_0 re-prefill tokens per step (already logged) |
   | Extra tokens, R1 | see §6, item 3 |

7. **Analysis code first.** `analysis/rq4_tests.py` and its unit tests are written, run on pilot data and
   committed before any C4 run, as done for H1–H3.

## 4. Proposed family RQ4 ("no overhead" claims, TOST ±5%, Holm within the family)

Each claim compares a treatment's median with Q4_0's. TOST checks that the ratio stays within ±5%, using a
stratified bootstrap over tasks, with the 3 core models as strata.

| # | Claim | Metric | Backend |
|---|---|---|---|
| 1 | R1 adds ≤ 5% decode latency vs Q4_0 | ms per token | CUDA |
| 2 | QAD-KL adds no latency vs Q4_0 | ms per token | CUDA |
| 3 | QAD-KL+CE adds no latency vs Q4_0 | ms per token | CUDA |
| 4 | QAD-KL adds no latency vs Q4_0 | ms per token | CPU |
| 5 | QAD-KL+CE adds no latency vs Q4_0 | ms per token | CPU |

Claims 4–5 depend on the CPU scope (§5). Reported without tests:

- R1's TTFT overhead (CUDA and CPU), and whether it is larger on CPU.
- File and resident memory ratios against the predictions (Q8_0 1.9×, R2c 2.9×, R1 1.0×).
- R2c's extra tokens.
- R3 TTFT.
- The Pareto frontiers.

Open point: whether claim 1 is "within ±5%" (two-sided TOST, as registered) or "at most +5%" (one-sided, the
literal H4 wording). I'd recommend two-sided, since it's what the registration's analysis section says.

## 5. CPU scope: option C chosen (deviation, to be filed on OSF before the runs)

**Decision 2026-11-27 (Ekant): option C.** The CPU arm runs the 21 claim configurations below; CUDA keeps the
full registered scope (all 51). This narrows C4 and is a deviation: it goes into the Dec 9 OSF update with the
RQ4 claim list, before any C4 job. The Dec 8 pilot measures the real CPU speed. If option C still does not fit
in about 3 days on 3 nodes, the repetitions stay at 10 and the plan is revisited before filing, not after.

The registered CPU arm (all 51 configurations) cannot finish in any reasonable time. Options:

| Option | CPU configurations | Estimate (4 cores ÷ assumed 4× speedup) | What it keeps |
|---|---|---|---|
| A. As registered | 51 | about 45 node-days | everything |
| B. Core models only | 39 (3 models × 4 formats + 9 repairs) | about 16 node-days | all repairs on CPU |
| C. Claims-only (recommended) | 21 (3 core models × BF16, Q4_0, R1, Q8_0, R2c, QAD-KL, QAD-KL+CE) | about 9 node-days, about 3 days wall on 3 nodes | every H4 statement and both CPU claims |

Any reduction is a deviation and needs an OSF update before the runs. A 1-day pilot (below) replaces the
assumed 4× with a measurement before the scope is filed.

## 6. Code still to write

1. **C4 driver** (`testbed/cost_run.py`, `configs/experiment/c4_*.yaml`, a Slurm template with `--exclusive`):
   blocks, interleaving, warm-up, idle-power window, repetition and block logging, cost-set hash check.
   Reuses the existing server, backend, runner and `GpuMeter`.
2. **Profiler additions** (`testbed/profiler.py`): per-process GPU memory, CPU peak RSS, idle power.
3. **R1 grammar-forced tokens.** llama-server does not report them. Options:
   - (a) Report R1's change in completion tokens vs Q4_0 per step. Simple, but it's a proxy.
   - (b) Replay each R1 step offline with the grammar sampler and count the tokens the grammar fully
     determined. Exact, but more work.

   I recommend (a), disclosed as a proxy.
4. **Analysis** (`analysis/rq4_tests.py` and tests): medians, stratified bootstrap CIs, TOST, Holm, memory ratios,
   Pareto frontiers against the RQ2 holdout effects.

## 7. Timeline (after the Dec 3–4 deadlines)

| When | What |
|---|---|
| Dec 5–8 | Driver, profiler additions, analysis code, unit tests; draw the cost set |
| Dec 8 | Pilot (Ekant's account, plus one short CUDA job from Vritika's to check the shared paths): 1 repetition of Qwen3-0.6B Q4_0 and R1 on CUDA, and on an exclusive CPU node (measures the CPU speedup and checks every metric) |
| Dec 9 | Confirm option C against the pilot speed and fix the RQ4 claim list; commit this plan; file the OSF update (claims, plus any deviation) |
| Dec 10–11 | CUDA runs (about 9 h on 4 nodes, two accounts) |
| Dec 10–20 | CPU runs (option C: about 3 days on 3 nodes) |
| After the runs | `analysis/rq4_tests.py` on the C4 data, then the H4 results doc |

## 8. Decisions needed

1. ~~CPU scope~~: **C, decided 2026-11-27** (§5); filed on OSF Dec 9 with the claims.
2. The RQ4 claim list (§4), and whether claim 1 is two-sided.
3. R1 extra tokens: proxy (a) or exact replay (b).
4. Greedy decoding and the TTFT definition (§3); both are recommended as written.
