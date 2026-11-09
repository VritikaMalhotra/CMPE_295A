# M2 decisions: open items before the analysis plan is frozen

Status 2026-11-08. These were asked at the 2026-10-30 advisor review (Workbook 2, "Decisions requested from the
advisor"); the repository has no notes from that review yet. Each item lists the proposal and what it
changes. **Record the decision and the date next to each item** (or link the meeting notes), then freeze the
pre-registration (`docs/preregistration_m2.md`). Nothing here is decided until the advisor/team says so.

| # | Question | Proposal | What depends on it | Decision |
|---|---|---|---|---|
| 1 | Context budget: bfcl parity (ctx 40,960, cap 4,096 new tokens per step) or the Workbook's FR-E3 (16,384 / 1,024)? | Keep 40,960 / 4,096: matches bfcl-eval's local handler, so our verdicts compare with the official evaluator (FR-E2); every run so far used it. Update FR-E3 instead. | All reference and main-grid configs (`server.ctx`, `decoding.n_predict`); P100 memory for Qwen3-4B BF16 | _pending_ |
| 2 | Primary M2 outcome: step-level failures (C6 classes per step) instead of episode accuracy? | Yes: episode accuracy is 0% at 0.6B and ≤10% at 4B on these tasks (Workbook 2, Table 36b) | H1 test unit; power analysis; inclusion rule wording | _pending_ |
| 3a | Add the class "missing call with a direct answer" to codebook v0? | Yes (58 hand-labelled Q4_0 outputs: 49 answer directly instead of calling) | `docs/codebook_v0.md`, `testbed/classifier.py` | _pending_ |
| 3b | Does a tool that returns an error payload without raising (e.g. "No such file") get its own class? | Yes, separate from `exec_error` (the executor's own error); found by the Oct 27 rater check | same | _pending_ |
| 4 | τ² simulated-user host | An A100/H100 job on the cluster (compute nodes cannot reach an outside VM; no cost); the GCP L4 VM is the fallback, and its quota request can be dropped | C4, S5, the τ² configs; GCP item in Table 37a | _pending_ |
| 5 | Reference decoding | 4 sampled trials (Qwen3 card: T 0.7, top_p 0.8, top_k 20) + 1 greedy pass per task and configuration (Workbook Ch. 3) | `ref_*` configs, run volume | _pending_ |
| 6 | Keep only τ² tasks whose gold actions write to the database | Yes (an agent that does nothing passes the others) | C1 task list for τ² | _pending_ |
| 7 | Backend as a recorded factor in every comparison; parity test greedy | Yes; FR-E17 parity is running (greedy, 200 fixed prompts per model) and decides pooling (≥ 99% identical) | Report and analysis code | _pending_ |
| 8 | Depth bins (step index within the episode) | 0–1, 2–3, 4–6, 7+ (≈ quartiles of BF16 steps in the Oct 28 runs for all three Qwen3 sizes: 26 / 25 / 29 / 20%) | H1 concentration test, model B of the mixed-effects analysis, damage map | _pending_ |
| 9 | M4 date: 2026-12-18 or 2026-12-11 | Either (37 days of float on that path) | Schedule only | _pending_ |
| 10 | Vision-language extension (FR-O6) as optional 295B scope, go/no-go at M6 | Yes | 295B scope | _pending_ |
| 11 | **Inclusion rule vs a step-level outcome.** The registered rule drops tasks the 16-bit reference never solves and moves a model with < 120 included multi-turn tasks to single-turn-only analysis. On the Oct 28 runs (2 sampled BF16 trials, 200 BFCL base tasks) Qwen3-0.6B solves 11 tasks and Qwen3-4B 27 (`python -m testbed.inclusion`), so every model would leave the multi-turn analysis. | If decision 2 makes step-level failures primary: keep all tasks for the step-level tests (failures per step are defined whether or not the episode succeeds), apply the success-based inclusion only to the secondary episode-success analysis, and drop the 120-task switch for the primary outcome. Report the solved-task counts. | Pre-registration §5, power analysis, Stage 1 → main-grid task lists | _pending_ |

## Facts measured for these decisions (2026-11-08)

- **Cluster per-user limit.** Every GPU partition (`gpuqs`, `gpuqm`, `gpuql`) has QoS `gpu3nodes`: at most 2 nodes
  per user. P100 nodes have 1 GPU, A100 nodes 4 (cs001 and cs002 are marked bad), H100 nodes 1. One account
  therefore runs at most 2 P100 jobs, or up to 8 single-GPU jobs on two A100 nodes. The NFR-5 pilot measures
  what this gives in episodes per hour (`results/m2/throughput.json`). CPU partitions have no such QoS
  (`defq` allows 2 CPUs per job).
- **CPU-backend speed.** One Qwen3-0.6B Q4_0 BFCL episode (8 steps, 3–8k-token prompts, prompt cache off) took
  11 minutes on 4 CPU cores. CPU-backend correctness runs need many cores per job or few episodes.
