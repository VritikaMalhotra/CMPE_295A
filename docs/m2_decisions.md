# M2 decisions: open items before the analysis plan is frozen

Status 2026-11-10: **all proposed, none decided.** These were asked at the 2026-10-30 advisor review (Workbook 2,
"Decisions requested from the advisor"); the repository has no notes from that review. Ekant proposed an answer to
all eleven on 2026-11-08 (column "Proposed answer"). Each becomes a decision only when the **advisor confirms**
it (planned: Fri 2026-11-13), and only then is the pre-registration (`docs/preregistration_m2.md`, still DRAFT)
frozen and filed. If the advisor decided differently on Oct 30, that wins. Two Oct 30 asks (slide 16 #1 and #4)
and three questions from the `m2-phase1` review are added below as pending.

| # | Question | Proposal | What depends on it | Proposed answer (not yet decided) |
|---|---|---|---|---|
| 1 | Context budget: bfcl parity (ctx 40,960, cap 4,096 new tokens per step) or the Workbook's FR-E3 (16,384 / 1,024)? | Keep 40,960 / 4,096: matches bfcl-eval's local handler, so our verdicts compare with the official evaluator (FR-E2); every run so far used it. Update FR-E3 instead. | All reference and main-grid configs (`server.ctx`, `decoding.n_predict`); P100 memory for Qwen3-4B BF16 | **40,960 / 4,096** (Ekant, 2026-11-08; advisor to confirm) |
| 2 | Primary M2 outcome: step-level failures (C6 classes per step) instead of episode accuracy? | Yes: episode accuracy is 0% at 0.6B and ≤10% at 4B on these tasks (Workbook 2, Table 36b) | H1 test unit; power analysis; inclusion rule wording | **Step-level primary**, episode success secondary (Ekant, 2026-11-08; advisor to confirm) |
| 3a | Add the class "missing call with a direct answer" to codebook v0? | Yes (58 hand-labelled Q4_0 outputs: 49 answer directly instead of calling) | `docs/codebook_v0.md`, `testbed/classifier.py` | **Add** to codebook v0 (implemented, on by default; Ekant, 2026-11-08; advisor to confirm) |
| 3b | Does a tool that returns an error payload without raising (e.g. "No such file") get its own class? | Yes, separate from `exec_error` (the executor's own error); found by the Oct 27 rater check | same | **Own class** `tool_error_payload` (implemented, on by default; Ekant, 2026-11-08; advisor to confirm) |
| 4 | τ² simulated-user host | An A100/H100 job on the cluster (compute nodes cannot reach an outside VM; no cost); the GCP L4 VM is the fallback, and its quota request can be dropped | C4, S5, the τ² configs; GCP item in Table 37a | **Cluster A100/H100 job**; drop the GCP L4 quota request (Ekant, 2026-11-08; advisor to confirm). Built 2026-11-08: `slurm/simuser.sbatch` + `scripts/tau2_submit.sh`, endpoint file hand-off (`testbed/simuser_service.py`); not yet run end to end on the cluster |
| 5 | Reference decoding | 4 sampled trials (Qwen3 card: T 0.7, top_p 0.8, top_k 20) + 1 greedy pass per task and configuration (Workbook Ch. 3) | `ref_*` configs, run volume | **4 sampled + 1 greedy** (Ekant, 2026-11-08; advisor to confirm) |
| 6 | Keep only τ² tasks whose gold actions write to the database | Yes (an agent that does nothing passes the others) | C1 task list for τ² | **Write tasks only** (201 of 278) (Ekant, 2026-11-08; advisor to confirm) |
| 7 | Backend as a recorded factor in every comparison; parity test greedy | Yes; FR-E17 parity is running (greedy, 200 fixed prompts per model) and decides pooling (≥ 99% identical) | Report and analysis code | **Backend as a factor**; pool only if FR-E17 parity ≥ 99% (Ekant, 2026-11-08; advisor to confirm) |
| 8 | Depth bins (step index within the episode) | 0–1, 2–3, 4–6, 7+ (≈ quartiles of BF16 steps in the Oct 28 runs for all three Qwen3 sizes: 26 / 25 / 29 / 20%) | H1 concentration test, model B of the mixed-effects analysis, damage map | **0–1, 2–3, 4–6, 7+** by step index within the episode (Ekant, 2026-11-08; advisor to confirm); `testbed/mixed_effects.py` `DEPTH_BINS` |
| 9 | M4 date: 2026-12-18 or 2026-12-11 | Either (37 days of float on that path) | Schedule only | **Keep 2026-12-18** (Ekant, 2026-11-08; advisor to confirm) |
| 10 | Vision-language extension (FR-O6) as optional 295B scope, go/no-go at M6 | Yes | 295B scope | **Optional 295B scope, go/no-go at M6** (Ekant, 2026-11-08; advisor to confirm) |
| 11 | **Inclusion rule vs a step-level outcome.** The registered rule drops tasks the 16-bit reference never solves and moves a model with < 120 included multi-turn tasks to single-turn-only analysis. On the Oct 28 runs (2 sampled BF16 trials, 200 BFCL base tasks) Qwen3-0.6B solves 11 tasks and Qwen3-4B 27 (`python -m testbed.inclusion`), so every model would leave the multi-turn analysis. | If decision 2 makes step-level failures primary: keep all tasks for the step-level tests (failures per step are defined whether or not the episode succeeds), apply the success-based inclusion only to the secondary episode-success analysis, and drop the 120-task switch for the primary outcome. Report the solved-task counts. | Pre-registration §5, power analysis, Stage 1 → main-grid task lists | **Episode-success analysis only**; all tasks for step-level tests; no 120-task switch for the primary outcome (Ekant, 2026-11-08; advisor to confirm) |

### Pending, added 2026-11-10

| # | Question | Proposal | What depends on it | Status |
|---|---|---|---|---|
| S16-1 | Oct 30 slide 16 #1: accept M1 as reached under `local=true` on Colab (cluster jobs were blocked until 2026-10-26), with the cluster criteria (two members' cluster accounts, S10) moved to M2 | Yes; the cluster criteria were then met for M2: S10 104/104 (2026-11-03), Ekant's clean-clone check on his own account (2026-11-08) | M1 record in Workbook 2 / the report | **pending** (advisor) |
| S16-4 | Oct 30 slide 16 #4: the dispersion convention (± 1 SD over the 4 sampled trials on every bar, 95% intervals for every effect, SD = 0 stated for greedy runs) and the mixed-effects analysis as a registered secondary analysis (lme4 in R, robustness rule in Workbook 2 Ch. 7) | Yes, as written in pre-registration §6; the planted-effect check of the R models passes (`results/m2/planted_effect.json`, 2026-11-10) | Pre-registration §6; every figure | **pending** (advisor) |
| 12 | Which step-level failure classes are confirmatory | Any failure, `exec_error` and `tool_error_payload` (powered at 2,000 episodes in every model, `results/m2/power_table.json`); `unknown_tool` and `missing_call` exploratory unless the Stage 1 design effect is near the episode-level value; the rest pooled or exploratory | Pre-registration §6, the Holm family | **pending** (advisor) |
| 13 | CPU-backend scope: parity is below 99% for every model, so every RQ1 ratio is reported per backend, but one Qwen3-0.6B CPU episode takes about 11 minutes on 4 cores and `defq` gives 2 CPUs per job, so a full CPU arm (2,000 sampled episodes per configuration) does not fit | CPU agent runs on a fixed subset stated before filing (e.g. Qwen3-0.6B and 1.7B, BFCL base, greedy pass) | Pre-registration §3, the run plan | **pending** (advisor) |
| 14 | NFR-5 is not met under QoS `gpu3nodes` (2 GPU nodes per user) | Each member submits a share of the main grid from their own account (`shard=i/n`), or the advisor asks the administrator for a higher limit | Run plan for M3/M4 | **pending** (advisor) |
| 15 | A simulated-user HTTP 500 on one request (gpt-oss output llama-server cannot parse, "does not match the expected peg-native format"). Until 2026-11-11 the runner stopped all τ² scheduling on any simulator error and recorded no attempt, so one SmolLM3 telecom request (14 of 14 attempts failed) held 207 episodes pending | Treat it as an infrastructure error of that episode, as pre-registration §5 already says for infrastructure errors: retried up to three times with the same seed, then excluded and listed. Outages (unreachable, 502/503/504) still stop τ² scheduling. Implemented in `9d8b13f` and used for SmolLM3 telecom (job 113054); evidence in `docs/cluster.md`, A2 | Pre-registration §5; every τ² run | **proposed** (Yashashav, 2026-11-11; advisor to confirm) |
| 16 | Empty simulated-user replies that fill gpt-oss's 32,768-token slot. 34 of SmolLM3's 46 exclusions end with `finish_reason = length` and the slot full; §5 excludes empty replies but scores "the simulated user runs out of context (its 32,768-token slot)" as a failure. 33 of the 46 follow an agent message that writes a tool call as text, so the exclusion depends on the agent | Keep excluding them for now (registered rule; SmolLM3 then misses A2's 99%). Ask whether slot-full empty replies should count as context overflow (scored as failure), since excluding agent-caused episodes can hide quantization damage (Qwen3-0.6B: BF16 3, Q4_0 4, Q4_K_M 1, IQ4_XS 7 exclusions) | Pre-registration §5; A2; every τ² comparison | **pending** (advisor) |

## Facts measured for these decisions (2026-11-08)

- **Cluster per-user limit.** Every GPU partition (`gpuqs`, `gpuqm`, `gpuql`) has QoS `gpu3nodes`: at most 2 nodes
  per user. P100 nodes have 1 GPU, A100 nodes 4 (cs001 and cs002 are marked bad), H100 nodes 1. One account
  therefore runs at most 2 P100 jobs, or up to 8 single-GPU jobs on two A100 nodes. The NFR-5 pilot measures
  what this gives in episodes per hour (`results/m2/throughput.json`). CPU partitions have no such QoS
  (`defq` allows 2 CPUs per job).
- **CPU-backend speed.** One Qwen3-0.6B Q4_0 BFCL episode (8 steps, 3–8k-token prompts, prompt cache off) took
  11 minutes on 4 CPU cores. CPU-backend correctness runs need many cores per job or few episodes.

## Review of `m2-phase1` (4cbfb49): where each follow-up stands (2026-11-10)

| # | Follow-up | Status |
|---|---|---|
| 1 | Rerun S10 on the merged branch (array template now quotes `only_config`) | Done: passed on `week-2026-11-13`, 104/104 episodes, no missing or duplicate episodes (array 111842 and its resubmissions; same experiment hash `4375dbd0be` as 2026-11-03; `results/m2/s10_check_rerun.json`, commit d1626fd) |
| 2 | "proposed" until the advisor confirms; pre-registration stays DRAFT | Done (this file; pre-registration status line) |
| 3 | Notes of the 2026-10-30 advisor review | Left: only the members who attended can write them |
| 4 | Confirmatory failure classes | Added as pending item 12 and in pre-registration §6 |
| 5 | NFR-5 under the 2-node QoS | Added as pending item 14 |
| 6 | The two 2026-11-09 simulated-user rules in pre-registration §5 | Done (pre-registration §5) |
| 7 | Slide 16 #1 and #4 | Added as pending items S16-1 and S16-4 |
| 8 | CPU-backend scope | Added as pending item 13 and in pre-registration §3 |
| 9 | A recorded run of the R script | Done: planted-effect check recorded (`results/m2/planted_effect.json`, job 111860, pass); one real-data run, Qwen3-0.6B BF16 vs Q4_K_M (`results/m2/mixed_effects_qwen3-0.6b.json`, job 112507, commit c5c5590) |
| 10 | Prompt handlers for SmolLM3 and LFM2.5 | Written from each model's chat template (`testbed/prompt_formats.py`, branch `wk11-ekant`); their BF16 references follow |
