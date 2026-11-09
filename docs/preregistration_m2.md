# Pre-registration draft (M2): quantization damage and repairs in small LLM agents

**Status: DRAFT for team and advisor review. Not filed.** Filing on OSF is done by the team/advisor; the
registration ID then goes into the Experiment entity (C11 manifest). Items marked _[pending]_ depend on
`docs/m2_decisions.md` or on Stage 1 results and must be filled before filing. Source of every rule:
Workbook 2 (Ch. 3 Hypotheses, Ch. 7 Statistical Analysis Plan); deviations from it are marked.

## 1. Study information
- Title: Where 4-bit quantization damages small tool-calling agents, and what each repair costs.
- Authors: Ekant Kapgate, Vritika Malhotra, Yashashav Devalapalli Kamalraj; advisor Dr. Bertin Cordova-Diba (SJSU).
- Research questions:
  - RQ1: Where does 4-bit damage concentrate in small-agent conversations, by step type and by conversation depth?
  - RQ2: How do grammar-constrained decoding (R1), added bits (R2: uniform, layer-selective, step-selective) and
    QAD retraining (R3) compare, each with its control, on the same models and benchmarks?
  - RQ3: Do repairs validated on overall scores fix the underlying failures, or only the score?
  - RQ4: What does each repair cost to deploy (memory, latency, extra tokens, energy) on llama.cpp's real 4-bit
    code paths, CUDA and CPU, for the same GGUF files?

## 2. Hypotheses (registered as written in Workbook 2, Ch. 3)
- **H1 (RQ1).** 4-bit formats raise per-step failures above the 16-bit reference (amplification ratio > 1), and
  the excess concentrates in tool-call steps and at greater depth. No directional prediction for CUDA vs CPU.
- **H2 (RQ2).** Each repair reduces the excess failures relative to its control. R1 removes format failures
  (malformed call, hallucinated tool, schema-invalid arguments) but not semantic ones (wrong tool, wrong value,
  policy violation); QAD beats the train-longer control. No directional prediction for step-selective routing
  vs Q8_0 uniform.
- **H3 (RQ3).** Repairs close more of the score gap than of the failure gap, and their score gains shrink when
  τ² is re-scored with an error budget of 2.
- **H4 (RQ4).** R1 adds no weight memory and ≤ 5% to median per-step decode latency on CUDA; overhead, if any,
  falls on TTFT (grammar compilation); larger overhead on the CPU backend. Q8_0 ≈ 1.9 × Q4_0 weight memory;
  R2c ≈ 2.9 × Q4_0 plus re-prefill tokens; R3 adds no inference cost (its cost is training GPU-hours).

## 3. Design
- Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, LFM2.5-1.2B-Instruct, LFM2.5-2.6B (BF16 GGUF references;
  conversions and KL checks in `results/m2/kl_*.json`, files and SHA-256 in `results/m2/models_manifest.json`).
  _[pending: the testbed has a prompt handler only for Qwen3; SmolLM3/LFM2.5 handlers before their runs]_
- Formats: Q4_0 (`--pure`), Q4_K_M, IQ4_XS (imatrix) at 4 bits; Q5_0, Q6_K, Q8_0 for R2.
- Runtime: llama.cpp `d812350` (v0.6.0), raw `/completion` endpoint, one slot, prompt cache off, CUDA (P100) and
  CPU backends on the same GGUF files.
- Benchmarks: BFCL v4 multi-turn (base, missing-function, missing-parameter; 100 tasks each, fixed-seed sample)
  and τ²-bench-verified, tasks whose gold agent actions write to the database (201 of 278; decision 6,
  Ekant 2026-11-08, advisor to confirm). Scoring with the official, unmodified checkers.
- Decoding (decision 5, Ekant 2026-11-08, advisor to confirm): 4 sampled trials (T 0.7, top_p 0.8, top_k 20, min_p 0) with fixed seeds plus
  1 greedy pass per task and configuration. Context (decision 1, same status): 40,960 tokens, ≤ 4,096 new tokens per
  step (budget `min(4096, ctx − prompt − 2)`, as bfcl-eval's local handler).
- Backend parity (FR-E17): greedy outputs on 200 fixed prompts per model, BF16 and Q4_0, CUDA vs CPU
  (`results/m2/parity_*.json`). Results are pooled across backends only if ≥ 99% of outputs are token-identical;
  otherwise every RQ1 ratio is reported per backend.

## 4. Measured variables
- Unit of analysis: the task. Trials are averaged within a task before paired tests; the greedy pass is
  analysed separately (flip rates, backend agreement).
- Primary outcome (decision 2, Ekant 2026-11-08, advisor to confirm): failures per agent step, by C6 failure class (codebook v0,
  `docs/codebook_v0.md`, with `tool_error_payload` and `missing_call_direct_answer` (decisions 3a/3b), commit _[hash]_). Episode success is secondary.
- Step type: the kind of output a step produces: tool call, text reply to the user, or end of conversation.
- Depth: the step's index within the episode, in bins 0–1, 2–3, 4–6, 7+ (decision 8, Ekant 2026-11-08, advisor
  to confirm; `testbed/mixed_effects.py` `DEPTH_BINS`).
- Costs (RQ4): weight memory, TTFT, decode tokens/s, extra tokens, NVML energy (CUDA); exclusive-node protocol.

## 5. Inclusion and exclusion (fixed in advance)
Decision 11 (Ekant 2026-11-08, advisor to confirm): the rules below apply to the secondary episode-success analysis only. The primary step-level tests use all tasks (failures per step are defined whether or not the episode succeeds), and the 120-task switch does not apply to them. Solved-task counts per model are reported (Oct 28 runs, 2 trials: Qwen3-0.6B 11, Qwen3-4B 27 of 200).
- Tasks the 16-bit reference fails in all 4 sampled trials are excluded per model from confirmatory tests; counts
  and an all-task sensitivity analysis are reported. The inclusion set is stored per model
  (`results/m2/inclusion_<model>.json`, with the reference experiment hash) _[pending: Stage 1 reference runs]_.
- A model with fewer than 120 included multi-turn tasks moves to single-turn-only confirmatory analysis; its
  multi-turn cells are reported descriptively.
- Infrastructure errors (backend crash, simulator timeout, node failure) are retried up to three times with the
  same seed; episodes that still fail are excluded and listed (implemented: `testbed/logger.py`
  `MAX_INFRA_ATTEMPTS = 4`, listed in each report). Episodes interrupted by a Slurm walltime limit are rerun
  from the start (resumed from the last finished episode), not excluded.
- Episodes that hit the step or error limit are outcomes, not exclusions.

## 6. Analysis plan
- Effect sizes: rate ratio (amplification) and difference in failures per 100 steps, with 95% CIs from a paired
  cluster bootstrap over tasks (10,000 resamples, all trials of a task resampled together).
- H1: paired permutation test (10,000 permutations) on per-task failure counts per model × format × backend;
  concentration by a Poisson GEE of failures on precision × step type and precision × depth, clustered by task.
- H2: paired permutation tests on task-level success rates and failure counts; McNemar's exact test on the
  greedy pass (secondary). "R2c matches Q8_0" only if TOST with a ±3-point margin rejects.
- H3: paired bootstrap of (gap closed on success − gap closed on failures); the 95% CI must exclude 0.
- H4: ≥ 10 interleaved repetitions; medians with bootstrap CIs; TOST with ±5% margin for every "no overhead"
  claim; Pareto frontier of failure reduction against each cost.
- Multiple comparisons: Holm within each RQ family; family sizes fixed here _[pending: list the contrasts per
  family; Workbook: 10 contrasts in the RQ2 family]_.
- Mixed-effects analysis (registered secondary, reported for every primary contrast): lme4 in R 4.4
  (`analysis/mixed_effects.R`, planted-effect check in the test suite): (A) failures per episode, Poisson with
  offset log(steps), negative binomial when Pearson dispersion > 1.5; (B) step-level failure, binomial,
  precision × step type + precision × depth + (1 | task/episode); (C) episode success, binomial,
  configuration + (1 | task). A conclusion is robust when the mixed model agrees with the bootstrap in direction
  and in Holm-corrected significance; otherwise reported as not robust with both estimates. Convergence
  fallback: bobyqa, then drop the episode intercept, then report the failure.
- Dispersion: every mean shown with ± 1 SD over the 4 sampled trials and the 95% bootstrap interval for every
  effect; greedy runs show 95% Wilson intervals over tasks (SD = 0).
- Power: ~330 failure events of one type per paired comparison for 80% power at a 1.5× ratio (α = 0.05/10), times
  the design effect D. Recomputed from the BF16 references (2026-11-09, `scripts/power_table.py`,
  `results/m2/power_table.json`; table in `docs/cluster.md`): with 2,000 sampled episodes per configuration,
  any failure, exec_error and tool_error_payload are detectable in every model even at the task-level D
  (3-21); unknown_tool and missing_call only if D is near the episode-level value (1-6). Rare failure types
  (invalid_args, malformed, forced_quit, truncated) are pooled or reported as exploratory.
- Any analysis whose code hash differs from the registered one is labelled exploratory by C10.

## 7. What is frozen with the registration
- Analysis code commit: _[hash]_ (C10 statistics, `analysis/mixed_effects.R`, report generator).
- Codebook v0 commit: _[hash]_; codebook v1 (after the κ study, M3) is filed as an amendment.
- Configs: reference and main-grid experiment files with their experiment hashes _[list]_.

## 8. Data already seen (honest disclosure)
All runs before registration are development data and excluded from confirmatory tests: the October pilot
(Colab), the M1 skeleton (Oct 24–26), the Oct 28 200-task damage runs (Qwen3 0.6B/1.7B/4B), the KL checks,
the FR-E17 parity runs and the NFR-5 throughput pilot. They informed the primary outcome (step level), the
depth bins and the backend-as-factor decision.
