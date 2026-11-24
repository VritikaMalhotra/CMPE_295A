# OSF pre-registration: quantization damage and repairs in small tool-calling agents

**Status: FILED on OSF Registries, 2026-11-22** (OSF Preregistration template, https://osf.io/xwq5y, embargoed until
2027-11-22; record `results/m3/osf_registration.json`). Known issue: OSF dropped the text `<model>` from the path
`results/m3/inclusion_holdout_<model>.json` in two fields of the filed form; the attached PDF has it intact.
Earlier status line: filing version, draft 2026-11-19 (Ekant), for team and advisor review. Written to the OSF Preregistration
template. It replaces `docs/preregistration_m2.md` (the M2 draft), whose rules it keeps except where marked
**[changed]**. Every decision it cites is in `docs/m2_decisions.md`: decisions 1–17 were confirmed by the advisor
(recorded 2026-11-16), decision 18 is proposed (Ekant, 2026-11-19). The checklist in §9 lists what must be done
on the day of filing. **OSF project:** https://osf.io/xwq5y/ (created 2026-11-22; the registration is filed from it).

**What changed since the M2 draft, and why.** The H1 damage grid (BF16 against Q4_0, Q4_K_M and IQ4_XS for six
models on the 501 development tasks, CUDA) was run to completion and analysed (damage map v1, 2026-11-19) before
this registration. Those data cannot test H1. **[changed]** H1 is therefore tested on a **held-out BFCL sample**
(the 300 BFCL multi-turn tasks outside the development sample) with **new seeds**, and the finished grid is
reported as exploratory (§8). The repairs (RQ2–RQ4) have not been run on any task and are confirmatory on the
same holdout.

## 1. Study information

**Title.** Where 4-bit quantization damages small tool-calling agents, and what each repair costs.

**Authors.** Ekant Kapgate, Vritika Malhotra, Yashashav Devalapalli Kamalraj (San José State University);
advisor Dr. Bertin Cordova-Diba.

**Description.** Small open language models (0.6–4B parameters) run as tool-calling agents on devices only after
4-bit weight quantization. We measure, step by step, where 4-bit formats raise agent failures relative to a 16-bit
reference on llama.cpp's real kernels, and we compare three repairs (grammar-constrained decoding, added bits,
quantization-aware distillation), each against its control, on failures, scores and deployment cost.

**Research questions.**
- RQ1: Where does 4-bit damage concentrate in small-agent conversations, by step type and conversation depth?
- RQ2: How do grammar-constrained decoding (R1), added bits (R2: uniform, layer-selective, step-selective) and QAD
  retraining (R3) compare, each with its control, on the same models and tasks?
- RQ3: Do repairs validated on overall scores fix the underlying failures, or only the score?
- RQ4: What does each repair cost to deploy (memory, latency, extra tokens, energy) on llama.cpp's real 4-bit code
  paths, CUDA and CPU, for the same GGUF files?

**Hypotheses** (as in Workbook 2, Ch. 3; tests in §6).
- **H1 (RQ1).** 4-bit formats raise per-step failures above the 16-bit reference (amplification ratio > 1), and
  the excess concentrates in tool-call steps and at greater depth. No directional prediction for CUDA vs CPU.
- **H2 (RQ2).** Each repair reduces the excess failures relative to its control. R1 removes format failures
  (malformed call, unknown tool, schema-invalid arguments) but not the others; QAD beats the train-longer control.
  No directional prediction for step-selective routing vs Q8_0 uniform, or for layer-selective vs uniform bits.
- **H3 (RQ3).** Repairs close more of the success gap than of the failure gap.
- **H4 (RQ4).** R1 adds no weight memory and ≤ 5% to median per-step decode latency on CUDA; any overhead falls on
  time to first token (grammar setup), and the overhead is larger on the CPU backend. Q8_0 ≈ 1.9 × Q4_0 weight
  memory; R2c ≈ 2.9 × Q4_0 plus re-prefill tokens; R3 adds no inference cost.

## 2. Design plan

**Study type.** Experiment: every condition is run on the same tasks with the same seeds, and conditions differ
in exactly one factor (model file, backend or repair). No human participants.

**Blinding.** Not applicable to model runs. The analysis code is frozen at registration (§7); C10 labels any
analysis run with a different code hash as exploratory.

**Factors.**
- Model: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, LFM2.5-1.2B-Instruct, LFM2.5-2.6B. Core models (all
  repairs, including R3): Qwen3-0.6B, Qwen3-1.7B, LFM2.5-1.2B-Instruct.
- Format: BF16 (reference); Q4_0 (`--pure`), Q4_K_M, IQ4_XS (imatrix) as the damage conditions. Files and SHA-256
  in `results/m3/models_manifest.json`.
- Backend: llama.cpp `d812350` CUDA (A100) and CPU, the same GGUF files; raw `/completion`, one slot, prompt
  cache off (NFR-2).
- Repair (from Q4_0, core models): none; R1 grammar (GBNF "free" grammar in the model's native call syntax,
  `docs/r1_grammars.md`); R2a uniform Q5_0, Q6_K, Q8_0; R2b layer-selective at 5.5 bits per weight (tensors ranked
  by single-tensor KL on held-out agent conversations); R2c step-selective, Q4_0 and Q8_0 servers, tool-call steps
  routed to Q8_0 (policy 1; the entropy policy is exploratory); R3 QAD-KL, QAD-KL+CE and the train-longer control
  (QAD-KL+CE with fake quantization off, then plain Q4_0), full-parameter on identical data, order and steps.
  **Training settings (fixed by the 2026-11-21 pilot on held-out KL only, before any R3 evaluation):** fp32 master
  weights with BF16 autocast; AdamW (β 0.9/0.95, no weight decay), peak learning rate 1e-5, 50 warmup steps, cosine
  decay; 32,768 tokens per step (2 × 8 × 2,048); one pass over the packed QAD mix (about 51M tokens, ≈ 1,570 steps);
  KL+CE loss = KL + 0.1 × CE (weight 0.5 made the student drift from its teacher, worst on LFM2.5); seed 295.
  Fake quantization covers every tensor `llama-quantize --pure Q4_0` quantizes (including a tied embedding/output
  matrix); export through an f32 GGUF is checked code for code (`train/qad.py`, `train/export_r3.py`).
- Decoding (decision 5): 4 sampled trials (T 0.7, top_p 0.8, top_k 20, min_p 0) with fixed seeds plus 1 greedy
  pass per task and configuration; context 40,960 tokens, ≤ 4,096 new tokens per step (decision 1).

## 3. Sampling plan

**Existing data.** Registration after some data collection, before any confirmatory data: the confirmatory tasks
and seeds below have not been run for any model or condition, except the overlap disclosed in §8.

**Confirmatory task set [changed].** The **BFCL v4 multi-turn holdout**: every task of `multi_turn_base`,
`multi_turn_miss_func` and `multi_turn_miss_param` (200 each) not in the development sample (seed 295,
`testbed/adapter.py`), i.e. 100 per category, **300 tasks**. List and source-file hashes:
`results/m3/holdout_tasks_bfcl.json` (SHA-256 of the newline-joined IDs `a7c6c695…49ed8e64`). Scored with
bfcl-eval's unmodified multi-turn checker. τ²-bench-verified is not in the confirmatory set: all 201 of its write
tasks were used in development (§8).

**Seeds [changed].** Sampled trials use seeds 5, 6, 7 and 8 (development used 1–4); the greedy pass uses seed 0.

**Confirmatory runs.**

| Stage | Configurations | Episodes |
|---|---|---|
| C1. H1 references and damage (CUDA) | 6 models × {BF16, Q4_0, Q4_K_M, IQ4_XS} | 24 × 1,500 = 36,000 |
| C2. H1 CPU subset (decision 13) | Qwen3-0.6B, Qwen3-1.7B × {BF16, Q4_0, Q4_K_M, IQ4_XS}, CPU, `multi_turn_base` holdout (100 tasks), greedy only | 800 |
| C3. Repairs (CUDA) | 3 core models × {R1, Q5_0, Q6_K, Q8_0, R2b, R2c, QAD-KL, QAD-KL+CE, train-longer} | 27 × 1,500 = 40,500 |
| C4. Costs (RQ4) | Every C1/C3 configuration on both backends, single-stream blocks on an exclusive node | ≥ 10 interleaved repetitions of a fixed 20-task cost set |

Stage order: C1 BF16 references first (the inclusion set for the secondary success analysis is fixed from them,
§5), then the 4-bit cells, C2, C3 and C4. The R2b ranking and the R3 checkpoints are produced before C3 from data
disjoint from both benchmarks (`docs/qad_data.md`; no 13-gram overlap).

**Stopping rule.** Every run specification is run to completion (finished or excluded under §5); no interim
looks at 4-bit or repair outcomes on the holdout. Monitoring is limited to run status (finished, pending,
excluded counts from `scripts/run_status.py`).

**Power** (`results/m3/power_table_bfcl_holdout.json`, from the BF16 development references restricted to BFCL;
1,200 sampled episodes per configuration; 80% power for a 1.5× ratio at α = 0.05/10, about 330 events × design
effect D). **Any failure** is detectable in all six models at the conservative task-level D (≤ 489 episodes
needed). **tool_error_payload** is detectable in five models (476–633 episodes) but not SmolLM3-3B (2,776).
**exec_error** is not detectable on BFCL alone in any model (it is mostly a τ² failure in development).
Other classes: `missing_call_direct_answer` and `missing_call` only in SmolLM3-3B (181 and 634), `invalid_args`
only in LFM2.5-1.2B (1,191). At the H1 family size (18 tests, α = 0.05/18) the requirement rises by about 10%;
any failure stays within 1,200 episodes for every model.

## 4. Variables

**Unit of analysis.** The task. Sampled trials are averaged within a task before paired tests; the greedy pass is
analysed separately (flip rates, backend agreement, the CPU subset).

**Primary outcome (decision 2).** Failures per agent step, by failure class, assigned by the C6 classifier,
codebook v0 (`docs/codebook_v0.md`, including `tool_error_payload` and `missing_call_direct_answer`, decisions
3a/3b). Failure classes: `truncated`, `malformed`, `forced_quit`, `unknown_tool`, `invalid_args`, `exec_error`,
`tool_error_payload`, `missing_call`, `missing_call_direct_answer`; **any failure** is the union.

**Confirmatory classes [changed from decision 12, because the holdout is BFCL only].** H1 and H2: any failure
(primary). H1 secondary family: `tool_error_payload` in the five models where it is powered. H2 contrast 1:
format failures (`malformed` + `unknown_tool` + `invalid_args`). All other classes, and `exec_error`, are
exploratory on the holdout.

**Secondary outcome.** Episode success (bfcl-eval verdict), mean ± 1 SD over the four sampled seeds.

**Step type.** Tool-call step (at least one call parsed or attempted) vs text step (reply, abstention or end of
turn). **Depth (decision 8).** Step index within the episode, bins 0–1, 2–3, 4–6, 7+.

**Costs (RQ4).** Weight memory (file and resident), time to first token, decode tokens/s, extra tokens
(re-prefill for R2c, grammar-forced tokens for R1), GPU energy per step (NVML, CUDA only), all on an
exclusively allocated node.

## 5. Inclusion and exclusion (fixed in advance)

- **Step-level tests use every task (decision 11).** The success-based inclusion rule applies only to the
  secondary episode-success analysis: tasks the BF16 reference fails in all four sampled trials are excluded per
  model from that analysis, with counts and an all-task sensitivity analysis reported; the inclusion set is
  stored per model (`results/m3/inclusion_holdout_<model>.json`) before any 4-bit holdout cell is analysed.
- **Infrastructure failures** (backend crash, server error, node failure) are retried with the same seed up to
  four attempts in all; an episode that fails all four is excluded and listed in the report
  (`testbed/logger.py`, `MAX_INFRA_ATTEMPTS = 4`). Episodes interrupted by a Slurm walltime limit are rerun, not
  excluded.
- **Outcomes, not exclusions:** episodes that hit bfcl's 20-step turn limit or the token budget.
- The τ² simulated-user rules (decisions 15, 16 and 18) do not apply to the BFCL holdout; they govern the
  exploratory τ² analyses (§8).
- **Pairing.** A task enters a paired contrast only with the trials both configurations finished; the number of
  pairs per contrast is reported.

## 6. Analysis plan

**Effect sizes.** Rate ratio of pooled per-step failure rates (amplification) and the difference in failures per
100 steps, each with a 95% CI from a paired cluster bootstrap over tasks (10,000 resamples, all trials of a task
resampled together). Every mean is shown with ± 1 SD over the four sampled trials; greedy results with 95%
Wilson intervals over tasks (SD = 0).

**H1 (stage C1).**
- Damage, family H1 (18 tests): for each model × format (CUDA), a paired permutation test (10,000 permutations)
  on per-task any-failure rates, BF16 vs 4-bit; H1 is supported for a cell if the Holm-corrected p < 0.05 and the
  ratio > 1.
- Concentration, family H1-loc (6 tests): for each format, a Poisson GEE of failures on precision × step type and
  precision × depth bin (pooled over models, model as a fixed effect, clustered by task); supported if the
  interaction is significant after Holm and larger in tool-call steps / deeper bins.
- Secondary family H1-tep (15 tests): as the damage family for `tool_error_payload`, five models × three formats.
- CPU subset, family H1-cpu (6 tests, stage C2): Qwen3-0.6B and 1.7B × three formats, greedy, paired permutation
  over tasks; the CUDA vs CPU difference in the BF16→4-bit change is reported with a bootstrap CI (no directional
  prediction).
- Sensitivity: the H1 family re-run on the 200 never-seen tasks (`miss_func` and `miss_param` holdout), see §8.

**H2 (stage C3).** Family RQ2, 10 contrasts, Q4_0 base, pooled over the three core models (task × model as the
cluster, model as a stratum), outcome any failure per step unless stated:

| # | Contrast | Prediction |
|---|---|---|
| 1 | R1 vs Q4_0, format failures (malformed + unknown_tool + invalid_args) | R1 lower |
| 2 | R1 vs Q4_0, any failure | none (two-sided) |
| 3 | Q5_0 vs Q4_0 | Q5_0 lower |
| 4 | Q8_0 vs Q4_0 | Q8_0 lower |
| 5 | R2b vs Q5_0 (matched 5.5 bits per weight) | none (two-sided) |
| 6 | R2c vs Q4_0 | R2c lower |
| 7 | R2c vs Q8_0: equivalence on success, TOST ± 3 points | none; "matches" only if TOST rejects |
| 8 | QAD-KL vs Q4_0 | QAD lower |
| 9 | QAD-KL+CE vs Q4_0 | QAD lower |
| 10 | QAD-KL+CE vs train-longer control (both Q4_0) | QAD lower |

Paired permutation tests on per-task rates; McNemar's exact test on the greedy pass as a secondary test; Holm
within the family. Per-model estimates are reported for every contrast. Exploratory: Q6_K, the R2c entropy
policy, R1 at BF16 (the grammar × quantization interaction), R1 on the CPU backend, R1 and R2 on Q4_K_M, vendor
QAD checkpoints (LFM2.5 Q4_0).

**H3.** For R1, Q8_0 and QAD-KL+CE (family RQ3, 3 tests), the paired bootstrap over tasks of
(share of the BF16→Q4_0 success gap closed) − (share of the any-failure gap closed); supported if the 95% CI
excludes 0 in the predicted direction. The τ² error-budget re-scoring is exploratory (τ² is not in the
confirmatory set) **[changed]**.

**H4 (stage C4).** ≥ 10 interleaved repetitions per configuration and backend; medians with bootstrap CIs; TOST
with a ± 5% margin for every "no overhead" claim (R1 decode latency on CUDA; R3 latency vs Q4_0); a Pareto
frontier of failure reduction (from H2) against each cost.

**Mixed-effects analysis (registered secondary, S16-4).** lme4 in R 4.4 (`analysis/mixed_effects.R`): (A)
failures per episode, Poisson with offset log(steps), negative binomial if Pearson dispersion > 1.5; (B)
step-level failure, binomial, precision × step type + precision × depth + (1 | task/episode); (C) episode
success, binomial, configuration + (1 | task). A conclusion is robust when the mixed model agrees with the
bootstrap in direction and in Holm-corrected significance; otherwise it is reported as not robust with both
estimates. Convergence fallback: bobyqa, then drop the episode intercept, then report the failure.

**Multiple comparisons.** Holm within each family: H1 (18), H1-loc (6), H1-tep (15), H1-cpu (6), RQ2 (10),
RQ3 (3), RQ4 (one per "no overhead" claim, listed in the C4 plan before the cost runs). No correction across
families; every test outside these families is exploratory and labelled so.

**Inference criteria.** Two-sided α = 0.05 after Holm unless a direction is stated above (then one-sided tests
are not used; the direction is checked on the estimate).

## 7. What is frozen with the registration

- **Analysis code:** commit `1e13897` on branch `m3-damage-runs` (`testbed/stats.py`, `testbed/mixed_effects.py`,
  `analysis/mixed_effects.R`, `analysis/damage_table.py`, `analysis/which_tasks_fail.py`, `testbed/report.py`,
  `testbed/classifier.py`); planted-effect check `results/m2/planted_effect.json`.
- **Codebook v0:** `docs/codebook_v0.md` and `testbed/classifier.py` at the same commit. Codebook v1 (after the
  agreement study) is filed as an amendment before any confirmatory analysis uses it; v0 remains the registered
  primary.
- **Tasks:** `results/m3/holdout_tasks_bfcl.json`. **Models:** `results/m3/models_manifest.json`.
- **Runtime:** llama.cpp `d812350`; simulator not used in the confirmatory set.
- **Configs** (`configs/experiment/hold_*.yaml`, written by `scripts/make_holdout_configs.py`, commit `01e4272`;
  model files identical to the development configs). Experiment hashes from a dry run on 2026-11-19 (no episode
  run):

  | Model | C1 reference (`hold_ref_*`, 1,500 specs) | C1 damage (`hold_h1_*`, 4,500 specs) | C2 CPU (`hold_cpu_*`, 400 specs) |
  |---|---|---|---|
  | Qwen3-0.6B | f0d5c6bc24 | 972b0eee83 | 1c3d26ff22 |
  | Qwen3-1.7B | 8301d88c51 | 29717480e3 | 55534ce544 |
  | Qwen3-4B | 5f7ef89998 | d94749b2e9 | – |
  | SmolLM3-3B | fd67d053ee | 9df3e80d3a | – |
  | LFM2.5-1.2B-Instruct | 52714d278f | 596aa47141 | – |
  | LFM2.5-2.6B | 1aafc04549 | d939176fda | – |

  The runner refuses a task list whose IDs do not match the recorded SHA-256 (`testbed.run.load_task_ids_file`,
  `tests/test_task_ids_file.py`).
- **Stage C3 configs** (`configs/experiment/hold_c3_*.yaml`, commit `cf928c8`; written before filing, so no
  amendment is needed): per core model, R1 (free grammar on Q4_0), Q5_0, Q6_K, Q8_0, R2b, R2c, QAD-KL, QAD-KL+CE
  and the train-longer control, each 4 sampled seeds (5–8) + greedy, 13,500 specs. Experiment hashes: Qwen3-0.6B
  `7877ed438f`, Qwen3-1.7B `d09d07829b`, LFM2.5-1.2B-Instruct `3a7fd77597`. Model files and SHA-256:
  `results/m3/models_manifest.json` (R2b `R2b-LS5.5`; R3 `R3-QAD-KL-Q4_0-pure`, `R3-QAD-KLCE-Q4_0-pure`,
  `R3-CTRL-Q4_0-pure`, `R3-CTRL-BF16`); R3 training record `results/m3/r3_training.json`.
- **R2c implementation** (`testbed.backends.RoutingBackend`, commit `406d7f8`; `tests/test_r2c.py`): two
  llama-server processes on one GPU, Q4_0 and Q8_0, each with its own KV cache. Each step first requests one token
  from Q4_0 with the step's own seed and sampling settings; if it opens a tool call (`<tool_call>`, LFM2.5
  `<|tool_call_start|>`), Q8_0 generates the whole step, otherwise Q4_0 does. Every step records the copy that served
  it (`served_by`) and the probe's prompt tokens and time; C9 charges these and the second copy's memory to R2c.
  Smoke test on 5 development tasks (2026-11-22): Qwen3-0.6B 16 of 38 steps routed, LFM2.5-1.2B 21 of 40; no Q4_0
  step made a call after a non-call probe.

## 8. Data already seen (disclosure)

All of the following are development data, excluded from every confirmatory test:

- The October pilot (Colab), the M1 skeleton (first 20 `multi_turn_base` tasks, Oct 24–26), and the **Oct 28
  damage runs: all 200 `multi_turn_base` tasks** for Qwen3-0.6B, 1.7B and 4B (BF16, Q4_K_M and Q4_0 with 2 seeds;
  R1 on Q4_0 and Q8_0 with 1 seed; earlier harness and codebook). **The 100 `multi_turn_base` holdout tasks were
  therefore seen, in aggregate, for the three Qwen3 models.** The primary H1 analysis keeps all 300 holdout tasks;
  the registered sensitivity analysis repeats the H1 family on the 200 `miss_func` and `miss_param` holdout
  tasks, which no run has used.
- The M2 BF16 references and the full **H1 grid on the 501 development tasks** (300 BFCL + 201 τ²-verified;
  6 models × BF16, Q4_0, Q4_K_M, IQ4_XS; seeds 1–4 + greedy; CUDA), including the damage map v1
  (`docs/m3_damage_map_v1.md`). These are reported as exploratory, with the τ² rules of decisions 15, 16 and 18.
- KL divergence checks (WikiText-2) on both backends, the FR-E17 backend parity runs, the NFR-5 throughput pilot,
  the R2b rankings and builds for the three core models, and the QAD data build.
- **R3 training** (2026-11-21): the pilot and the nine full runs were judged only on KL to the BF16 model, on the R2b
  ranking conversations (disjoint from both benchmarks) and on WikiText-2 (`docs/m3_results.md`). No R3 checkpoint
  has been run on any benchmark task.
- **R2c smoke test** (2026-11-22): 5 development `multi_turn_base` tasks (not in the holdout), seed 1, to check the
  routing; no other repair has been run on agent tasks since the Oct 28 runs.

What the development data informed: the step-level primary outcome, the depth bins, the backend as a factor, the
confirmatory classes (through the power table), and the choice of a held-out task set.

## 9. Before filing (checklist)

1. Team and advisor approve this version, including the holdout design and decision 18.
2. ~~Write the holdout experiment configs and record their hashes in §7.~~ Done 2026-11-19 (`01e4272`).
3. ~~Re-check that the analysis files in §7 are unchanged since `1e13897`.~~ Checked 2026-11-22: unchanged.
4. ~~File on OSF (Preregistration template) and add the registration ID to this file.~~ Filed 2026-11-22,
   https://osf.io/xwq5y (`results/m3/osf_registration.json`). Stage C1 may now be submitted.
