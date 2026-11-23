# OSF filing sheet: answers for the "OSF Preregistration" template

Paste-ready answers for each field of OSF's Preregistration template, taken from `docs/preregistration_osf.md`
(the full document, attached to the registration as a PDF). File from the project https://osf.io/xwq5y/ only
after the advisor approves (pre-registration §9). Repository: https://github.com/VritikaMalhotra/CMPE_295A (public),
branch `m3-damage-runs`.

**Before you start:** upload the files in `osf_upload/` (next to the repository folder) to the project's OSF Storage,
so the registration freezes them. On the registration form choose the template **OSF Preregistration**, and decide
on an embargo (OSF allows up to four years; with none, it is public at once).

---

## Study information

**Title**
Where 4-bit quantization damages small tool-calling agents, and what each repair costs

**Authors**
Ekant Kapgate, Vritika Malhotra, Yashashav Devalapalli Kamalraj (San José State University); advisor Dr. Bertin
Cordova-Diba. (Add each as an OSF contributor.)

**Description**
Small open language models (0.6–4B parameters) run as tool-calling agents on devices only after 4-bit weight
quantization. We measure, step by step, where 4-bit formats raise agent failures relative to a 16-bit reference on
llama.cpp's real kernels (CUDA and CPU), and we compare three repairs, grammar-constrained decoding (R1), added bits
(R2: uniform, layer-selective, step-selective) and quantization-aware distillation (R3), each against its control,
on failures, scores and deployment cost. Research questions: RQ1 where 4-bit damage concentrates by step type and
conversation depth; RQ2 how the repairs compare, each with its control; RQ3 whether repairs fix the underlying
failures or only the score; RQ4 what each repair costs to deploy. Confirmatory tests use a held-out set of 300 BFCL
multi-turn tasks and new seeds; earlier development runs are disclosed and reported as exploratory.

**Hypotheses**
H1 (RQ1). 4-bit formats raise per-step failures above the 16-bit reference (amplification ratio > 1), and the excess
concentrates in tool-call steps and at greater conversation depth. No directional prediction for CUDA vs CPU.
H2 (RQ2). Each repair reduces the excess failures relative to its control. R1 removes format failures (malformed
call, unknown tool, schema-invalid arguments) but not the others; QAD beats the train-longer control. No directional
prediction for step-selective routing vs Q8_0 uniform, or for layer-selective vs uniform bits.
H3 (RQ3). Repairs close more of the success gap than of the failure gap.
H4 (RQ4). R1 adds no weight memory and ≤ 5% to median per-step decode latency on CUDA; any overhead falls on time to
first token, and is larger on the CPU backend. Q8_0 ≈ 1.9 × Q4_0 weight memory; R2c ≈ 2.9 × Q4_0 plus re-prefill
tokens; R3 adds no inference cost.

## Design plan

**Study type**
Experiment.

**Blinding**
No blinding is involved in this study. (Model runs; the analysis code is frozen at a registered commit and any
analysis run with different code is labelled exploratory.)

**Is there any additional blinding in this study?**
No.

**Study design**
Within-task paired design. Every configuration runs the same 300 tasks with the same seeds; configurations differ in
exactly one factor. Factors: model (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, LFM2.5-1.2B-Instruct, LFM2.5-2.6B);
format (BF16 reference; Q4_0, Q4_K_M, IQ4_XS); backend (llama.cpp d812350, CUDA A100 and CPU, same GGUF files); repair
from Q4_0 on the three core models (R1 GBNF grammar; R2a Q5_0, Q6_K, Q8_0; R2b layer-selective at 5.5 bits per weight;
R2c Q4_0 + Q8_0 with tool-call steps routed to Q8_0; R3 QAD-KL, QAD-KL+CE and a train-longer control). Each task and
configuration has 4 sampled trials (temperature 0.7, top_p 0.8, top_k 20, seeds 5–8) and 1 greedy pass (seed 0).
Stages: C1 references and 4-bit damage (36,000 episodes), C2 CPU subset (800), C3 repairs (40,500), C4 cost blocks.
R3 training settings were fixed by a pilot on held-out KL only (learning rate 1e-5, CE weight 0.1, one pass of about
51M tokens). Full details: attached PDF, §2–§3.

**Randomization**
No randomization of units: every configuration runs every task. Sampling is stochastic with fixed, recorded seeds;
the task order of the QAD training data is fixed by seed 295. Cost runs (C4) interleave configurations in random order.

## Sampling plan

**Existing data**
Registration prior to creation of data. (The confirmatory tasks and seeds have not been run for any model or
condition; see the explanation below.)

**Explanation of existing data**
Development data exist and are excluded from all confirmatory tests: the October pilot, the M1 skeleton (20 BFCL base
tasks), Oct 28 runs on all 200 BFCL base tasks for the three Qwen3 models, and a full H1 grid on 501 development tasks
(300 BFCL + 201 τ²-bench-verified) with seeds 1–4, analysed as "damage map v1". The 100 BFCL base holdout tasks were
therefore seen in aggregate for the Qwen3 models (Oct 28); a registered sensitivity analysis repeats H1 on the 200
miss_func and miss_param holdout tasks that no run has used. Also existing: KL checks, backend-parity runs, R2b
rankings, the QAD training data, R3 training judged only on KL (no benchmark runs), and an R2c smoke test on 5
development tasks. Attached PDF §8.

**Data collection procedures**
Episodes are generated by the open testbed (commit in the attached PDF §7) on the SJSU CoE HPC cluster: llama-server
(llama.cpp d812350) with one slot and prompt caching off, the model's own chat template sent to the raw completion
endpoint, context 40,960 tokens, at most 4,096 new tokens per step, bfcl's 20-step turn limit. Tasks: BFCL v4
multi-turn base, miss_func and miss_param, the 100 tasks per category outside the development sample (list and
SHA-256 in results/m3/holdout_tasks_bfcl.json), scored by bfcl-eval's unmodified multi-turn checker. Every step is
logged and labelled by the failure classifier (codebook v0). Experiment configs and their hashes: PDF §7.

**Sample size**
300 tasks per configuration × (4 sampled trials + 1 greedy pass) = 1,500 episodes per configuration; 24 C1
configurations (36,000 episodes), 8 C2 configurations on 100 tasks greedy only (800), 27 C3 configurations (40,500).

**Sample size rationale**
Power computed from the BF16 development references restricted to BFCL (results/m3/power_table_bfcl_holdout.json):
for 80% power to detect a 1.5× failure-rate ratio at α = 0.05/10, about 330 failure events times the clustering design
effect. With 1,200 sampled episodes per configuration, "any failure" is detectable in all six models at the
conservative task-level design effect (≤ 489 episodes needed); tool_error_payload in five of six; other classes only
in some models, so they are exploratory. The task count is bounded by the BFCL holdout (all 300 unused tasks).

**Stopping rule**
Every run specification is run to completion (finished or excluded under the exclusion rules); no interim look at
4-bit or repair outcomes on the holdout. Monitoring is limited to run status counts.

## Variables

**Manipulated variables**
Model (6); weight format (BF16, Q4_0, Q4_K_M, IQ4_XS); backend (CUDA, CPU; CPU only for Qwen3-0.6B and 1.7B, greedy,
BFCL base holdout); repair (none, R1, Q5_0, Q6_K, Q8_0, R2b, R2c, QAD-KL, QAD-KL+CE, train-longer control); decoding
(4 sampled seeds, 1 greedy pass).

**Measured variables**
Primary: failures per agent step by failure class (truncated, malformed, forced_quit, unknown_tool, invalid_args,
exec_error, tool_error_payload, missing_call, missing_call_direct_answer; "any failure" is their union), assigned by
the classifier (codebook v0). Secondary: episode success (bfcl-eval verdict). Per step: step type (tool-call vs text)
and depth (step index, bins 0–1, 2–3, 4–6, 7+). Costs (RQ4): weight memory, time to first token, decode tokens/s,
extra tokens (R2c probe and re-prefill, R1 grammar), GPU energy (NVML).

**Indices**
Amplification ratio = pooled failure rate per step under the condition / under its comparison, and the difference in
failures per 100 steps, both on the (task, seed) pairs both configurations finished. Gap closed (H3) = share of the
BF16→Q4_0 gap removed by a repair, computed for success and for any failure.

## Analysis plan

**Statistical models**
Effect sizes with 95% CIs from a paired cluster bootstrap over tasks (10,000 resamples; all trials of a task resampled
together). H1: per model × format, a paired permutation test (10,000 permutations) on per-task any-failure rates, BF16
vs 4-bit (family of 18); concentration by a Poisson GEE of failures on precision × step type and precision × depth
bin, clustered by task (family of 6); tool_error_payload as a secondary family (15); CPU subset by paired
permutation on the greedy pass (6). H2: 10 contrasts pooled over the three core models (task × model as cluster):
R1 vs Q4_0 on format failures and on any failure; Q5_0 and Q8_0 vs Q4_0; R2b vs Q5_0; R2c vs Q4_0; R2c vs Q8_0 as
equivalence on success (TOST ± 3 points); QAD-KL and QAD-KL+CE vs Q4_0; QAD-KL+CE vs the train-longer control. Paired
permutation tests on per-task rates, McNemar's exact test on the greedy pass as secondary. H3: paired bootstrap of
(success gap closed − failure gap closed) for R1, Q8_0 and QAD-KL+CE (family of 3). H4: ≥ 10 interleaved repetitions,
medians with bootstrap CIs, TOST ± 5% for "no overhead" claims. Registered secondary: mixed-effects models in R (lme4)
for every primary contrast; a conclusion is robust when they agree with the bootstrap. Full detail: PDF §6.

**Transformations**
Sampled trials are averaged within each task before paired tests; failure rates are per agent step; depth is binned
(0–1, 2–3, 4–6, 7+); format failures = malformed + unknown_tool + invalid_args.

**Inference criteria**
Two-sided α = 0.05 after Holm correction within each family (H1 18, H1-loc 6, H1-tep 15, H1-cpu 6, RQ2 10, RQ3 3,
RQ4 one per "no overhead" claim). A directional hypothesis is supported only if the corrected test rejects and the
estimate is in the predicted direction. Equivalence claims need TOST to reject (± 3 points for R2c vs Q8_0 success,
± 5% for cost). No correction across families; every other test is exploratory and labelled so.

**Data exclusion**
Step-level tests use every task. Tasks the BF16 reference fails in all 4 sampled trials are excluded per model from
the secondary episode-success analysis only (counts and an all-task sensitivity analysis reported). Infrastructure
failures are retried with the same seed up to 4 attempts; an episode failing all 4 is excluded and listed. Episodes
hitting the step limit or token budget are outcomes, not exclusions. Walltime interruptions are rerun.

**Missing data**
A paired contrast uses only the (task, seed) pairs both configurations finished; the number of pairs is reported for
every contrast. Episodes excluded after 4 infrastructure failures are listed per configuration.

**Exploratory analysis**
The development data (H1 grid on 501 tasks including τ²-bench-verified, damage map v1); τ² error-budget re-scoring;
failure classes other than the confirmatory ones; Q6_K; the R2c entropy-routing policy; R1 at BF16 and on the CPU
backend; R1 and R2 on Q4_K_M; vendor QAD checkpoints; the R3 control at 16 bits. All labelled exploratory.

## Other

**Other**
Codebook v1 (after an inter-rater agreement study) will be filed as an amendment before any confirmatory analysis uses
it; v0 stays the registered primary. Frozen with this registration: analysis code commit 1e13897, holdout task list,
model manifest with SHA-256 of every file, experiment configs and hashes for stages C1–C3 (commits 01e4272, cf928c8),
the R2c implementation (406d7f8) and the R3 training record. Decision log: docs/m2_decisions.md in the repository.
