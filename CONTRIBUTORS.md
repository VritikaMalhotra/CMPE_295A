# Contributors and provenance

This repository contains the Workbook 1 pilot/testbed code for the CMPE 295A
project. Attribution below was confirmed by Ekant Kapgate and Vritika Malhotra
in the project workspace on 2026-10-09.

## Contributors

- Ekant Kapgate — `ekant1999` — runner/backends, grammar-constrained repair,
  cost/energy profiling, and related inference spikes.
- Vritika Malhotra — `VritikaMalhotra` — BFCL benchmark integration, benchmark
  evaluation, failure/accuracy analysis, and figure/slide-number analysis.
- Yashashav Devalapalli Kamalraj — `yashashav-dk` — repository scaffold, QAD
  checkpoint/resume spike, model/experiment setup, and v2 harness integration.

## Code-to-contribution mapping

| Historical commit | Work represented | Attribution |
|---|---|---|
| `d17871d` | Initial testbed layout and experiment structure | Yashashav |
| `439f38a` | S1/S2 llama.cpp build, quantization, and raw tool-call runner | Ekant |
| `af30253` | S4 BFCL single-turn pilot and figure support | Vritika |
| `5346cfa` | S3/R1 grammar-constrained decoding and schema-to-GBNF support | Ekant |
| `6038606` | S7 toy QAD checkpoint/resume experiment | Yashashav |
| `f0bfdfd` | Slide-number extraction and non-live category analysis | Vritika |
| `e73cd81` | S6 NVML energy profiling and energy figure | Ekant |
| `1766e9c` | v2 BFCL harness, grammar options, and v2 figures | Vritika, with Yashashav integration support |

The historical hashes above refer to the original local repository history.
The attributed branch preserves that history's file content and records the
attribution separately; dates are retained only where supported by the local
session logs and repository metadata.

## Week of 2026-11-09 to 2026-11-13

This week's commits on this branch (184 commits ahead of `main`, including merges) by author:

### Ekant Kapgate (71 commits)

- Cluster and KL checks: GGUF KL checks against BF16, llama-perplexity and llama-imatrix builds, KL reporting, the
  cluster runbook, and the M2 cluster results.
- M2 references and infrastructure: BF16 reference configs (Qwen3 0.6B, 1.7B, 4B), the inclusion filter, codebook
  v0, the τ² simulated-user job and reference runs, and the retry and exclusion rules.
- Testing and CI: unit tests, GitHub Actions CI, a grammar property test, a golden-episode replay test, and the
  CUDA-vs-CPU parity check.
- M3 preparation and analysis: H1 damage-run configs, the mixed-effects models in R (lme4), the M3 results commits,
  and the R1 Pythonic grammar.
- Late-week runs: the A5 S10 rerun, and the B1 Qwen3-1.7B exclusion checks and notes.
- CONTRIBUTORS.md updates for this week.

### Yashashav Devalapalli Kamalraj (69 commits)

- R2b and B3/B4: the R2b ranking window and chain, the B3 Qwen3-0.6B R2b build, and the SmolLM3 tied-head diagnosis.
- A2 reference runs: FR-E2 results for LFM2.5-1.2B, LFM2.5-2.6B and SmolLM3, the parser fix, and the
  surrogate-pair work.
- Deck and CONTRIBUTORS: the advisor-deck updater and content, and the CONTRIBUTORS status.
- Analysis and docs: `which_tasks_fail`, `prompt_identity_check`, `benchmark_domains`, `run_status`, the R1 grammar
  doc, and the evidence-audit corrections.
- Cluster and other: `pack.sbatch`, the `rank_submit` chain, and the lfm2 format fixes.

### Vritika Malhotra (44 commits)

- Labels and statistics: the H2 label kit and its blank sheets, the B5 re-export, the A3 mixed-effects run, and the
  T36 feasibility note.
- Pre-registration and decisions: the A4 pre-registration hashes and follow-up rows, and decisions 15–17.
- Deck: the frozen deck, the last content edits, and the Qwen3-0.6B cell content.
- Merges and A2: merges of her branch, plus some A2 runs and run_status refreshes.

Earlier M2 work by Ekant on `m2-phase1`, already in the history and credited there: codebook v0, the inclusion
filter, the simulated user as a cluster job, the Qwen3 BF16 references, FR-E2, the power table, parity and
throughput, and the mixed-effects models in R (lme4 models A/B/C, b1e61cb) that A3 ran on real data this week. Vritika's labels, κ study and codebook v1 are hers to do; this week only prepared the blank sheets.
