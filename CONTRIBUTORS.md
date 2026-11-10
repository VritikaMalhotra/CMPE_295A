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

## Week of 2026-11-09 to 2026-11-13 (recorded by Yashashav)

This week Yashashav ran every package from their own cluster account and git identity, including packages that
Workbook 2 assigns to Ekant and Vritika. Each commit body names the package and its Workbook 2 owner, and the work
sits on branches named for that owner. `main`, `m2` and `m2-phase1` were not touched. The Workbook 2 owners keep
their packages from the following week. This section records who did what; it does not change ownership.

| Package (Workbook 2) | WB2 owner | Run by this week | Branch |
|---|---|---|---|
| A1 prompt handlers for SmolLM3-3B and LFM2.5 | none | Yashashav | `wk11-ekant` |
| A2 BF16 references for those three models | none | Yashashav (runs submitted) | `wk11-ekant` |
| A3 statistics (C10): planted-effect check, real-data mixed-effects run | Vritika | Yashashav | `wk11-vritika` |
| A4 pre-registration fixes (not filed) | Vritika (DL-3) | Yashashav | `wk11-vritika` |
| A5 S10 rerun on the integration branch | Ekant | Yashashav | `wk11-ekant` |
| B1 H1 4-bit damage runs | Ekant | Yashashav | `wk11-ekant` |
| B2 R1 grammars (JSON and Pythonic) | Ekant | Yashashav | `wk11-ekant` |
| B3 J1 R2b ranking and build | Yashashav | Yashashav | `m3-yashashav-prep` |
| B4 SmolLM3 anomaly diagnosis | Yashashav | Yashashav | `m3-yashashav-prep` |
| B5 H2 label kit (blank sheets, guide, kappa script; no labels) | Vritika | Yashashav | `wk11-vritika` |
| C1–C4 docs answering the 11-09 feedback (agents and data, architecture, which tasks fail, worked examples) | none | Yashashav | `wk11-ekant` |
| C5 advisor deck for 11-13 | none | Yashashav | `m3-yashashav-prep` |

Earlier M2 work by Ekant on `m2-phase1`, already in the history and credited there: codebook v0, the inclusion
filter, the simulated user as a cluster job, the Qwen3 BF16 references, FR-E2, the power table, parity and
throughput. Vritika's labels, κ study and codebook v1 are hers to do; this week only prepared the blank sheets.
