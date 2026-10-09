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
