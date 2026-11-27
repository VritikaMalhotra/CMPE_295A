# H3 registered tests, family RQ3: development data, stand-in configurations (mechanics check only)

Generated 2026-11-27T12:24:45-08:00 by `analysis/rq3_tests.py` (sources SHA-256/16: `analysis/rq3_tests.py` bf7ac209e6df33c8, `analysis/h1_tests.py` 719781e69f149ca1, `testbed/classifier.py` 6574e61cdb696eef). JSON: `results/m3/rq3_tests_dev_check.json`.

Method: §6 RQ3 as registered; pooled over the core models (cluster task x model, model stratum); trials finished under BF16, Q4_0 and the repair; per-cluster mean success and mean failure rate; D = share of success gap closed - share of failure gap closed; model-stratified bootstrap, 10,000 resamples; Holm over 3; all tasks (inclusion sets as sensitivity). Models: qwen3-0.6b, qwen3-1.7b, lfm2.5-1.2b-instruct. Sampled seeds [1, 2, 3, 4].

## RQ3: 0 of 3 supported (Holm, alpha 0.05, D > 0)

| Repair | Clusters / trials | Success gap BF16 − Q4_0 (pts) [95% CI] | Failure gap Q4_0 − BF16 (per 100 steps) [95% CI] | Share of success gap closed [95% CI] | Share of failure gap closed [95% CI] | D [95% CI] | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 1503 / 5886 | 1.7 [1.1, 2.3] | 6.01 [5.01, 6.98] | 0.26 [-0.05, 0.52] | 0.97 [0.85, 1.11] | -0.71 [-1.04, -0.43] | 0.0002 | 0.0006 | **not supported** |
| Q8_0 | 1503 / 5864 | 1.7 [1.0, 2.3] | 6.00 [5.02, 6.99] | 0.73 [0.48, 1.00] | 0.58 [0.45, 0.70] | 0.15 [-0.11, 0.44] | 0.2816 | 0.5631 | **not supported** |
| QAD-KL+CE | 0 | – | – | – | – | – | – | – | missing |

Shares: 1 = the repair closes the whole BF16 → Q4_0 gap, 0 = none of it, < 0 = worse than Q4_0. Replicates with a non-positive gap are dropped from the share and D intervals (counted below).

### Rates by configuration (mean ± 1 SD over the four sampled seeds, three models pooled)

| Repair | BF16 success % | Q4_0 success % | Repair success % | BF16 failures / 100 | Q4_0 failures / 100 | Repair failures / 100 | Replicates with gap ≤ 0 |
|---|---|---|---|---|---|---|---|
| R1 | 3.4 ± 0.5 | 1.8 ± 0.6 | 2.2 ± 0.4 | 28.9 ± 1.5 | 32.8 ± 1.7 | 27.7 ± 0.9 | 0.0 % |
| Q8_0 | 3.4 ± 0.5 | 1.8 ± 0.6 | 3.0 ± 0.3 | 28.9 ± 1.5 | 32.8 ± 1.7 | 31.7 ± 1.7 | 0.0 % |

## Per-model estimates (report only)

| Repair | Model | Clusters | Success gap (pts) | Failure gap (per 100) | Share success | Share failure | D [95% CI] |
|---|---|---|---|---|---|---|---|
| R1 | lfm2.5-1.2b-instruct | 501 | 1.0 | 5.16 | -1.18 | 1.36 | -2.54 [-8.30, -1.65] |
| R1 | qwen3-0.6b | 501 | 1.2 | 7.06 | 0.92 | 1.25 | -0.33 [-0.91, 0.43] |
| R1 | qwen3-1.7b | 501 | 2.7 | 5.79 | 0.50 | 0.28 | 0.22 [-0.15, 0.64] |
| Q8_0 | lfm2.5-1.2b-instruct | 501 | 1.0 | 5.16 | 0.61 | 0.81 | -0.19 [-1.05, 0.94] |
| Q8_0 | qwen3-0.6b | 501 | 1.2 | 7.08 | 0.64 | 0.10 | 0.54 [0.04, 1.13] |
| Q8_0 | qwen3-1.7b | 501 | 2.7 | 5.77 | 0.81 | 0.98 | -0.17 [-0.58, 0.32] |

## Sensitivity: inclusion sets (tasks the BF16 reference solves at least once; report only)

| Repair | Clusters | Success gap (pts) | Share success | Share failure | D [95% CI] | p (unadjusted) |
|---|---|---|---|---|---|---|
| R1 | 84 | 33.8 | 0.17 | 0.36 | -0.19 [-0.43, 0.10] | 0.1610 |
| Q8_0 | 84 | 33.7 | 0.61 | 0.99 | -0.38 [-0.63, -0.17] | 0.0002 |
