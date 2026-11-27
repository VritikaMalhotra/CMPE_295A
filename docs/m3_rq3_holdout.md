# H3 registered tests, family RQ3: confirmatory holdout, registered tests implemented after registration (commit b437791)

Generated 2026-11-27T12:26:25-08:00 by `analysis/rq3_tests.py` (sources SHA-256/16: `analysis/rq3_tests.py` bf7ac209e6df33c8, `analysis/h1_tests.py` 719781e69f149ca1, `testbed/classifier.py` 6574e61cdb696eef). JSON: `results/m3/rq3_holdout.json`.

Method: §6 RQ3 as registered; pooled over the core models (cluster task x model, model stratum); trials finished under BF16, Q4_0 and the repair; per-cluster mean success and mean failure rate; D = share of success gap closed - share of failure gap closed; model-stratified bootstrap, 10,000 resamples; Holm over 3; all tasks (inclusion sets as sensitivity). Models: qwen3-0.6b, qwen3-1.7b, lfm2.5-1.2b-instruct. Sampled seeds [5, 6, 7, 8].

## RQ3: 0 of 3 supported (Holm, alpha 0.05, D > 0)

| Repair | Clusters / trials | Success gap BF16 − Q4_0 (pts) [95% CI] | Failure gap Q4_0 − BF16 (per 100 steps) [95% CI] | Share of success gap closed [95% CI] | Share of failure gap closed [95% CI] | D [95% CI] | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 900 / 3600 | 2.0 [1.1, 2.9] | 11.26 [10.14, 12.39] | 0.00 [-0.11, 0.09] | 0.30 [0.25, 0.36] | -0.30 [-0.42, -0.20] | 0.0004 | 0.0012 | **not supported** |
| Q8_0 | 900 / 3600 | 2.0 [1.1, 2.9] | 11.26 [10.12, 12.39] | 0.83 [0.63, 1.06] | 0.95 [0.91, 0.99] | -0.12 [-0.32, 0.11] | 0.2708 | 0.5415 | **not supported** |
| QAD-KL+CE | 900 / 3600 | 2.0 [1.1, 2.9] | 11.26 [10.14, 12.39] | 0.97 [0.64, 1.46] | 0.96 [0.90, 1.02] | 0.02 [-0.32, 0.50] | 0.9481 | 0.9481 | **not supported** |

Shares: 1 = the repair closes the whole BF16 → Q4_0 gap, 0 = none of it, < 0 = worse than Q4_0. Replicates with a non-positive gap are dropped from the share and D intervals (counted below).

### Rates by configuration (mean ± 1 SD over the four sampled seeds, three models pooled)

| Repair | BF16 success % | Q4_0 success % | Repair success % | BF16 failures / 100 | Q4_0 failures / 100 | Repair failures / 100 | Replicates with gap ≤ 0 |
|---|---|---|---|---|---|---|---|
| R1 | 3.8 ± 0.5 | 1.8 ± 0.2 | 1.8 ± 0.2 | 27.0 ± 1.2 | 35.5 ± 1.2 | 32.1 ± 1.5 | 0.0 % |
| Q8_0 | 3.8 ± 0.5 | 1.8 ± 0.2 | 3.4 ± 0.4 | 27.0 ± 1.2 | 35.5 ± 1.2 | 27.6 ± 1.0 | 0.0 % |
| QAD-KL+CE | 3.8 ± 0.5 | 1.8 ± 0.2 | 3.7 ± 0.3 | 27.0 ± 1.2 | 35.5 ± 1.2 | 28.1 ± 0.8 | 0.0 % |

## Per-model estimates (report only)

| Repair | Model | Clusters | Success gap (pts) | Failure gap (per 100) | Share success | Share failure | D [95% CI] |
|---|---|---|---|---|---|---|---|
| R1 | lfm2.5-1.2b-instruct | 300 | 0.2 | 4.70 | -0.50 | 0.98 | -1.48 [-3.95, 0.17] |
| R1 | qwen3-0.6b | 300 | 2.4 | 22.31 | 0.03 | 0.19 | -0.15 [-0.23, -0.02] |
| R1 | qwen3-1.7b | 300 | 3.3 | 6.77 | 0.00 | 0.22 | -0.22 [-0.39, -0.06] |
| Q8_0 | lfm2.5-1.2b-instruct | 300 | 0.2 | 4.70 | -0.50 | 0.97 | -1.47 [-4.93, 0.94] |
| Q8_0 | qwen3-0.6b | 300 | 2.4 | 22.31 | 1.03 | 0.99 | 0.04 [-0.18, 0.44] |
| Q8_0 | qwen3-1.7b | 300 | 3.3 | 6.77 | 0.75 | 0.77 | -0.02 [-0.27, 0.29] |
| QAD-KL+CE | lfm2.5-1.2b-instruct | 300 | 0.2 | 4.70 | -2.50 | 0.38 | -2.88 [-9.17, 0.68] |
| QAD-KL+CE | qwen3-0.6b | 300 | 2.4 | 22.31 | 1.45 | 1.10 | 0.34 [-0.08, 1.26] |
| QAD-KL+CE | qwen3-1.7b | 300 | 3.3 | 6.77 | 0.80 | 0.87 | -0.07 [-0.50, 0.53] |

## Sensitivity: inclusion sets (tasks the BF16 reference solves at least once; report only)

| Repair | Clusters | Success gap (pts) | Share success | Share failure | D [95% CI] | p (unadjusted) |
|---|---|---|---|---|---|---|
| R1 | 51 | 39.2 | -0.01 | 0.03 | -0.04 [-0.25, 0.19] | 0.6993 |
| Q8_0 | 51 | 39.2 | 0.73 | 0.91 | -0.18 [-0.35, -0.01] | 0.0378 |
| QAD-KL+CE | 51 | 39.2 | 0.65 | 0.93 | -0.28 [-0.48, -0.08] | 0.0088 |
