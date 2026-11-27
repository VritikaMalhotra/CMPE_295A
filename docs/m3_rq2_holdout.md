# H2 registered tests, family RQ2: confirmatory holdout C3, registered tests implemented after registration (commit ea72e4c)

Generated 2026-11-27T11:06:44-08:00 by `analysis/rq2_tests.py` (sources SHA-256/16: `analysis/rq2_tests.py` cf9fa966752586ce, `analysis/h1_tests.py` 719781e69f149ca1, `analysis/damage_table.py` 27daed21c262c1ca, `testbed/stats.py` 934a086ebb5c4422, `testbed/classifier.py` 6574e61cdb696eef). JSON: `results/m3/rq2_holdout.json`.

Method: §6 RQ2 as registered; cluster = task x model, model stratum; per-cluster rate = mean over paired sampled trials of failures/steps; paired permutation (10,000 sign flips); Holm within the family; pooled ratio with a model-stratified cluster bootstrap CI; contrast 7 TOST +/- 3 points on success (permutation, inclusion set; all-task sensitivity); secondary McNemar on greedy success (frozen stats.mcnemar_exact), Holm within its own 10. Models: qwen3-0.6b, qwen3-1.7b, lfm2.5-1.2b-instruct. Sampled seeds [5, 6, 7, 8]; greedy seed 0.

## RQ2: 7 of 7 directional contrasts supported

| # | Contrast | Outcome | Clusters / pairs | Treatment per 100 steps | Comparison per 100 steps | Ratio [95% CI] | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | R1 vs Q4_0 | format | 900 / 3600 | 0.12 ± 0.04 | 6.93 ± 0.68 | 0.02 [0.01, 0.03] | <0.0001 | 0.0010 | **supported** |
| 2 | R1 vs Q4_0 | any | 900 / 3600 | 32.13 ± 1.54 | 35.54 ± 1.21 | 0.90 [0.88, 0.93] | <0.0001 | 0.0010 | **significant, lower** |
| 3 | Q5_0 vs Q4_0 | any | 900 / 3600 | 30.99 ± 1.29 | 35.54 ± 1.21 | 0.87 [0.84, 0.91] | <0.0001 | 0.0010 | **supported** |
| 4 | Q8_0 vs Q4_0 | any | 900 / 3600 | 27.57 ± 0.98 | 35.54 ± 1.21 | 0.78 [0.75, 0.80] | <0.0001 | 0.0010 | **supported** |
| 5 | R2b vs Q5_0 | any | 900 / 3600 | 29.05 ± 0.76 | 30.99 ± 1.29 | 0.94 [0.91, 0.97] | 0.0002 | 0.0010 | **significant, lower** |
| 6 | R2c vs Q4_0 | any | 900 / 3600 | 33.72 ± 2.10 | 35.54 ± 1.21 | 0.95 [0.92, 0.97] | 0.0005 | 0.0010 | **supported** |
| 7 | R2c vs Q8_0 | success | 51 / 204 | 32.4 ± 5.2 % success | 55.9 ± 6.5 % success | Δ success -23.5 pts, 90% CI [-31.4, -16.2], margin ± 3 | 1.0000 | 1.0000 | **equivalence not shown** |
| 8 | QAD-KL vs Q4_0 | any | 900 / 3600 | 28.11 ± 0.73 | 35.54 ± 1.21 | 0.79 [0.76, 0.82] | <0.0001 | 0.0010 | **supported** |
| 9 | QAD-KL+CE vs Q4_0 | any | 900 / 3600 | 28.11 ± 0.80 | 35.54 ± 1.21 | 0.79 [0.76, 0.82] | <0.0001 | 0.0010 | **supported** |
| 10 | QAD-KL+CE vs train-longer | any | 900 / 3600 | 28.11 ± 0.80 | 33.86 ± 1.65 | 0.83 [0.80, 0.86] | <0.0001 | 0.0010 | **supported** |

Rates: mean ± 1 SD over the four sampled seeds of the pooled (all three models) rate. Contrast 7 uses the registered inclusion sets (tasks the BF16 reference solves at least once); its p is the TOST p.

Contrast 7, all-task sensitivity (900 clusters): Δ success -1.4 pts, 90% CI [-2.0, -0.9], TOST p <0.0001.

## Per-model estimates

| # | Contrast | Model | Clusters | Treatment / comparison per 100 steps | Ratio [95% CI] | p (unadjusted) |
|---|---|---|---|---|---|---|
| 1 | R1 vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 0.15 / 9.67 | 0.02 [0.01, 0.03] | <0.0001 |
| 1 | R1 vs Q4_0 | qwen3-0.6b | 300 | 0.12 / 6.84 | 0.02 [0.01, 0.03] | <0.0001 |
| 1 | R1 vs Q4_0 | qwen3-1.7b | 300 | 0.08 / 4.02 | 0.02 [0.00, 0.03] | <0.0001 |
| 2 | R1 vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 28.47 / 33.42 | 0.85 [0.82, 0.88] | <0.0001 |
| 2 | R1 vs Q4_0 | qwen3-0.6b | 300 | 40.93 / 45.43 | 0.90 [0.87, 0.93] | <0.0001 |
| 2 | R1 vs Q4_0 | qwen3-1.7b | 300 | 28.94 / 29.81 | 0.97 [0.91, 1.04] | 0.0027 |
| 3 | Q5_0 vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 30.30 / 33.42 | 0.91 [0.87, 0.94] | <0.0001 |
| 3 | Q5_0 vs Q4_0 | qwen3-0.6b | 300 | 34.33 / 45.43 | 0.76 [0.70, 0.81] | <0.0001 |
| 3 | Q5_0 vs Q4_0 | qwen3-1.7b | 300 | 28.84 / 29.81 | 0.97 [0.88, 1.06] | <0.0001 |
| 4 | Q8_0 vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 28.61 / 33.42 | 0.86 [0.82, 0.89] | <0.0001 |
| 4 | Q8_0 vs Q4_0 | qwen3-0.6b | 300 | 29.01 / 45.43 | 0.64 [0.59, 0.69] | <0.0001 |
| 4 | Q8_0 vs Q4_0 | qwen3-1.7b | 300 | 25.11 / 29.81 | 0.84 [0.79, 0.90] | <0.0001 |
| 5 | R2b vs Q5_0 | lfm2.5-1.2b-instruct | 300 | 31.33 / 30.30 | 1.03 [1.00, 1.07] | 0.0147 |
| 5 | R2b vs Q5_0 | qwen3-0.6b | 300 | 32.12 / 34.33 | 0.94 [0.89, 0.98] | 0.0018 |
| 5 | R2b vs Q5_0 | qwen3-1.7b | 300 | 24.16 / 28.84 | 0.84 [0.77, 0.92] | 0.0002 |
| 6 | R2c vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 30.25 / 33.42 | 0.91 [0.87, 0.94] | <0.0001 |
| 6 | R2c vs Q4_0 | qwen3-0.6b | 300 | 44.73 / 45.43 | 0.98 [0.95, 1.02] | 0.0236 |
| 6 | R2c vs Q4_0 | qwen3-1.7b | 300 | 28.34 / 29.81 | 0.95 [0.90, 1.01] | 0.8510 |
| 8 | QAD-KL vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 32.39 / 33.42 | 0.97 [0.93, 1.01] | <0.0001 |
| 8 | QAD-KL vs Q4_0 | qwen3-0.6b | 300 | 27.40 / 45.43 | 0.60 [0.55, 0.66] | <0.0001 |
| 8 | QAD-KL vs Q4_0 | qwen3-1.7b | 300 | 23.97 / 29.81 | 0.80 [0.75, 0.86] | <0.0001 |
| 9 | QAD-KL+CE vs Q4_0 | lfm2.5-1.2b-instruct | 300 | 32.90 / 33.42 | 0.98 [0.95, 1.02] | 0.0011 |
| 9 | QAD-KL+CE vs Q4_0 | qwen3-0.6b | 300 | 26.76 / 45.43 | 0.59 [0.54, 0.64] | <0.0001 |
| 9 | QAD-KL+CE vs Q4_0 | qwen3-1.7b | 300 | 23.75 / 29.81 | 0.80 [0.75, 0.85] | <0.0001 |
| 10 | QAD-KL+CE vs train-longer | lfm2.5-1.2b-instruct | 300 | 32.90 / 32.88 | 1.00 [0.96, 1.04] | 0.1577 |
| 10 | QAD-KL+CE vs train-longer | qwen3-0.6b | 300 | 26.76 / 41.94 | 0.64 [0.59, 0.69] | <0.0001 |
| 10 | QAD-KL+CE vs train-longer | qwen3-1.7b | 300 | 23.75 / 28.71 | 0.83 [0.76, 0.90] | 0.0011 |

## Secondary: McNemar's exact test on greedy-pass success (report only)

| # | Contrast | Pairs | Success treatment [95% Wilson] | Success comparison [95% Wilson] | Discordant (treatment only / comparison only) | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| 1 | R1 vs Q4_0 | 900 | 1.8 % [1.1, 2.9] | 1.8 % [1.1, 2.9] | 0 / 0 | 1.0000 | 1.0000 |
| 2 | R1 vs Q4_0 | 900 | 1.8 % [1.1, 2.9] | 1.8 % [1.1, 2.9] | 0 / 0 | 1.0000 | 1.0000 |
| 3 | Q5_0 vs Q4_0 | 900 | 3.2 % [2.3, 4.6] | 1.8 % [1.1, 2.9] | 17 / 4 | 0.0072 | 0.0360 |
| 4 | Q8_0 vs Q4_0 | 900 | 3.4 % [2.4, 4.8] | 1.8 % [1.1, 2.9] | 21 / 6 | 0.0059 | 0.0355 |
| 5 | R2b vs Q5_0 | 900 | 3.4 % [2.4, 4.8] | 3.2 % [2.3, 4.6] | 13 / 11 | 0.8388 | 1.0000 |
| 6 | R2c vs Q4_0 | 900 | 2.1 % [1.4, 3.3] | 1.8 % [1.1, 2.9] | 8 / 5 | 0.5811 | 1.0000 |
| 7 | R2c vs Q8_0 | 900 | 2.1 % [1.4, 3.3] | 3.4 % [2.4, 4.8] | 2 / 14 | 0.0042 | 0.0293 |
| 8 | QAD-KL vs Q4_0 | 900 | 4.0 % [2.9, 5.5] | 1.8 % [1.1, 2.9] | 23 / 3 | <0.0001 | 0.0008 |
| 9 | QAD-KL+CE vs Q4_0 | 900 | 5.0 % [3.8, 6.6] | 1.8 % [1.1, 2.9] | 32 / 3 | <0.0001 | <0.0001 |
| 10 | QAD-KL+CE vs train-longer | 900 | 5.0 % [3.8, 6.6] | 2.9 % [2.0, 4.2] | 26 / 7 | 0.0013 | 0.0105 |
