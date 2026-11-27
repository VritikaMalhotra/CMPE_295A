# H2 registered tests, family RQ2: development data, stand-in configurations (mechanics check only)

Generated 2026-11-27T11:05:02-08:00 by `analysis/rq2_tests.py` (sources SHA-256/16: `analysis/rq2_tests.py` cf9fa966752586ce, `analysis/h1_tests.py` 719781e69f149ca1, `analysis/damage_table.py` 27daed21c262c1ca, `testbed/stats.py` 934a086ebb5c4422, `testbed/classifier.py` 6574e61cdb696eef). JSON: `results/m3/rq2_tests_dev_check.json`.

Method: §6 RQ2 as registered; cluster = task x model, model stratum; per-cluster rate = mean over paired sampled trials of failures/steps; paired permutation (10,000 sign flips); Holm within the family; pooled ratio with a model-stratified cluster bootstrap CI; contrast 7 TOST +/- 3 points on success (permutation, inclusion set; all-task sensitivity); secondary McNemar on greedy success (frozen stats.mcnemar_exact), Holm within its own 10. Models: qwen3-0.6b, qwen3-1.7b, lfm2.5-1.2b-instruct. Sampled seeds [1, 2, 3, 4]; greedy seed 0.

## RQ2: 3 of 7 directional contrasts supported

| # | Contrast | Outcome | Clusters / pairs | Treatment per 100 steps | Comparison per 100 steps | Ratio [95% CI] | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | R1 vs Q4_0 | format | 0 | – | – | – | – | – | missing |
| 2 | R1 vs Q4_0 | any | 0 | – | – | – | – | – | missing |
| 3 | Q5_0 vs Q4_0 | any | 1503 / 5876 | 31.81 ± 1.67 | 32.78 ± 1.69 | 0.97 [0.94, 1.00] | <0.0001 | 0.0010 | **supported** |
| 4 | Q8_0 vs Q4_0 | any | 1503 / 5896 | 28.85 ± 1.53 | 32.79 ± 1.67 | 0.88 [0.85, 0.91] | <0.0001 | 0.0010 | **supported** |
| 5 | R2b vs Q5_0 | any | 0 | – | – | – | – | – | missing |
| 6 | R2c vs Q4_0 | any | 1503 / 5898 | 27.51 ± 0.90 | 32.77 ± 1.70 | 0.84 [0.81, 0.87] | <0.0001 | 0.0010 | **supported** |
| 7 | R2c vs Q8_0 | success | 84 / 334 | 33.2 ± 4.5 % success | 60.5 ± 9.3 % success | Δ success -27.2 pts, 90% CI [-32.5, -21.7], margin ± 3 | 1.0000 | 1.0000 | **equivalence not shown** |
| 8 | QAD-KL vs Q4_0 | any | 0 | – | – | – | – | – | missing |
| 9 | QAD-KL+CE vs Q4_0 | any | 0 | – | – | – | – | – | missing |
| 10 | QAD-KL+CE vs train-longer | any | 0 | – | – | – | – | – | missing |

Rates: mean ± 1 SD over the four sampled seeds of the pooled (all three models) rate. Contrast 7 uses the registered inclusion sets (tasks the BF16 reference solves at least once); its p is the TOST p.

Contrast 7, all-task sensitivity (1503 clusters): Δ success -1.2 pts, 90% CI [-1.6, -0.7], TOST p <0.0001.

## Per-model estimates

| # | Contrast | Model | Clusters | Treatment / comparison per 100 steps | Ratio [95% CI] | p (unadjusted) |
|---|---|---|---|---|---|---|
| 3 | Q5_0 vs Q4_0 | lfm2.5-1.2b-instruct | 501 | 30.46 / 34.45 | 0.88 [0.85, 0.92] | <0.0001 |
| 3 | Q5_0 vs Q4_0 | qwen3-0.6b | 501 | 38.20 / 31.18 | 1.23 [1.13, 1.33] | 0.6212 |
| 3 | Q5_0 vs Q4_0 | qwen3-1.7b | 501 | 27.67 / 32.21 | 0.86 [0.81, 0.91] | <0.0001 |
| 4 | Q8_0 vs Q4_0 | lfm2.5-1.2b-instruct | 501 | 29.81 / 34.43 | 0.87 [0.83, 0.90] | <0.0001 |
| 4 | Q8_0 vs Q4_0 | qwen3-0.6b | 501 | 30.47 / 31.13 | 0.98 [0.91, 1.06] | <0.0001 |
| 4 | Q8_0 vs Q4_0 | qwen3-1.7b | 501 | 26.52 / 32.29 | 0.82 [0.78, 0.87] | <0.0001 |
| 6 | R2c vs Q4_0 | lfm2.5-1.2b-instruct | 501 | 25.46 / 34.43 | 0.74 [0.69, 0.79] | <0.0001 |
| 6 | R2c vs Q4_0 | qwen3-0.6b | 501 | 28.94 / 31.11 | 0.93 [0.87, 0.99] | <0.0001 |
| 6 | R2c vs Q4_0 | qwen3-1.7b | 501 | 28.54 / 32.25 | 0.88 [0.83, 0.95] | 0.0519 |

## Secondary: McNemar's exact test on greedy-pass success (report only)

| # | Contrast | Pairs | Success treatment [95% Wilson] | Success comparison [95% Wilson] | Discordant (treatment only / comparison only) | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| 1 | R1 vs Q4_0 | 0 | – | – | – | – | – |
| 2 | R1 vs Q4_0 | 0 | – | – | – | – | – |
| 3 | Q5_0 vs Q4_0 | 1468 | 3.3 % [2.5, 4.3] | 2.0 % [1.4, 2.9] | 27 / 9 | 0.0039 | 0.0354 |
| 4 | Q8_0 vs Q4_0 | 1471 | 4.0 % [3.1, 5.1] | 2.0 % [1.4, 2.9] | 35 / 6 | <0.0001 | <0.0001 |
| 5 | R2b vs Q5_0 | 0 | – | – | – | – | – |
| 6 | R2c vs Q4_0 | 1471 | 3.1 % [2.4, 4.1] | 2.0 % [1.4, 2.9] | 27 / 11 | 0.0139 | 0.1108 |
| 7 | R2c vs Q8_0 | 1498 | 3.1 % [2.3, 4.1] | 3.9 % [3.1, 5.0] | 13 / 26 | 0.0533 | 0.3728 |
| 8 | QAD-KL vs Q4_0 | 0 | – | – | – | – | – |
| 9 | QAD-KL+CE vs Q4_0 | 0 | – | – | – | – | – |
| 10 | QAD-KL+CE vs train-longer | 0 | – | – | – | – | – |
