# H1 damage preview: smollm3-3b (development data, preview)

**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may still be running (pair counts below). Method: paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled rates; classifier codebook v0. JSON: `results/m3/damage_preview_smollm3-3b.json`.

## Q4_0-pure vs BF16: 1934 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 91, 'tau2_verified_retail': 365, 'tau2_verified_telecom': 278}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 33.32 | 44.26 | 1.33 | [1.27, 1.39] | 10.94 [9.20, 12.76] | 4253 / 5104 |
| `missing_call_direct_answer` | 21.30 | 24.33 | 1.14 | [1.07, 1.22] | 3.03 [1.57, 4.60] | 2719 / 2806 |
| `truncated` | 1.22 | 12.58 | 10.30 | [8.37, 12.91] | 11.36 [10.24, 12.55] | 156 / 1451 |
| `missing_call` | 5.87 | 7.33 | 1.25 | [1.08, 1.45] | 1.46 [0.50, 2.46] | 749 / 845 |
| `malformed` | 0.89 | 0.02 | 0.02 | [0.00, 0.05] | -0.87 [-1.11, -0.64] | 113 / 2 |
| `exec_error` | 0.34 | 0.00 | 0.00 | [0.00, 0.00] | -0.34 [-0.52, -0.19] | 43 / 0 |
| `forced_quit` | 0.03 | 0.00 | 0.00 | – | -0.03 [-0.06, -0.01] | 4 / 0 |
| `invalid_args` | 0.46 | 0.00 | 0.00 | [0.00, 0.00] | -0.46 [-0.66, -0.29] | 59 / 0 |
| `tool_error_payload` | 2.96 | 0.00 | 0.00 | [0.00, 0.00] | -2.96 [-3.84, -2.22] | 378 / 0 |
| `unknown_tool` | 0.25 | 0.00 | 0.00 | [0.00, 0.00] | -0.25 [-0.39, -0.14] | 32 / 0 |

## Q4_K_M vs BF16: 1940 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 89, 'tau2_verified_retail': 369, 'tau2_verified_telecom': 282}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 33.18 | 33.86 | 1.02 | [0.98, 1.06] | 0.68 [-0.63, 1.96] | 4255 / 4250 |
| `missing_call_direct_answer` | 21.20 | 21.75 | 1.03 | [0.97, 1.08] | 0.54 [-0.62, 1.71] | 2719 / 2729 |
| `missing_call` | 5.84 | 5.65 | 0.97 | [0.85, 1.10] | -0.19 [-0.95, 0.58] | 749 / 709 |
| `tool_error_payload` | 2.95 | 2.33 | 0.79 | [0.62, 1.01] | -0.61 [-1.39, 0.03] | 378 / 293 |
| `truncated` | 1.23 | 2.01 | 1.63 | [1.26, 2.10] | 0.78 [0.36, 1.21] | 158 / 252 |
| `malformed` | 0.88 | 1.23 | 1.39 | [1.03, 1.90] | 0.35 [0.03, 0.67] | 113 / 154 |
| `invalid_args` | 0.46 | 0.35 | 0.76 | [0.49, 1.19] | -0.11 [-0.29, 0.07] | 59 / 44 |
| `exec_error` | 0.34 | 0.29 | 0.88 | [0.65, 1.16] | -0.04 [-0.13, 0.04] | 43 / 37 |
| `unknown_tool` | 0.25 | 0.25 | 0.99 | [0.56, 1.72] | -0.00 [-0.14, 0.12] | 32 / 31 |
| `forced_quit` | 0.03 | 0.01 | 0.26 | – | -0.02 [-0.06, 0.01] | 4 / 1 |

## IQ4_XS vs BF16: 1932 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 89, 'tau2_verified_retail': 363, 'tau2_verified_telecom': 280}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 33.39 | 34.94 | 1.05 | [1.01, 1.08] | 1.56 [0.36, 2.79] | 4251 / 4425 |
| `missing_call_direct_answer` | 21.36 | 24.05 | 1.13 | [1.07, 1.18] | 2.70 [1.55, 3.95] | 2719 / 3046 |
| `missing_call` | 5.88 | 5.51 | 0.94 | [0.81, 1.09] | -0.37 [-1.21, 0.48] | 749 / 698 |
| `truncated` | 1.22 | 2.55 | 2.10 | [1.66, 2.66] | 1.33 [0.88, 1.80] | 155 / 323 |
| `tool_error_payload` | 2.97 | 1.34 | 0.45 | [0.35, 0.58] | -1.63 [-2.38, -1.00] | 378 / 170 |
| `malformed` | 0.88 | 0.52 | 0.59 | [0.40, 0.86] | -0.36 [-0.62, -0.11] | 112 / 66 |
| `exec_error` | 0.34 | 0.50 | 1.47 | [0.96, 2.50] | 0.16 [-0.01, 0.34] | 43 / 63 |
| `invalid_args` | 0.46 | 0.24 | 0.53 | [0.31, 0.83] | -0.22 [-0.39, -0.06] | 59 / 31 |
| `unknown_tool` | 0.25 | 0.22 | 0.88 | [0.43, 1.76] | -0.03 [-0.19, 0.13] | 32 / 28 |
| `forced_quit` | 0.03 | 0.00 | 0.00 | – | -0.03 [-0.06, -0.01] | 4 / 0 |
