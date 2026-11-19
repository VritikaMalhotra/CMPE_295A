# H1 damage preview: lfm2.5-1.2b (development data, preview)

**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may still be running (pair counts below). Method: paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled rates; classifier codebook v0. JSON: `results/m3/damage_preview_lfm2.5-1.2b.json`.

## Q4_0-pure vs BF16: 1993 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 101, 'tau2_verified_retail': 408, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 29.81 | 34.43 | 1.16 | [1.11, 1.20] | 4.62 [3.45, 5.81] | 6998 / 7808 |
| `exec_error` | 12.25 | 10.58 | 0.86 | [0.81, 0.92] | -1.66 [-2.39, -0.93] | 2875 / 2400 |
| `tool_error_payload` | 7.93 | 8.48 | 1.07 | [1.01, 1.14] | 0.56 [0.05, 1.07] | 1861 / 1924 |
| `unknown_tool` | 3.59 | 5.49 | 1.53 | [1.35, 1.73] | 1.90 [1.25, 2.57] | 843 / 1246 |
| `invalid_args` | 2.90 | 5.34 | 1.84 | [1.64, 2.09] | 2.44 [1.97, 2.92] | 681 / 1210 |
| `missing_call` | 1.71 | 2.13 | 1.25 | [1.11, 1.42] | 0.43 [0.20, 0.67] | 401 / 484 |
| `missing_call_direct_answer` | 0.84 | 1.46 | 1.73 | [1.46, 2.07] | 0.62 [0.41, 0.83] | 197 / 330 |
| `malformed` | 0.60 | 0.94 | 1.58 | [1.24, 2.02] | 0.34 [0.15, 0.54] | 140 / 213 |
| `forced_quit` | 0.00 | 0.00 | – | – | 0.00 [0.00, 0.01] | 0 / 1 |

## Q4_K_M vs BF16: 1993 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 103, 'tau2_verified_retail': 406, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 29.79 | 25.47 | 0.86 | [0.81, 0.90] | -4.32 [-5.77, -2.86] | 6996 / 5544 |
| `exec_error` | 12.23 | 8.10 | 0.66 | [0.61, 0.72] | -4.13 [-4.94, -3.36] | 2873 / 1763 |
| `tool_error_payload` | 7.92 | 7.89 | 1.00 | [0.94, 1.06] | -0.03 [-0.53, 0.46] | 1861 / 1718 |
| `invalid_args` | 2.90 | 2.60 | 0.90 | [0.78, 1.03] | -0.29 [-0.66, 0.09] | 681 / 567 |
| `unknown_tool` | 3.59 | 2.32 | 0.65 | [0.55, 0.75] | -1.27 [-1.76, -0.80] | 843 / 505 |
| `missing_call` | 1.71 | 2.25 | 1.32 | [1.15, 1.50] | 0.54 [0.28, 0.81] | 401 / 489 |
| `missing_call_direct_answer` | 0.84 | 1.20 | 1.44 | [1.20, 1.73] | 0.36 [0.18, 0.56] | 197 / 262 |
| `malformed` | 0.60 | 1.08 | 1.82 | [1.37, 2.40] | 0.49 [0.24, 0.76] | 140 / 236 |
| `truncated` | 0.00 | 0.02 | – | – | 0.02 [0.00, 0.06] | 0 / 4 |

## IQ4_XS vs BF16: 1989 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 100, 'tau2_verified_retail': 405, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 29.83 | 30.45 | 1.02 | [0.99, 1.05] | 0.63 [-0.24, 1.51] | 6989 / 7084 |
| `exec_error` | 12.24 | 12.28 | 1.00 | [0.95, 1.06] | 0.04 [-0.59, 0.68] | 2869 / 2857 |
| `tool_error_payload` | 7.94 | 7.20 | 0.91 | [0.86, 0.96] | -0.74 [-1.20, -0.29] | 1861 / 1675 |
| `invalid_args` | 2.90 | 3.79 | 1.31 | [1.17, 1.47] | 0.89 [0.51, 1.28] | 679 / 881 |
| `unknown_tool` | 3.59 | 3.72 | 1.03 | [0.93, 1.15] | 0.12 [-0.28, 0.53] | 842 / 865 |
| `missing_call` | 1.71 | 1.59 | 0.93 | [0.82, 1.05] | -0.12 [-0.33, 0.08] | 401 / 370 |
| `malformed` | 0.60 | 0.95 | 1.59 | [1.24, 2.05] | 0.35 [0.16, 0.55] | 140 / 221 |
| `missing_call_direct_answer` | 0.84 | 0.89 | 1.06 | [0.90, 1.25] | 0.05 [-0.09, 0.19] | 197 / 207 |
| `truncated` | 0.00 | 0.03 | – | – | 0.03 [0.00, 0.09] | 0 / 8 |
