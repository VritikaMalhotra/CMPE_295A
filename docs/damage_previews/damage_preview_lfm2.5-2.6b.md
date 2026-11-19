# H1 damage preview: lfm2.5-2.6b (development data, preview)

**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may still be running (pair counts below). Method: paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled rates; classifier codebook v0. JSON: `results/m3/damage_preview_lfm2.5-2.6b.json`.

## Q4_0-pure vs BF16: 2000 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 101, 'tau2_verified_retail': 416, 'tau2_verified_telecom': 283}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 10.95 | 11.03 | 1.01 | [0.95, 1.07] | 0.08 [-0.62, 0.80] | 2901 / 3131 |
| `tool_error_payload` | 4.46 | 4.82 | 1.08 | [1.01, 1.15] | 0.36 [0.04, 0.68] | 1182 / 1367 |
| `unknown_tool` | 3.85 | 3.10 | 0.80 | [0.70, 0.92] | -0.75 [-1.29, -0.25] | 1020 / 879 |
| `exec_error` | 1.20 | 1.34 | 1.11 | [0.90, 1.39] | 0.14 [-0.13, 0.42] | 319 / 381 |
| `truncated` | 0.69 | 0.58 | 0.84 | [0.67, 1.06] | -0.11 [-0.25, 0.03] | 183 / 165 |
| `invalid_args` | 0.20 | 0.53 | 2.67 | [1.85, 3.85] | 0.33 [0.19, 0.48] | 52 / 149 |
| `missing_call_direct_answer` | 0.27 | 0.32 | 1.19 | [0.88, 1.66] | 0.05 [-0.04, 0.15] | 72 / 92 |
| `missing_call` | 0.23 | 0.31 | 1.33 | [0.91, 1.98] | 0.08 [-0.02, 0.18] | 61 / 87 |
| `malformed` | 0.04 | 0.03 | 0.75 | [0.11, 2.42] | -0.01 [-0.05, 0.03] | 10 / 8 |
| `forced_quit` | 0.01 | 0.01 | 1.40 | – | 0.00 [-0.01, 0.02] | 2 / 3 |

## Q4_K_M vs BF16: 2002 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 103, 'tau2_verified_retail': 415, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 10.98 | 9.96 | 0.91 | [0.85, 0.96] | -1.02 [-1.68, -0.38] | 2912 / 2593 |
| `tool_error_payload` | 4.46 | 4.56 | 1.02 | [0.96, 1.10] | 0.10 [-0.21, 0.42] | 1182 / 1187 |
| `unknown_tool` | 3.88 | 3.10 | 0.80 | [0.68, 0.93] | -0.78 [-1.34, -0.24] | 1030 / 807 |
| `exec_error` | 1.21 | 0.95 | 0.79 | [0.63, 0.97] | -0.26 [-0.49, -0.04] | 320 / 247 |
| `truncated` | 0.69 | 0.46 | 0.67 | [0.54, 0.82] | -0.23 [-0.35, -0.11] | 183 / 120 |
| `missing_call` | 0.23 | 0.37 | 1.60 | [1.16, 2.27] | 0.14 [0.04, 0.25] | 61 / 96 |
| `missing_call_direct_answer` | 0.27 | 0.33 | 1.23 | [0.93, 1.66] | 0.06 [-0.02, 0.16] | 72 / 87 |
| `invalid_args` | 0.20 | 0.15 | 0.74 | [0.46, 1.13] | -0.05 [-0.12, 0.02] | 52 / 38 |
| `forced_quit` | 0.01 | 0.02 | 3.06 | – | 0.02 [0.00, 0.03] | 2 / 6 |
| `malformed` | 0.04 | 0.02 | 0.51 | [0.10, 1.53] | -0.02 [-0.05, 0.01] | 10 / 5 |

## IQ4_XS vs BF16: 2003 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 103, 'tau2_verified_retail': 416, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 10.98 | 10.73 | 0.98 | [0.93, 1.03] | -0.25 [-0.83, 0.31] | 2912 / 2853 |
| `tool_error_payload` | 4.46 | 4.69 | 1.05 | [0.99, 1.12] | 0.23 [-0.03, 0.51] | 1182 / 1247 |
| `unknown_tool` | 3.88 | 3.47 | 0.89 | [0.78, 1.02] | -0.41 [-0.91, 0.07] | 1030 / 922 |
| `exec_error` | 1.21 | 1.03 | 0.86 | [0.72, 1.02] | -0.17 [-0.38, 0.02] | 320 / 275 |
| `truncated` | 0.69 | 0.56 | 0.81 | [0.66, 0.98] | -0.13 [-0.26, -0.01] | 183 / 148 |
| `missing_call_direct_answer` | 0.27 | 0.35 | 1.27 | [0.97, 1.72] | 0.07 [-0.01, 0.16] | 72 / 92 |
| `missing_call` | 0.23 | 0.34 | 1.49 | [1.09, 2.05] | 0.11 [0.03, 0.20] | 61 / 91 |
| `invalid_args` | 0.20 | 0.28 | 1.44 | [1.06, 1.98] | 0.09 [0.01, 0.16] | 52 / 75 |
| `malformed` | 0.04 | 0.01 | 0.20 | [0.00, 0.75] | -0.03 [-0.06, -0.00] | 10 / 2 |
| `forced_quit` | 0.01 | 0.00 | 0.50 | – | -0.00 [-0.02, 0.01] | 2 / 1 |
