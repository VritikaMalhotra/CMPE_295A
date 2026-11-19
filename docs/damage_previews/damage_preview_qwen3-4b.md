# H1 damage preview: qwen3-4b (development data, preview)

**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may still be running (pair counts below). Method: paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled rates; classifier codebook v0. JSON: `results/m3/damage_preview_qwen3-4b.json`.

## Q4_0-pure vs BF16: 2002 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 104, 'tau2_verified_retail': 415, 'tau2_verified_telecom': 283}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 23.05 | 22.55 | 0.98 | [0.94, 1.02] | -0.50 [-1.45, 0.47] | 5453 / 4801 |
| `tool_error_payload` | 6.16 | 7.19 | 1.17 | [1.09, 1.25] | 1.03 [0.57, 1.51] | 1457 / 1530 |
| `unknown_tool` | 8.51 | 7.03 | 0.83 | [0.78, 0.88] | -1.48 [-2.04, -0.93] | 2013 / 1497 |
| `exec_error` | 6.59 | 5.72 | 0.87 | [0.78, 0.97] | -0.87 [-1.54, -0.21] | 1559 / 1218 |
| `missing_call` | 0.90 | 1.03 | 1.15 | [0.96, 1.39] | 0.13 [-0.04, 0.31] | 212 / 219 |
| `missing_call_direct_answer` | 0.57 | 0.95 | 1.66 | [1.34, 2.10] | 0.38 [0.21, 0.56] | 136 / 203 |
| `invalid_args` | 0.19 | 0.45 | 2.35 | [1.35, 4.49] | 0.26 [0.07, 0.45] | 45 / 95 |
| `truncated` | 0.01 | 0.10 | 12.22 | – | 0.09 [0.02, 0.21] | 2 / 22 |
| `malformed` | 0.08 | 0.07 | 0.82 | [0.33, 1.69] | -0.01 [-0.06, 0.03] | 19 / 14 |
| `forced_quit` | 0.04 | 0.01 | 0.33 | – | -0.03 [-0.07, 0.00] | 10 / 3 |

## Q4_K_M vs BF16: 2001 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 103, 'tau2_verified_retail': 414, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 23.05 | 22.85 | 0.99 | [0.95, 1.04] | -0.21 [-1.24, 0.83] | 5453 / 4996 |
| `unknown_tool` | 8.54 | 7.42 | 0.87 | [0.81, 0.93] | -1.12 [-1.67, -0.59] | 2021 / 1623 |
| `tool_error_payload` | 6.16 | 6.37 | 1.03 | [0.96, 1.11] | 0.21 [-0.25, 0.68] | 1457 / 1392 |
| `exec_error` | 6.56 | 5.76 | 0.88 | [0.80, 0.96] | -0.80 [-1.38, -0.22] | 1551 / 1260 |
| `missing_call` | 0.90 | 2.03 | 2.27 | [1.89, 2.80] | 1.14 [0.83, 1.48] | 212 / 445 |
| `missing_call_direct_answer` | 0.57 | 1.02 | 1.77 | [1.38, 2.31] | 0.44 [0.24, 0.66] | 136 / 222 |
| `invalid_args` | 0.19 | 0.14 | 0.72 | [0.35, 1.24] | -0.05 [-0.15, 0.03] | 45 / 30 |
| `malformed` | 0.08 | 0.07 | 0.91 | [0.15, 3.16] | -0.01 [-0.09, 0.09] | 19 / 16 |
| `forced_quit` | 0.04 | 0.02 | 0.54 | – | -0.02 [-0.07, 0.02] | 10 / 5 |
| `truncated` | 0.01 | 0.01 | 1.62 | – | 0.01 [-0.02, 0.03] | 2 / 3 |

## IQ4_XS vs BF16: 1999 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 103, 'tau2_verified_retail': 412, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 23.07 | 22.65 | 0.98 | [0.94, 1.02] | -0.42 [-1.32, 0.50] | 5451 / 5392 |
| `unknown_tool` | 8.55 | 8.18 | 0.96 | [0.90, 1.02] | -0.37 [-0.87, 0.13] | 2021 / 1948 |
| `tool_error_payload` | 6.17 | 6.86 | 1.11 | [1.04, 1.19] | 0.70 [0.26, 1.15] | 1457 / 1634 |
| `exec_error` | 6.56 | 5.99 | 0.91 | [0.84, 1.00] | -0.57 [-1.14, -0.01] | 1549 / 1425 |
| `missing_call` | 0.90 | 0.79 | 0.88 | [0.71, 1.10] | -0.10 [-0.30, 0.08] | 212 / 189 |
| `missing_call_direct_answer` | 0.58 | 0.52 | 0.90 | [0.71, 1.11] | -0.06 [-0.18, 0.06] | 136 / 123 |
| `invalid_args` | 0.19 | 0.23 | 1.19 | [0.69, 2.21] | 0.04 [-0.08, 0.14] | 45 / 54 |
| `malformed` | 0.08 | 0.05 | 0.57 | [0.20, 1.64] | -0.03 [-0.10, 0.02] | 19 / 11 |
| `forced_quit` | 0.04 | 0.03 | 0.69 | – | -0.01 [-0.06, 0.02] | 10 / 7 |
| `truncated` | 0.01 | 0.00 | 0.50 | – | -0.00 [-0.02, 0.01] | 2 / 1 |
