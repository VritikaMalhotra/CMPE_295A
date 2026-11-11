# H1 damage preview: qwen3-0.6b (development data, preview)

**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may still be running (pair counts below). Method: paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled rates; classifier codebook v0. JSON: `results/m3/damage_preview_qwen3-0.6b.json` (job 112508).

## Q4_0-pure vs BF16: 1999 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 102, 'tau2_verified_retail': 413, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 30.47 | 31.13 | 1.02 | [0.95, 1.10] | 0.66 [-1.65, 3.01] | 5534 / 4915 |
| `missing_call_direct_answer` | 2.87 | 9.78 | 3.40 | [2.91, 4.06] | 6.91 [5.84, 8.08] | 522 / 1545 |
| `tool_error_payload` | 9.41 | 5.68 | 0.60 | [0.53, 0.68] | -3.73 [-4.76, -2.80] | 1710 / 897 |
| `unknown_tool` | 6.99 | 5.41 | 0.78 | [0.62, 0.95] | -1.57 [-2.87, -0.31] | 1269 / 855 |
| `missing_call` | 1.02 | 4.33 | 4.22 | [3.37, 5.44] | 3.30 [2.63, 4.05] | 186 / 683 |
| `invalid_args` | 2.68 | 2.81 | 1.05 | [0.78, 1.42] | 0.13 [-0.68, 0.95] | 487 / 444 |
| `malformed` | 0.29 | 1.58 | 5.40 | [3.44, 9.56] | 1.29 [0.94, 1.66] | 53 / 249 |
| `exec_error` | 7.16 | 1.41 | 0.20 | [0.14, 0.26] | -5.74 [-6.65, -4.86] | 1300 / 223 |
| `forced_quit` | 0.04 | 0.09 | 2.30 | [0.81, 7.81] | 0.05 [-0.01, 0.12] | 7 / 14 |
| `truncated` | 0.00 | 0.03 | – | – | 0.03 [0.01, 0.06] | 0 / 5 |

## Q4_K_M vs BF16: 2000 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 101, 'tau2_verified_retail': 415, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 30.44 | 28.96 | 0.95 | [0.90, 1.00] | -1.48 [-2.98, 0.03] | 5532 / 4852 |
| `tool_error_payload` | 9.41 | 7.98 | 0.85 | [0.78, 0.92] | -1.43 [-2.22, -0.68] | 1710 / 1337 |
| `unknown_tool` | 6.98 | 7.57 | 1.08 | [0.98, 1.21] | 0.59 [-0.18, 1.37] | 1269 / 1269 |
| `missing_call_direct_answer` | 2.87 | 4.70 | 1.64 | [1.43, 1.89] | 1.83 [1.28, 2.44] | 522 / 788 |
| `invalid_args` | 2.68 | 3.20 | 1.19 | [0.94, 1.54] | 0.52 [-0.20, 1.22] | 487 / 536 |
| `exec_error` | 7.14 | 2.45 | 0.34 | [0.28, 0.41] | -4.70 [-5.60, -3.83] | 1298 / 410 |
| `missing_call` | 1.02 | 1.79 | 1.75 | [1.43, 2.17] | 0.77 [0.48, 1.07] | 186 / 300 |
| `malformed` | 0.29 | 0.93 | 3.19 | [1.84, 5.93] | 0.64 [0.29, 1.03] | 53 / 156 |
| `truncated` | 0.00 | 0.33 | – | – | 0.33 [0.08, 0.64] | 0 / 55 |
| `forced_quit` | 0.04 | 0.01 | 0.15 | [0.00, 1.02] | -0.03 [-0.07, 0.00] | 7 / 1 |

## IQ4_XS vs BF16: 1995 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 98, 'tau2_verified_retail': 413, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 30.45 | 38.16 | 1.25 | [1.19, 1.31] | 7.71 [5.99, 9.40] | 5520 / 6997 |
| `exec_error` | 7.12 | 9.93 | 1.39 | [1.24, 1.57] | 2.81 [1.77, 3.89] | 1291 / 1821 |
| `tool_error_payload` | 9.43 | 9.61 | 1.02 | [0.92, 1.13] | 0.18 [-0.81, 1.23] | 1710 / 1763 |
| `unknown_tool` | 7.00 | 8.24 | 1.18 | [1.03, 1.35] | 1.24 [0.23, 2.33] | 1269 / 1510 |
| `invalid_args` | 2.66 | 5.69 | 2.14 | [1.60, 2.84] | 3.03 [1.75, 4.42] | 482 / 1043 |
| `missing_call_direct_answer` | 2.88 | 3.28 | 1.14 | [1.00, 1.30] | 0.40 [0.01, 0.81] | 522 / 601 |
| `missing_call` | 1.03 | 0.83 | 0.81 | [0.65, 1.02] | -0.20 [-0.42, 0.02] | 186 / 152 |
| `malformed` | 0.29 | 0.47 | 1.60 | [0.97, 2.81] | 0.18 [-0.01, 0.37] | 53 / 86 |
| `forced_quit` | 0.04 | 0.11 | 2.97 | [1.15, 10.84] | 0.08 [0.01, 0.15] | 7 / 21 |
