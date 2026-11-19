# H1 damage preview: qwen3-1.7b (development data, preview)

**Development data, preview.** Not confirmatory: the pre-registration is not filed, and the cells may still be running (pair counts below). Method: paired (task, seed); seeds 1-4; cluster bootstrap over tasks, 10,000 resamples; ratio of pooled rates; classifier codebook v0. JSON: `results/m3/damage_preview_qwen3-1.7b.json`.

## Q4_0-pure vs BF16: 1904 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 93, 'tau2_verified_retail': 327, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 26.52 | 32.29 | 1.22 | [1.15, 1.28] | 5.77 [4.17, 7.35] | 5642 / 6410 |
| `exec_error` | 9.11 | 11.08 | 1.22 | [1.10, 1.34] | 1.97 [0.95, 2.98] | 1939 / 2200 |
| `tool_error_payload` | 8.49 | 7.73 | 0.91 | [0.81, 1.02] | -0.76 [-1.70, 0.13] | 1806 / 1534 |
| `unknown_tool` | 5.83 | 6.24 | 1.07 | [0.91, 1.25] | 0.41 [-0.57, 1.35] | 1240 / 1238 |
| `missing_call` | 1.43 | 2.54 | 1.77 | [1.50, 2.11] | 1.10 [0.77, 1.47] | 305 / 504 |
| `invalid_args` | 0.94 | 2.13 | 2.28 | [1.38, 3.98] | 1.20 [0.44, 2.06] | 199 / 423 |
| `missing_call_direct_answer` | 0.55 | 2.04 | 3.68 | [2.80, 5.07] | 1.49 [1.14, 1.88] | 118 / 405 |
| `malformed` | 0.14 | 0.39 | 2.79 | [1.33, 8.15] | 0.25 [0.07, 0.44] | 30 / 78 |
| `truncated` | 0.00 | 0.08 | – | – | 0.08 [0.03, 0.14] | 0 / 15 |
| `forced_quit` | 0.02 | 0.07 | 2.79 | – | 0.04 [-0.01, 0.09] | 5 / 13 |

## Q4_K_M vs BF16: 1994 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 101, 'tau2_verified_retail': 409, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 26.92 | 28.96 | 1.08 | [1.01, 1.14] | 2.04 [0.29, 3.78] | 6019 / 6238 |
| `exec_error` | 10.32 | 11.20 | 1.08 | [0.98, 1.20] | 0.88 [-0.21, 1.97] | 2308 / 2412 |
| `tool_error_payload` | 8.08 | 8.86 | 1.10 | [1.00, 1.21] | 0.79 [-0.03, 1.63] | 1806 / 1909 |
| `unknown_tool` | 5.56 | 3.44 | 0.62 | [0.51, 0.75] | -2.12 [-3.03, -1.24] | 1243 / 741 |
| `invalid_args` | 0.91 | 2.64 | 2.89 | [2.02, 4.37] | 1.72 [1.07, 2.46] | 204 / 568 |
| `missing_call` | 1.36 | 1.49 | 1.09 | [0.91, 1.31] | 0.12 [-0.14, 0.40] | 305 / 320 |
| `missing_call_direct_answer` | 0.53 | 0.80 | 1.52 | [1.09, 2.17] | 0.28 [0.05, 0.53] | 118 / 173 |
| `malformed` | 0.13 | 0.42 | 3.11 | [1.77, 7.99] | 0.28 [0.14, 0.45] | 30 / 90 |
| `forced_quit` | 0.02 | 0.10 | 4.36 | – | 0.08 [0.03, 0.12] | 5 / 21 |
| `truncated` | 0.00 | 0.02 | – | – | 0.02 [0.00, 0.04] | 0 / 4 |

## IQ4_XS vs BF16: 1979 paired episodes, 501 tasks

Pairs by benchmark: {'bfcl_multi_turn_base': 400, 'bfcl_multi_turn_miss_func': 400, 'bfcl_multi_turn_miss_param': 400, 'tau2_verified_airline': 95, 'tau2_verified_retail': 400, 'tau2_verified_telecom': 284}

| Failure class | BF16 per 100 steps | 4-bit per 100 steps | Amplification ratio | 95% CI | Difference per 100 steps [95% CI] | Events BF16 / 4-bit |
|---|---|---|---|---|---|---|
| **any failure** | 26.99 | 27.63 | 1.02 | [0.97, 1.07] | 0.64 [-0.72, 1.96] | 5977 / 5861 |
| `unknown_tool` | 5.60 | 8.45 | 1.51 | [1.29, 1.75] | 2.85 [1.72, 3.98] | 1241 / 1793 |
| `tool_error_payload` | 8.16 | 8.01 | 0.98 | [0.92, 1.05] | -0.15 [-0.70, 0.39] | 1806 / 1699 |
| `exec_error` | 10.24 | 7.78 | 0.76 | [0.70, 0.82] | -2.46 [-3.18, -1.76] | 2268 / 1651 |
| `missing_call` | 1.38 | 1.21 | 0.88 | [0.73, 1.05] | -0.17 [-0.41, 0.07] | 305 / 257 |
| `invalid_args` | 0.92 | 1.09 | 1.19 | [0.81, 1.69] | 0.17 [-0.20, 0.55] | 204 / 232 |
| `missing_call_direct_answer` | 0.53 | 0.99 | 1.86 | [1.40, 2.54] | 0.46 [0.24, 0.70] | 118 / 210 |
| `forced_quit` | 0.02 | 0.05 | 2.30 | – | 0.03 [-0.01, 0.07] | 5 / 11 |
| `malformed` | 0.14 | 0.03 | 0.21 | [0.00, 0.65] | -0.11 [-0.21, -0.02] | 30 / 6 |
| `truncated` | 0.00 | 0.01 | – | – | 0.01 [0.00, 0.02] | 0 / 2 |
