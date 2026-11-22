# M3 results (Yashashav's packages: IQ4_XS + KL on both backends, R2a variants, QAD data)

Runbook: `docs/m3_quant.md`. QAD data: `docs/qad_data.md`. Everything here ran from Yashashav's account. The
sections through the QAD mix ran on 2026-11-09, at commits `fa32f84` → `4a021c8` of `m3-yashashav-prep`; B4 and R2b
ran 2026-11-10 to 11-11 (dates and job IDs in their sections). llama.cpp d812350; the CUDA build ran on
P100 nodes (gpuqs), the CPU build on 32-core cpuqs nodes.

## Checks before the runs (runbook step 0–1)
- Full test suite on the cluster: 183 passed, 8 skipped (before this work's new tests).
- `--tensor-type` probe on Qwen3-0.6B (job 110335): exactly one tensor (`blk.0.attn_q.weight`) came out Q4_0, the
  other 197 weight tensors BF16, norms F32. The R2b recipe works as written.
- Apptainer 1.5.5 builds without root on Ubuntu images (fakeroot command + root-mapped namespace; no subuid
  entry). Alpine images fail.
- FR-E5 check of the imatrix text: 0 of 36,719 WikiText-2 train lines share a 13-gram with the benchmark text
  (`results/m3/calib_ngram_check.json`). A planted BFCL turn and a planted tau2 task line were both caught.

## New model files (runbook step 2; jobs 110339–110344, 2–11 min each)
IQ4_XS (imatrix from 200 × 512 tokens of the checked text; llama.cpp's default IQ4_XS recipe), Q5_0-pure and
Q6_K-pure for all six models, in `$TESTBED_MODELS` next to the M2 files. SHA-256 sums, byte counts, weight counts:
`results/m3/models_manifest.json` (the M2 entries were re-checked unchanged). Whole-file bits per weight:

| Model | Q4_0-pure | IQ4_XS | Q4_K_M | Q5_0-pure | Q6_K-pure | Q8_0 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 4.57 | 4.79 | 5.15 | 5.57 | 6.63 | 8.57 |
| Qwen3-1.7B | 4.53 | 4.63 | 5.05 | 5.53 | 6.59 | 8.52 |
| LFM2.5-1.2B-Instruct | 4.52 | 4.53 | 5.00 | 5.52 | 6.58 | 8.52 |
| Qwen3-4B | 4.51 | 4.52 | 4.97 | 5.51 | 6.58 | 8.51 |
| SmolLM3-3B | 4.52 | 4.48 | 4.98 | 5.52 | 6.58 | 8.52 |
| LFM2.5-2.6B | 4.53 | 4.50 | 4.97 | 5.53 | 6.59 | 8.53 |

## KL against BF16 on both backends (runbook step 3)
Same settings as M2: WikiText-2 test (SHA-256 `173c87a5…`), context 512, 50 chunks, `llama-perplexity
--kl-divergence`. The BF16 base pass runs on the same backend as the variant.
- **CUDA:** M2's Q4_0-pure / Q4_K_M / Q8_0 runs (`results/m2/`) plus the new files (jobs 110345, 110390–110394).
- **CPU:** jobs 110336–110338 and 110395–110400.
- **Merging:** `scripts/merge_kl.py` checks that the parts share model, eval text, context and chunks, and that
  their BF16 perplexity agrees, before writing one file per model and backend: `results/m3/kl_{cuda,cpu}_<model>.json`.

Mean KLD:

| Model | Backend | Q4_0-pure | IQ4_XS | Q4_K_M | Q5_0-pure | Q6_K-pure | Q8_0 |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | CUDA | 0.2455 | 0.1023 | 0.1083 | 0.0605 | 0.0138 | 0.0033 |
| Qwen3-0.6B | CPU | 0.2445 | 0.1072 | 0.1143 | 0.0596 | 0.0195 | 0.0026 |
| Qwen3-1.7B | CUDA | 0.2184 | 0.0787 | 0.1069 | 0.0592 | 0.0153 | 0.0054 |
| Qwen3-1.7B | CPU | 0.2137 | 0.0831 | 0.1086 | 0.0572 | 0.0199 | 0.0031 |
| LFM2.5-1.2B-Instruct | CUDA | 0.1840 | 0.0958 | 0.1020 | 0.0472 | 0.0106 | 0.0030 |
| LFM2.5-1.2B-Instruct | CPU | 0.1839 | 0.0951 | 0.1058 | 0.0459 | 0.0121 | 0.0019 |
| Qwen3-4B | CUDA | 0.1250 | 0.0738 | 0.0780 | 0.0448 | 0.0176 | 0.0118 |
| Qwen3-4B | CPU | 0.1160 | 0.0677 | 0.0756 | 0.0351 | 0.0119 | 0.0017 |
| SmolLM3-3B | CUDA | 0.2400 | 0.0484 | 0.0448 | 0.0601 | 0.0144 | 0.0060 |
| SmolLM3-3B | CPU | 0.2368 | 0.0461 | 0.0426 | 0.0565 | 0.0122 | 0.0017 |
| LFM2.5-2.6B | CUDA | 0.2475 | 0.1501 | 0.1648 | 0.0757 | 0.0179 | 0.0056 |
| LFM2.5-2.6B | CPU | 0.2449 | 0.1551 | 0.1690 | 0.0730 | 0.0214 | 0.0034 |

With this, the KL part of the M3 exit is met: Q4_0-pure, Q4_K_M and IQ4_XS of every model, KL-checked against
BF16 on both backends.

**Observations to discuss with the team**
1. **SmolLM3-3B breaks the expected order.** Q5_0-pure (0.060 CUDA / 0.057 CPU) is worse than Q4_K_M (0.045 /
   0.043) and IQ4_XS. The other five models follow Q8_0 < Q6_K < Q5_0 < IQ4_XS ≤ Q4_K_M < Q4_0-pure on both backends.
   - Cause, confirmed 2026-11-10: the precision of SmolLM3's tied output head. See "SmolLM3-3B: the tied output
     head (B4)" below.
   - This affects how the R2a uniform-bits arm reads on SmolLM3 (runbook choice 1).
2. **IQ4_XS ≤ Q4_K_M on 5 of 6 models**, at 0.3–0.5 fewer bits per weight. The imatrix text is WikiText-2 train
   and the KL text is WikiText-2 test, so the KL check is in the same domain as the calibration. The agentic
   damage runs are the real test.
3. **Backend effect.** CPU KL is higher than CUDA for Q6_K-pure on 4 of 6 models (e.g. Qwen3-0.6B 0.0195 vs
   0.0138) and higher for Q4_K_M on 4 of 6, and lower for Q8_0 everywhere. Qwen3-4B Q8_0 is 0.0118 on CUDA vs
   0.0017 on CPU. This is consistent with `m2-phase1`'s parity result (backend stays a factor).

## SmolLM3-3B: the tied output head (B4, 2026-11-10)

**Tensor types.** Job 111868 read every model's files with llama.cpp's `gguf` reader
(`results/m3/tensor_types.jsonl`). SmolLM3-3B has **no `output.weight`**: the output head is tied to
`token_embd.weight`, so that one tensor (8.5% of the weights) is both the embedding and the head. The three
recipes treat it differently:

| SmolLM3 file | `token_embd` (= head) | Other 2-D tensors |
|---|---|---|
| Q4_K_M | Q6_K | 216 Q4_K, plus 36 at Q6_K (18 `attn_v`, 18 `ffn_down`: llama.cpp's `use_more_bits` rule picks 18 of 36 layers; `tensor_types.jsonl` records the totals, 37 Q6_K with the head) |
| IQ4_XS | Q6_K | 216 IQ4_XS, plus 36 at Q5_K |
| Q5_0-pure | **Q5_0** | 252 Q5_0 |

**Test** (job 111872, CPU). We built three diagnostic files in `models/diag_b4`, kept apart from the study
files. Each changes one thing. KL against BF16 used the step 3 settings (WikiText-2 test, ctx 512, 50 chunks).
Results: `results/m3/kl_cpu_diag_b4_SmolLM3-3B.json`. The first three rows are the step 3 CPU results.

| File | Head | Rest | Bits per weight | Mean KLD (CPU) | Same top token |
|---|---|---|---|---|---|
| Q5_0-pure | Q5_0 | Q5_0 | 5.52 | 0.0565 | 86.8% |
| Q4_K_M | Q6_K | Q4_K + 36 Q6_K | 4.98 | 0.0426 | 89.2% |
| IQ4_XS | Q6_K | IQ4_XS + 36 Q5_K | 4.48 | 0.0461 | 88.9% |
| Q5_0-pure-embdQ6K | **Q6_K** | Q5_0 | 5.61 | **0.0284** | 91.0% |
| Q5_0-pure-kmQ6K | Q6_K | Q5_0 + the same 36 at Q6_K | 5.76 | 0.0219 | 91.7% |
| Q4_K_M-embdQ5_0 | **Q5_0** | Q4_K + 36 Q6_K | 4.89 | **0.0698** | 85.9% |

**Cause: the head's precision.**
- Raising only the tied head from Q5_0 to Q6_K halves Q5_0-pure's KLD (0.0565 → 0.0284). That puts it below
  Q4_K_M, at +0.09 bits per weight.
- Lowering only the head in Q4_K_M from Q6_K to Q5_0 raises its KLD from 0.0426 to 0.0698. That is worse than
  Q5_0-pure.
- The 36 `attn_v`/`ffn_down` tensors that Q4_K_M keeps at Q6_K explain a further, smaller part (0.0284 → 0.0219).
- Qwen3-4B, LFM2.5-1.2B and LFM2.5-2.6B also tie the head, and `--pure` also puts it at Q5_0, but they keep the
  expected order. So SmolLM3 is unusually sensitive to the precision of its head. This also fits SmolLM3 having
  the widest Q4_0-pure gap of the six (0.237 against 0.043 for Q4_K_M on CPU).

**Consequence for R2a (team call, not made here).** On SmolLM3, the "uniform 5.5 bits" arm (Q5_0-pure) mostly
measures the head at 5.5 bits, not 5.5 bits everywhere. Options:
- report the head type next to every SmolLM3 R2 result; or
- add a Q5_0 variant with the head at Q6_K (5.61 bits per weight) as a sensitivity check.
R2b's selection is not affected: it keeps the embedding/head at the base type and ranks only block weights.

## QAD training mix and R2b calibration conversations (package K; job 110408, 6 min)
Built per `docs/qad_data.md`; summary in `results/m3/qad_mix_manifest.json` (`pass: true`).

| Source | Candidates | Parse failed | 13-gram hit | Duplicate | Kept | Tokens (Qwen3-0.6B) |
|---|---|---|---|---|---|---|
| Toucan (R2b holdout) | 2,000 | 0 | 0 | 0 | 2,000 | 9.09M (incl. tool schemas) |
| Toucan | 4,025 | 0 | 0 | 170 (holdout hits) | 3,855 | 15.00M |
| Nemotron tool_calling | 2,713 | 0 | 73 | 16 | 2,624 | 5.00M |
| ToolACE | 8,544 | 452 | 3 | 1 | 8,088 | 5.00M |
| FineWeb-Edu | 19,871 | 0 | 0 | 2 | 19,869 | 20.00M |
| Cosmopedia v2 | 7,110 | 0 | 0 | 1 | 7,109 | 5.00M |
| **Training mix** | | | | | **41,545** | **50.00M** |

`rank_text.txt` (2,000 conversations, 7.0M tokens, no tool schemas) shares no 13-gram with the benchmark text, and
its ids are disjoint from the training set.

**R2b ranking window (default D-1, decided 11-09; set 2026-11-10).**
- **Ranking text.** `rank_text_t256.txt` is the same 2,000 held-out conversations with every tool reply cut to 256
  Qwen3-0.6B tokens (`train/build_qad_mix.py rank-text`, job 111874):
  - 1,690 of 6,806 tool replies cut (`results/m3/rank_text_t256.json`; log `results/m3/logs/rank-text_111874.out`);
  - 4,632,056 tokens;
  - SHA-256 `dc06ce1f93061baccb8088fc96e4673472bacc3884029febea038a9253cfa438`;
  - shares no 13-gram with the benchmarks (`rank_text_t256.json` next to it).
- **Timing probe on Qwen3-0.6B** (ctx 512; `slurm/rank_probe.sbatch`; logs in `results/m3/logs/rank-probe_*.out`):

  | Chunks | GPU | Base pass | Per tensor | Base-logits file | Job |
  |---|---|---|---|---|---|
  | 20 | A100 | 11 s | 4.9–6.8 s (4 tensors) | 1.5 GB | 111875 |
  | 200 | A100 | 32 s | 15.5–15.7 s (the first of 4 took 25.8 s) | 15.5 GB | 111875 |
  | 1,000 | P100 | 242 s | 200.6–205.9 s (3 tensors) | 77.5 GB | 111927 |

  The base-logits file grows by 77.5 MB per chunk (Qwen3's 151k vocabulary): the ranking run's 1,000-chunk file
  `runs/rank/Qwen3-0.6B/BF16.kld` is 77,491,448,020 bytes. The 20- and 200-chunk sizes are that rate × chunks; the
  probe files were deleted after the probe. Past a few tens of GB it no longer
  fits the job's page cache, so every tensor rereads it from NFS. That is why the 1,000-chunk point was measured
  rather than extrapolated from 200.
- **RANK_CHUNKS = 1,000.** At 1,000 chunks, one model's ranking takes 196 tensors × about 204 s ÷ 2 GPUs ≈
  5.6 h, within the 6 h budget. The fitted limit is about 1,080 chunks (6 h × 2 GPUs ÷ 196 tensors = 220 s per tensor, at about 0.204 s
  per chunk), so we used the largest measured value
  under it. The window is 512,000 tokens, about 11% of the ranking text, or about 220 conversations (20 chunks
  covered about 4).
  - Assumption: the two array tasks reading the same base file at once may run slower than the single-task probe.
- **Recorded.** Every score file records ctx, chunks and the text's SHA-256. `rank_tensors.py run` refuses base
  or score files made with other settings, and `select` refuses a mix.
- **Submitted 2026-11-10** (`scripts/rank_submit.sh`; base → ranking array of 2 tasks → select + build + FR-E10
  check → KL on CUDA and CPU):

  | Model | Base | Ranking array | Select + build | KL CUDA | KL CPU |
  |---|---|---|---|---|---|
  | Qwen3-0.6B | 112000 | 112001 | 112002 | 112003 | 112004 |
  | Qwen3-1.7B | 112005 | 112006 | 112007 | 112008 | 112009 |
  | LFM2.5-1.2B-Instruct | 112010 | 112011 | 112012 | 112013 | 112014 |

**Qwen3-0.6B R2b result (jobs 112001–112004; the last one ended 2026-11-11 17:52).**
- **Ranking.** All 196 candidate tensors have a score file, and all 196 record ctx 512, 1,000 chunks and text SHA-256
  `dc06ce1f…`. Array 112001 ran on two P100s (g2, g4; nodes and times here from `sacct`).
  - Highest score: `blk.2.ffn_down.weight`, KLD 0.0105 when it alone is Q4_0.
  - Lowest: `blk.1.attn_k.weight`, 0.00016.
- **Selection** (`results/m3/r2b_selection_Qwen3-0.6B.json`; build input `r2b_tensor_types_Qwen3-0.6B.txt`).
  78 tensors go to Q8_0, 42.6% of linear weights, for a predicted 5.501 bits per weight.
  - By kind, of 28 each: ffn_up 24, attn_v 21, ffn_down 13, ffn_gate 10, attn_output 5, attn_k 3, attn_q 2.
  - By layer: spread over layers 0–22, plus 26 and 27. None come from layers 23–25.
- **Build and FR-E10** (job 112002, `results/m3/r2b_check_Qwen3-0.6B.json`). Measured 5.5646 bits per weight, so
  FR-E10 (5.5 ± 0.1) passes.
  - The 0.0633 above the prediction is exactly the file's 5,951,808 bytes that are not tensor data (GGUF header,
    metadata and padding): 522,817,856 file bytes minus 516,866,048 tensor bytes from the weight counts below.
    The Q4_0-pure file has the same 5,951,808 non-tensor bytes.
  - Weights by type: Q4_0 563.9M, Q8_0 187.7M, F32 65.5k.
  - Added to `results/m3/models_manifest.json` as `R2b-LS5.5` (SHA-256 `5d20e96f…`).
- **KL against BF16** (112003 CUDA on a P100, 112004 CPU). Settings are the WikiText-2 ones of the table above.
  Merged into `results/m3/kl_{cuda,cpu}_Qwen3-0.6B.json` with `scripts/merge_kl.py`; the BF16 perplexity of the
  parts agrees.

  | Qwen3-0.6B | Bits per weight | KLD CUDA | KLD CPU | Same top token, CUDA |
  |---|---|---|---|---|
  | IQ4_XS | 4.79 | 0.1023 | 0.1072 | 83.7% |
  | Q4_K_M | 5.15 | 0.1083 | 0.1143 | 83.3% |
  | **R2b-LS5.5** | **5.56** | **0.1021** | **0.1014** | **83.0%** |
  | Q5_0-pure | 5.57 | 0.0605 | 0.0596 | 86.6% |

- **Reading.** At the same file size, R2b's WikiText KL is about 1.7× that of uniform Q5_0-pure. Against IQ4_XS, which
  uses 0.77 fewer bits per weight, it ties on CUDA (0.1021 vs 0.1023) and is about 5% lower on CPU (0.1014 vs 0.1072).
  - This check is outside R2b's ranking domain. R2b ranks tensors by KL on held-out agent conversations, not on
    WikiText. Whether it helps on agent steps is not measured: no damage run uses the R2b file yet.
  - A possible reason, not tested: R2b leaves 57% of linear weights at Q4_0 and spends the extra bits on the other
    43%, whereas Q5_0 raises every tensor by one bit.
  - The ranking scores each tensor alone. If the damage from tensors kept at Q4_0 adds up in a way those
    single-tensor scores don't capture, the selection can be far from the best one for its size.
- **The other two core models: submitted 2026-11-10, not finished.**
  - Base logits are done: Qwen3-1.7B 112005 (17:51–17:58) and LFM2.5-1.2B-Instruct 112010 (17:58–18:03).
  - Ranking task 112006_0 (1.7B) started 18:03 on g2.
  - Every P100 node has one GPU, and the 2-node cap is taken by cs002 (agents) and one P100. So the ranking tasks
    run one at a time (112006_1 and 112011 wait), not two in parallel as the 6 h budget assumed.

## R2b for Qwen3-1.7B and LFM2.5-1.2B-Instruct (resumed 2026-11-19, Ekant's account)

The Nov 10 arrays had failed: Qwen3-1.7B hit the 8 h limit at 179 of 196 tensors, and LFM2.5-1.2B crashed on its
first tensor (llama.cpp printed a non-UTF-8 byte; fixed in `4c23d1a`). Resumed on P100s with the existing BF16
base logits and score files (`RANK_SKIP_BASE=1`, `RANK_TIME=12:00:00`): ranking 156550 / 156555, select + build
156551 / 156556, KL 156552–156553 / 156557–158. Same settings as 0.6B (ctx 512, 1,000 chunks, text SHA-256
`dc06ce1f…`; KL on WikiText-2 test, 50 chunks).

- **Selection and FR-E10** (`results/m3/r2b_selection_*.json`, `r2b_check_*.json`, `r2b_tensor_types_*.txt`):
  - Qwen3-1.7B: 59 tensors to Q8_0, 36.0% of linear weights, predicted 5.501, measured **5.524** bits per weight.
  - LFM2.5-1.2B: 29 tensors to Q8_0, 29.8% of linear weights, predicted 5.555, measured **5.573** bits per weight.
  - Both pass 5.5 ± 0.1. Added to `results/m3/models_manifest.json` as `R2b-LS5.5` (SHA-256 `f6903fb2…`,
    `4585181b…`).
  - LFM2.5's ten `shortconv.conv.weight` kernels score KL 0 exactly: llama-quantize leaves them unquantized, so they
    cost ranking time (about 9 min each) but cannot be selected. Drop them from `rank_tensors.py list` before the
    next ranking.
- **KL against BF16** (merged into `results/m3/kl_{cuda,cpu}_<model>.json`):

  | Model | File | Bits per weight | KLD CUDA | KLD CPU |
  |---|---|---|---|---|
  | Qwen3-1.7B | IQ4_XS | 4.63 | 0.0787 | 0.0831 |
  | | Q4_K_M | 5.05 | 0.1069 | 0.1086 |
  | | **R2b-LS5.5** | **5.52** | **0.0920** | **0.0900** |
  | | Q5_0-pure | 5.53 | 0.0592 | 0.0572 |
  | LFM2.5-1.2B | IQ4_XS | 4.53 | 0.0958 | 0.0951 |
  | | Q4_K_M | 5.00 | 0.1020 | 0.1058 |
  | | **R2b-LS5.5** | **5.57** | **0.0975** | **0.0943** |
  | | Q5_0-pure | 5.52 | 0.0472 | 0.0459 |

- **Reading.** The 0.6B pattern repeats: at the same file size, R2b's WikiText KL is 1.6× (1.7B) and 2.1× (LFM)
  that of uniform Q5_0-pure, and no better than IQ4_XS at about 0.9–1.0 fewer bits per weight. This check is
  outside R2b's ranking domain (agent conversations); contrast 5 of the registered RQ2 family (R2b vs Q5_0, two
  sided) is the real test.

## R3 trainer and pilot (2026-11-21, Ekant; R3 owner Yashashav)

No production QAD trainer existed (only the 50-step S7 toy), so `train/qad.py` was written: the three arms on
identical data order, exact Q4_0 fake quantization of every tensor `--pure Q4_0` quantizes (including the tied
embedding/output matrix), fp32 master weights with BF16 autocast, 30-minute checkpoints and a USR1 requeue that
resumes with bit-identical losses (`tests/test_qad.py`). Data: the QAD mix packed per model into 2,048-token
sequences (`$CMPE295_GROUP/data/qad_pack/<model>`: Qwen3 25,132 sequences, LFM2.5 24,606; 58 LFM2.5 records the
template could not render). Eval: 64 sequences of the R2b ranking conversations, disjoint from the mix.

Held-out per-token KL to the BF16 teacher, Q4_0 student (fake quantization on) after 150 steps (4.9M tokens), H100:

| Model | Plain Q4_0 | QAD-KL | KL+CE, CE 0.5 | KL+CE, CE 0.1 | Control (KL+CE 0.5, FQ off) | tok/s (KL) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.177 | 0.080 | 0.109 | 0.082 | 0.203 | 14,500 |
| Qwen3-1.7B | 0.185 | 0.084 | 0.166 (60 steps) | 0.091 | – | 8,060 |
| LFM2.5-1.2B | 0.408 | 0.245 | 0.496 (60 steps) | 0.273 | – | 14,160 |

- QAD-KL roughly halves the Q4_0 student's divergence on the Qwen3 models and cuts it 40% on LFM2.5 within 5M tokens.
- A CE weight of 0.5 pulls the student away from its teacher (its 16-bit KL rises) and on LFM2.5 makes Q4_0 worse
  than no training; 0.1 keeps nearly all of the QAD-KL gain. **Registered setting: CE weight 0.1**, learning rate
  1e-5 for every model (pre-registration §2). Chosen on held-out KL only, not on any benchmark.
- Export check on the 0.6B QAD-KL pilot: 197 Q4_0 tensors, all bit-identical to the trained codes; count matches the
  197 fake-quantized weights.
- Peak memory: 35 GB (0.6B), 62 GB (1.7B), 37 GB (LFM2.5) at micro-batch 2. Full pass per arm: about 1 h (0.6B,
  LFM2.5) and 2 h (1.7B).

## R3 full runs (2026-11-21)

Nine runs (jobs 162971–162979): three arms × three core models, one pass over the packed QAD mix (1,570 steps,
51.4M tokens for Qwen3; 1,537 steps for LFM2.5), CE weight 0.1, learning rate 1e-5, one H100 each, no requeues
(45 min to 1.9 h). Summary with every eval point: `results/m3/r3_training.json`.

**Held-out agent-conversation KL to BF16** (64 R2b ranking conversations, Q4_0 student at the end of training):

| Model | Plain Q4_0 | QAD-KL | QAD-KL+CE | Control (trained at 16 bits, then Q4_0) |
|---|---|---|---|---|
| Qwen3-0.6B | 0.177 | **0.043** | 0.049 | 0.160 |
| Qwen3-1.7B | 0.185 | **0.050** | 0.059 | 0.150 |
| LFM2.5-1.2B | 0.408 | **0.145** | 0.166 | 0.285 |

**Export** (`train/export_r3.py`, jobs 164193–164205 and 164247–164249; reports `results/m3/r3_export_*.json`):
every Q4_0 file holds exactly the trained codes (max difference 0.0; 197 tensors for Qwen3, 93 for LFM2.5, each equal
to the number of fake-quantized weights). Files `$TESTBED_MODELS/<model>-R3-{QAD-KL,QAD-KLCE,CTRL}-Q4_0-pure.gguf`
and `<model>-R3-CTRL-BF16.gguf`, in `results/m3/models_manifest.json`. The first 1.7B exports failed because the
base model's shard index was copied into `final/` (fixed in `7cc9875`).

**WikiText-2 KL to BF16** (out of the training domain; CUDA, CPU within 0.003; merged into `results/m3/kl_*`):

| Model | Plain Q4_0 | QAD-KL | QAD-KL+CE | Control Q4_0 | Control BF16 |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.246 | **0.069** | 0.072 | 0.227 | 0.006 |
| Qwen3-1.7B | 0.218 | **0.081** | 0.087 | 0.191 | 0.012 |
| LFM2.5-1.2B | 0.184 | 0.191 | **0.157** | 0.238 | 0.068 |

- **QAD recovers most of the Q4_0 divergence** on the Qwen3 models (−72% / −63% on WikiText, −73–76% in domain),
  far more than the train-longer control (−7–13%), so the gain comes from training under quantization.
- **LFM2.5 does not transfer out of domain:** QAD-KL cuts in-domain KL from 0.41 to 0.15 but is slightly worse than
  plain Q4_0 on WikiText; KL+CE is 15% better. Its control also moves the 16-bit model far more (KL 0.068 vs 0.006–0.012
  for Qwen3). The registered agent-task contrasts (RQ2 contrasts 8–10) are the test that counts.
- These checks are development data; R3's confirmatory evaluation is stage C3 on the BFCL holdout.
