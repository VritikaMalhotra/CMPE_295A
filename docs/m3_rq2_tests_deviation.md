# Deviation note: RQ2 (H2) test code written after registration

Pre-registration: https://osf.io/xwq5y, §6 (H2, family RQ2), §2 and §7. Companion to
`docs/m3_h1_tests_deviation.md`, which covers the same situation for H1; to be filed on OSF with it.

## What was missing

The code frozen at `1e13897` has no RQ2 tests: no paired permutation test, no Holm correction, no TOST and no
model-stratified pooling. It does have the per-model effect size (`analysis/damage_table.ratio_ci`) and McNemar's
exact test (`testbed/stats.mcnemar_exact`). Found on 2026-11-27 with the H1 gap, before any C3 outcome was read.

## What was added

`analysis/rq2_tests.py` and `tests/test_rq2_tests.py`. No frozen file changed. The module reuses the H1 module
(`perm_test_paired`, `holm`, `wilson`, the loader and its relabelling with the frozen classifier) and the frozen
`ratio_ci` (per-model estimates) and `mcnemar_exact` (greedy secondary).

Configurations, by (variant, repair) in the run specs: Q4_0 = (`q4_0`, none) from C1; R1 = (`q4_0`, `r1`);
Q5_0, Q8_0, R2b, R2c = (`q5_0` / `q8_0` / `r2b` / `r2c_tool`, `r2`); QAD-KL, QAD-KL+CE, train-longer =
(`r3_qad_kl` / `r3_qad_klce` / `r3_ctrl`, `r3`); all CUDA. Before the C3 run only these labels and the spec
counts were read from the C3 directories.

## Choices §6 leaves open (fixed before any C3 outcome was read)

1. **Cluster and pooling.** A cluster is a task within one core model (task × model); the three models enter as
   strata. Per-cluster rate: the mean over the cluster's paired sampled trials (seeds 5–8) of each episode's
   failures / steps (as for H1). Test statistic: the mean per-cluster difference (treatment − comparison),
   10,000 random sign flips, two-sided.
2. **Verdicts.** "Lower" contrasts (1, 3, 4, 6, 8, 9, 10): supported if Holm p < 0.05 and the pooled ratio
   (treatment / comparison) < 1. Two-sided contrasts (2, 5): "significant" if Holm p < 0.05, with the direction
   from the pooled ratio. Contrast 7: "matches" only if the Holm-adjusted TOST p < 0.05.
3. **Pooled ratio CI.** Paired cluster bootstrap stratified by model (tasks resampled within each model),
   10,000 resamples. Per-model estimates use the frozen `ratio_ci` (unstratified within a model).
4. **Format failures** (contrast 1): `malformed` + `unknown_tool` + `invalid_args` per step (§4).
5. **Contrast 7 (TOST).** Outcome: episode success (bfcl verdict), per-cluster mean over paired sampled trials.
   Margin ± 3 percentage points. Two one-sided sign-flip permutation tests on the shifted differences
   (d + 0.03 > 0 and d − 0.03 < 0), p_TOST = the larger; a 90% bootstrap CI is shown. Tasks: the registered
   inclusion sets (§5: the success analysis excludes tasks the BF16 reference fails in all four trials), with an
   all-task sensitivity. With the holdout inclusion sets this leaves few clusters (Qwen3-0.6B 11, Qwen3-1.7B 20,
   LFM2.5-1.2B 20 tasks), so equivalence is hard to show; this was known before the run.
6. **Secondary McNemar** (report only): greedy pass (seed 0), episode success, pairs = model × task pooled over
   the three models, frozen `mcnemar_exact`; Holm within these 10. §6 does not say which binary outcome; success
   is the one the frozen `stats.compare` uses.
7. **Randomness.** One generator, seed 295, fixed order (the ten contrasts, then the secondary tests).

## Checks before the C3 run

- Unit tests (`tests/test_rq2_tests.py`, 7 tests, pass; 19 with the H1, damage-table and stats tests): one-sided
  permutation p matches exact enumeration; TOST shows equivalence for a null difference and not for a 6-point
  shift or noisy data; the stratified ratio CI covers a planted 0.7 ratio; a planted reduction is detected and
  a null contrast is not; contrast 7 uses the inclusion set and reports the all-task sensitivity; McNemar uses
  greedy pairs only; the contrast list equals §6.
- Development mechanics check (Slurm 172343; `docs/m3_rq2_tests_dev_check.md`,
  `results/m3/rq2_tests_dev_check.json`): the development H1 grid of the three core models with stand-in names
  (Q8_0 → BF16, Q5_0 → IQ4_XS, R2c → Q4_K_M; no repair has run on development tasks). The stand-in "Q8_0 vs
  Q4_0" per-model ratios equal 1 / the H1 development ratios to 6 decimals for all three models; contrast 7 runs
  with the development inclusion sets (84 clusters) and its all-task sensitivity (1,503). A guard now refuses two
  names mapped to the same configuration (the first stand-in map did that and silently dropped a contrast).
- Code: `analysis/rq2_tests.py` SHA-256 `cf9fa966…` in the development check.

## Holdout run

_Filled in after the run._

## Status of the results

As for H1: whether these count as confirmatory or carry the exploratory label of §2 is for the team and the
advisor to decide; until then they are labelled "registered tests, implemented after registration".
