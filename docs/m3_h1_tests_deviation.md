# Deviation note: H1 test code written after registration

Pre-registration: https://osf.io/xwq5y (filed 2026-11-22). This note covers §6 (H1 analysis plan), §2 (blinding:
"C10 labels any analysis run with a different code hash as exploratory") and §7 (frozen analysis code, commit
`1e13897`). It is meant to be filed on OSF as a transparent change before the H1 results are reported.

## What was missing

The code frozen at `1e13897` has the registered effect sizes (`analysis/damage_table.py`: ratio of pooled rates,
paired cluster bootstrap CI), the mixed-effects secondary (`analysis/mixed_effects.R`, `testbed/mixed_effects.py`)
and McNemar/bootstrap for episode success (`testbed/stats.py`). It does **not** have the H1 tests that §6 names:
the paired permutation test (10,000 permutations) on per-task any-failure rates, the Holm correction within each
family, and the Poisson GEE of family H1-loc. Found on 2026-11-27, before any H1 test was run on the holdout.

## What was added

`analysis/h1_tests.py` (new file) and `tests/test_h1_tests.py`. No frozen file changed: `git diff 1e13897` over
the seven files of §7 is empty. The new module imports the frozen pieces rather than copying them
(`damage_table.ratio_ci`, `classifier.classify_step`, `mixed_effects.depth_bin`). It has its own loader, because the
frozen `which_tasks_fail.load` keeps only the development seeds 1–4; it reads each run's Parquet tables (rebuilt
from the per-episode files by the frozen report) and relabels every step with the frozen classifier (codebook v0),
as the frozen damage table does.

## Choices §6 leaves open (fixed before any 4-bit holdout cell was analysed)

1. **Per-task rate.** §4: "sampled trials are averaged within a task before paired tests". The per-task rate is
   the mean, over the task's paired trials, of each episode's failures / steps.
2. **Permutation test.** Random sign flips of the per-task differences (4-bit − BF16), 10,000 flips, statistic
   the mean difference, two-sided, p = (1 + #{|perm| ≥ |obs|}) / (10,000 + 1). Pairing as §5: a trial enters
   only when both configurations finished it.
3. **"Ratio > 1"** uses the frozen ratio of pooled rates (`damage_table.ratio_ci`), with its bootstrap CI.
4. **H1-loc GEE.** Poisson, log link, independence working correlation, robust sandwich covariance without a
   small-sample correction (the statsmodels GEE default; the implementation matches statsmodels to 1e-6 in the
   unit test). Rows: steps of paired trials, BF16 and one format, six models, CUDA, sampled seeds. Terms:
   precision, step type, depth bin, model (fixed effect), precision × step type, precision × depth bin. Cluster:
   the task (benchmark/task ID), pooled over models. Step type is two-level as in §4: a tool-call step has at
   least one tool-call block or a call class (`malformed`, `unknown_tool`, `invalid_args`, `exec_error`,
   `tool_error_payload`, `ok_call`); every other step is a text step. Depth bins: the frozen `depth_bin`
   (0–1, 2–3, 4–6, 7+ by step index within the episode). Step-type test: Wald z on precision × call. Depth test:
   joint Wald χ² (3 df) on precision × depth. Direction: precision × call > 0; for depth, precision × d7+ > 0.
5. **H1-cpu.** Greedy pass (seed 0), one trial per task, so the per-task rate is that episode's rate; rates
   shown with a 95% Wilson interval on the pooled step proportion (SD = 0). The CUDA vs CPU difference in the
   BF16 → 4-bit change is the per-task difference of differences against the C1 CUDA greedy pass on the same
   100 `multi_turn_base` tasks, with a 10,000-resample bootstrap CI over tasks.
6. **Report-only sensitivity** (added after the development check, before the holdout run): each H1-type
   contrast also reports the permutation p on the within-task pooled rate (failures / steps summed over the
   task's paired trials). It never decides support. Reason: on development data the two weightings disagree for
   some cells (Qwen3-0.6B Q4_0: per-task mean rate 0.298 → 0.368, p < 0.0001, but pooled ratio 1.02).
7. **Randomness.** One `numpy` generator, seed 295 (the frozen damage table's default), used in a fixed order.

## Checks before the holdout run

- Unit tests (`tests/test_h1_tests.py`, 8 tests, all pass; with `tests/test_damage_table.py`, 9 pass): the permutation test detects a planted shift, keeps
  its size under the null (200 null datasets), and matches exact enumeration; Holm matches a textbook example;
  Wilson matches a known value; the GEE recovers planted precision × call and precision × depth effects, does not
  flag a null, and matches statsmodels' GEE (coefficients and robust SEs).
- Development data (H1 grid, seeds 1–4, 501 tasks; exploratory under §8): see "Development check" below.

## Development check

Development H1 grid (24 run directories, seeds 1–4, 501 tasks incl. τ²), Slurm 172330 and, with the final
code, 172339 (`analysis/h1_tests.py` SHA-256 `719781e6…`, identical on the laptop and the cluster). Output:
`docs/m3_h1_tests_dev_check.md`, `results/m3/h1_tests_dev_check.json` (exploratory, §8).

- **Agreement with the frozen damage table:** for Qwen3-0.6B and SmolLM3-3B (6 cells), the frozen
  `damage_table.py` run in the same job and the new module give identical pair counts, event and step counts
  and ratios (to 6 decimals).
- **Relabelling:** 0 steps where the frozen classifier's label differs from the logged one (24 directories).
- **Runs end to end** on about 250,000 steps per H1-loc format in under a minute; GEE converged for all three.
- Development results, for orientation only: H1 8/18, H1-tep 0/15, H1-sens 10/18, H1-loc 0/6 (interactions
  significant but in the opposite direction for Q4_0 and Q4_K_M: larger damage in text steps and shallow bins).

## What had been seen of the holdout before the tests ran

- Run monitoring as §3 allows: finished, pending and excluded counts (`scripts/run_status.py`).
- The per-run `report.md` files that the frozen report job writes when an array ends.
- During monitoring on 2026-11-26, the step failure-class counts of `hold_h1_qwen3_0p6b` (Q4_0, Q4_K_M and IQ4_XS
  pooled; no BF16 comparison) and the Q4_K_M rows of the CPU Qwen3-0.6B report were printed while checking the
  run tables. No BF16 vs 4-bit comparison of the holdout was computed or looked at before the tests ran.
- The inclusion sets (`results/m3/inclusion_holdout_<model>.json`) were built from the BF16 references on
  2026-11-27, before the H1 run, as §5 requires. Five of the six models have fewer than 120 included multi-turn
  tasks (the secondary success analysis is affected; the step-level H1 tests use every task, decision 11).

## Status of the results

Whether H1 results from this code count as confirmatory (registered tests, code written after registration and
disclosed here) or carry the exploratory label of §2 is for the team and the advisor to decide. Until then the
outputs are labelled "registered tests, implemented after registration".
