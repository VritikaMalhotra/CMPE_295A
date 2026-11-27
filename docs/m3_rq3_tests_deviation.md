# Deviation note: RQ3 (H3) test code written after registration

Pre-registration: https://osf.io/xwq5y, §1 (H3: "Repairs close more of the success gap than of the failure gap")
and §6 (family RQ3). Companion to `docs/m3_h1_tests_deviation.md` and `docs/m3_rq2_tests_deviation.md`; to be
filed on OSF with them.

## What was missing

The code frozen at `1e13897` has no RQ3 test (no gap shares, no bootstrap of their difference, no Holm). Found
on 2026-11-27 with the H1 and RQ2 gaps.

## What was added

`analysis/rq3_tests.py` and `tests/test_rq3_tests.py`; no frozen file changed. It reuses the H1 module's loader
(relabelling with the frozen classifier) and `holm`.

## What was already known when these choices were made

Unlike H1 and RQ2, RQ3 was implemented **after** the H1 and RQ2 holdout results had been computed and read. Those
results show the any-failure rates of BF16, Q4_0, R1, Q8_0 and QAD-KL+CE (so most of the failure-gap side of H3),
and greedy-pass success for Q4_0 and the repairs (RQ2 secondary McNemar). Sampled-trial success of BF16, Q4_0 and
the three repairs on all tasks had not been computed. To keep the choices below from being steered by what was
seen, each follows the registration's wording where it has one, and the alternative is reported as a sensitivity.

## Choices §6 leaves open

1. **Pooling.** Pooled over the three core models, as RQ2 ("3 tests" means one per repair): cluster = task within a
   model, model = stratum; the bootstrap resamples tasks within each model, 10,000 resamples.
2. **Trials.** A cluster enters with the sampled trials (seeds 5–8) that BF16, Q4_0 and the repair all finished.
   Per cluster: mean episode success (bfcl verdict) and mean episode any-failure rate (failures / steps).
   Pooled quantities: means over clusters.
3. **Shares.** share_success = (S_repair − S_Q4_0) / (S_BF16 − S_Q4_0); share_failure = (F_Q4_0 − F_repair) /
   (F_Q4_0 − F_BF16); D = share_success − share_failure; H3 predicts D > 0.
4. **Decision.** §6: "supported if the 95% CI excludes 0 in the predicted direction", with Holm within RQ3. Both
   are applied through a bootstrap p: p = 2 · min(#(D* ≤ 0) + 1, #(D* ≥ 0) + 1) / (B + 1), capped at 1; Holm over
   the three; supported if Holm p < 0.05 and D > 0. The unadjusted 95% percentile CI, and whether it excludes 0
   in the predicted direction, are reported next to it.
5. **Tasks.** All tasks: §5 limits the success-based inclusion rule to "the secondary episode-success analysis",
   and selecting tasks on the same BF16 trials that define the success gap would inflate that gap. The inclusion
   sets are reported as a sensitivity.
6. **Undefined shares.** If a gap's point estimate is ≤ 0 the share is undefined and the test is "not testable".
   Bootstrap replicates with a gap ≤ 0 are dropped from the share and D intervals and counted; the test is
   flagged unstable when they exceed 2.5%.
7. **Randomness.** One generator, seed 295, fixed order (R1, Q8_0, QAD-KL+CE; pooled, then per model, then the
   inclusion sensitivity).

## Checks before the holdout run

- Unit tests (`tests/test_rq3_tests.py`, 4 tests, pass; 23 with the H1, RQ2, damage-table and stats tests): the
  share formula on a worked example; a planted D = 0.6 is detected and covered, a null is not flagged; a negative
  gap is "not testable" and a tiny gap is flagged unstable; only trials finished under all three configurations
  enter. The last test caught a column-naming bug (fixed before the development run).
- Development mechanics check (Slurm job in `h1check_dev`, `docs/m3_rq3_tests_dev_check.md`,
  `results/m3/rq3_tests_dev_check.json`): the development grid of the three core models with stand-ins
  (R1 → Q4_K_M, Q8_0 → IQ4_XS; QAD has no development data). An independent pandas recomputation from the run
  tables gives the module's per-model shares and D for Qwen3-0.6B exactly (0.920, 1.249, −0.329; 501 clusters).
- Code: `analysis/rq3_tests.py` SHA-256 `bf7ac209…` in the development check.

## Holdout run

_Filled in after the run._

## Status of the results

As for H1 and RQ2: confirmatory or exploratory under §2 is for the team and the advisor to decide; until then the
outputs are labelled "registered tests, implemented after registration". RQ3 also carries the disclosure above
(choices made after the H1 and RQ2 results were seen).
