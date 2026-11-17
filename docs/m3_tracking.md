# M3 tracking: run results and next phase

Living tracker for the damage-map phase (M3, due 2026-12-04). Counts come from `scripts/run_status.py` run on
the cluster against `/scratch/cmpe295-bertin/runs/` (`results/m3/run_status.json`), never typed in.
Development data only: the registered test runs after the pre-registration is filed.

Last refresh: **2026-11-16 18:47 PST** (Ekant's account, read-only over the group folder).

## 1. BF16 references (M2): complete

| Model | Finished / 2,505 | Excluded | FR-E2 |
|---|---|---|---|
| Qwen3-0.6B | 2,502 | 3 | pass (Qwen3 total 7,506 / 7,506) |
| Qwen3-1.7B | 2,502 | 3 | pass |
| Qwen3-4B | 2,504 | 1 | pass |
| LFM2.5-1.2B | 2,495 | 10 | pass (100%) |
| LFM2.5-2.6B | 2,504 | 1 | pass (2,504 / 2,504) |
| SmolLM3-3B | 2,458 (98.1%, under the 99% bar) | 47 | 2,455 / 2,456, decision 17 |

## 2. H1 4-bit damage cells

| Cell | Finished / 2,505 | Excluded | Pending | Status |
|---|---|---|---|---|
| Qwen3-0.6B Q4_0 | 2,501 | 4 | 0 | done (Nov 10) |
| Qwen3-0.6B Q4_K_M | 2,504 | 1 | 0 | done (Nov 10) |
| Qwen3-0.6B IQ4_XS | 2,498 | 7 | 0 | done (Nov 10) |
| Qwen3-1.7B Q4_0 | 2,382 | **123** | 0 | done (Nov 12); 4.9% excluded, see §4 |
| Qwen3-1.7B Q4_K_M | 2,496 | 9 | 0 | done (Nov 12) |
| Qwen3-1.7B IQ4_XS | 2,480 | 25 | 0 | done (Nov 12) |
| LFM2.5-1.2B Q4_0 | 2,498 | 7 | 0 | done (Nov 12) |
| LFM2.5-1.2B Q4_K_M | 2,119 | 0 | 386 (τ² retail) | **resubmitted 2026-11-16** |
| LFM2.5-1.2B IQ4_XS | 2,056 | 4 | 445 (τ² retail) | **resubmitted 2026-11-16** |
| Qwen3-4B × 3 formats | – | – | 7,515 | not submitted |
| SmolLM3-3B × 3 formats | – | – | 7,515 | not submitted |
| LFM2.5-2.6B × 3 formats | – | – | 7,515 | not submitted |

### 2026-11-16 resubmission (LFM2.5-1.2B Q4_K_M and IQ4_XS)

- **Why they stopped:** on 2026-11-12 between 21:19 and 23:15 the simulated user became unreachable
  (`ServiceUnavailableError`, `BrokenPipeError`). The runner stopped τ² scheduling as designed and left the
  episodes pending. The arrays ended on 2026-11-13 at 01:00 and nobody resubmitted them.
- **How:** fresh clone at the runs' commit `8d94524` (`/scratch/cmpe295-bertin/src/015945761/rerun-8d94524`),
  `scripts/tau2_submit.sh h1_lfm2p5_1p2b_q4_k_m h1_lfm2p5_1p2b_iq4_xs`. Planning matched the existing
  experiment hashes (`dac01ff9a0`, `52af219f58`), so only pending episodes run.
- **Jobs (Ekant's account):** simulator 120510 (A100, cs004); agents 120511 (Q4_K_M, 1 task) and 120513
  (IQ4_XS, 3 tasks); reports 120512, 120514; simulator stop 120515.
- **Next:** refresh `run_status.json` when the reports finish; check the exclusion counts.

## 3. Next phase: plan to M3 (2026-12-04)

| Item | Owner (WB2) | Target | Status |
|---|---|---|---|
| Finish LFM2.5-1.2B Q4_K_M / IQ4_XS | Ekant | 2026-11-18 | running |
| Submit H1 cells for Qwen3-4B, SmolLM3-3B, LFM2.5-2.6B | Ekant | start 2026-11-17 | not started; 2-node QoS limit (decision 14) |
| Damage preview for Qwen3-1.7B and LFM2.5-1.2B (as for 0.6B) | Vritika | after cells finish | not started |
| R1 grammars per tool-call format | Ekant | – | built (B2) |
| R2 variants (R2b layer-selective beyond 0.6B) | Yashashav | 2026-11-22 | 0.6B built (5.56 bpw); `r2b-build` jobs 112007 / 112012 stuck (DependencyNeverSatisfied) |
| QAD data (critical path) | Yashashav | – | mix manifest done |
| Label 300 pilot steps, second labeler, kappa → codebook v1 | Vritika, Ekant | – | label kit ready |
| Damage map v1 | Vritika | 2026-12-04 | not started |
| File OSF pre-registration | Vritika | after advisor decisions | draft; blocked |
| Report Ch. 1–2 draft to advisor | team | **2026-11-24** | outline done |
| Demo recording | team | 2026-12-03 | storyboard done |
| Canvas submission / Project Expo | team | 2026-12-04 / 2026-12-11 | – |

## 4. Open issues

- **Advisor decisions** 1–17 and S16-1, S16-4 are still "advisor to confirm" in `docs/m2_decisions.md`;
  Nov 13 review outcomes are not recorded in `meetings/` yet.
- **Qwen3-1.7B Q4_0: 123 exclusions** (BF16: 3), mostly empty simulated-user replies. The deck's count (59) is
  out of date. Bears on decision 16: excluding agent-caused empty replies can hide quantization damage.
- **Cluster queue (Yashashav to cancel):** `r2b-build` 112007 / 112012 will never start, so the `kl-check`
  jobs 112008, 112009, 112013, 112014 behind them won't either.
- **Throughput:** 9 cells (≈ 22,500 episodes) remain; at two GPU nodes per user, each member submitting a share
  (`shard=i/n`) is the fastest path pending decision 14.
