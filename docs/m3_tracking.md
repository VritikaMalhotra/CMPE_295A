# M3 tracking: run results and next phase

Living tracker for the damage-map phase (M3, due 2026-12-04). Counts come from `scripts/run_status.py` run on
the cluster against `/scratch/cmpe295-bertin/runs/` (`results/m3/run_status.json`), never typed in.
Development data only: the registered test runs after the pre-registration is filed.

Last refresh: **2026-11-18T22:27-08:00** (Ekant's account, read-only over the group folder).

## 1. BF16 references (M2): complete

| Model | Finished / 2,505 | Excluded | FR-E2 |
|---|---|---|---|
| Qwen3-0.6B | 2,502 | 3 | pass (Qwen3 total 7,506 / 7,506) |
| Qwen3-1.7B | 2,502 | 3 | pass |
| Qwen3-4B | 2,504 | 1 | pass |
| LFM2.5-1.2B | 2,495 | 10 | pass, 2,494 / 2,494 (rerun 2026-11-16) |
| LFM2.5-2.6B | 2,504 | 1 | pass, 2,504 / 2,504 (rerun 2026-11-16) |
| SmolLM3-3B | 2,458 (98.1%, under the 99% bar) | 47 | pass, 2,456 / 2,456 after decision 17 (2026-11-16) |

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
| Qwen3-4B × 3 formats | 0 | 0 | 7,515 | **submitted 2026-11-16** |
| SmolLM3-3B × 3 formats | 0 | 0 | 7,515 | **submitted 2026-11-16** |
| LFM2.5-2.6B × 3 formats | 0 | 0 | 7,515 | **submitted 2026-11-16** |

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

### 2026-11-16 submission: the 9 remaining H1 cells

- **Configs:** `configs/experiment/h1_{qwen3_4b,smollm3_3b,lfm2p5_2p6b}_{q4_0,q4_k_m,iq4_xs}.yaml`, derived from each
  BF16 reference exactly as the existing H1 configs (only name, variant and model file differ).
- **How:** clone of `m3-damage-runs` at `7476c23` (`/scratch/cmpe295-bertin/src/015945761/m3-damage-runs`; runner
  code identical to `8d94524`), `scripts/tau2_attach.sh` onto simulator 120510, so all 11 arrays share one
  simulated user and stay within the 2-node QoS.
- **Jobs (Ekant's account), agents / report:**

| Cell | Run dir | Agents | Report |
|---|---|---|---|
| Qwen3-4B Q4_0 | `h1_qwen3_4b_q4_0/12141ac1a9` | 120547 | 120548 |
| Qwen3-4B Q4_K_M | `h1_qwen3_4b_q4_k_m/828a67648f` | 120549 | 120550 |
| Qwen3-4B IQ4_XS | `h1_qwen3_4b_iq4_xs/494b882fbb` | 120551 | 120552 |
| SmolLM3-3B Q4_0 | `h1_smollm3_3b_q4_0/6002a19027` | 120553 | 120554 |
| SmolLM3-3B Q4_K_M | `h1_smollm3_3b_q4_k_m/4d7c2962b3` | 120555 | 120556 |
| SmolLM3-3B IQ4_XS | `h1_smollm3_3b_iq4_xs/f27453f6dd` | 120557 | 120558 |
| LFM2.5-2.6B Q4_0 | `h1_lfm2p5_2p6b_q4_0/3aa6ba0c98` | 120559 | 120560 |
| LFM2.5-2.6B Q4_K_M | `h1_lfm2p5_2p6b_q4_k_m/01063262a7` | 120561 | 120562 |
| LFM2.5-2.6B IQ4_XS | `h1_lfm2p5_2p6b_iq4_xs/dbae64a12c` | 120563 | 120564 |

- **Simulator stop:** 120515 → 120566 → 121863 → **122012**, which waits for all 11 arrays.
- **2026-11-16 22:15 restart.** No agent task had started (cs003 and cs004 full; cs001 and cs002 excluded). The
  administrator reported cs001 fixed and cs002 usable without GPU 0. cs001 GPUs 0–2 passed a llama.cpp smoke test
  (jobs 121849–121851), so every pending array now excludes only cs002 (Slurm still hands out cs002's GPU 0).
  The first LFM2.5-1.2B tasks on cs001 then failed: the simulated-user SQLite caches in Yash's run directories were
  `rw-r-----` (written via his `gpusafe` wrappers, not `scripts/hpc_env.sh`, which sets `umask 0007`), so the cache
  layer crashed and the agent saw "Connection error". Fix: the 12 cache files replaced by group-writable copies
  (row counts checked; originals kept in `simuser_cache_orig_017856371/`); the 8 one-attempt retry records and
  2 partial step files from the failed tasks removed, so no episode was excluded. Resubmitted: Q4_K_M 121861
  (report 121862), IQ4_XS 122010 (report 122011); simulator stop now 122012 (waits for all 11 arrays).
- **2026-11-17 08:29: simulator crash.** After 13.5 h, 120510's llama-server segfaulted (exit -11). No retry records
  were written after the crash, so no episode was excluded. Recovery (08:40–08:55): every pending τ² array task held
  (tasks 6–11 of the 12-task arrays, 121861, 122010), a new simulator **130139** started at the same endpoint path,
  pending BFCL tasks set to `Nice=1000` so τ² uses the simulator first, held tasks released. SmolLM3 Q4_K_M task 6
  (idle, waiting for an endpoint) cancelled and resubmitted as 130146. Cells whose arrays had ended with τ² episodes
  pending were attached to 130139: Qwen3-4B Q4_0 (2 pending) 130161, Q4_K_M (1) 130163, IQ4_XS (36) 130165,
  SmolLM3 Q4_0 (622) 130167. Simulator stop now **130169** (scancels 130139 after all arrays).
- **Progress 2026-11-17 08:38:** Qwen3-4B Q4_0 2,503 / Q4_K_M 2,501 / IQ4_XS 2,464 finished; SmolLM3 Q4_0 1,860,
  Q4_K_M 485; SmolLM3 IQ4_XS and all LFM2.5-2.6B cells not started; LFM2.5-1.2B Q4_K_M 2,119 / IQ4_XS 2,111.
- **2026-11-17 18:42: second simulator crash.** 130139's llama-server segfaulted (exit -11) after 10 h, like 120510
  after 13.5 h. No retry records after the crash; no episode excluded. By 2026-11-18 12:12 every BFCL episode of all
  18 H1 cells was finished; 2,707 τ² episodes were pending in 11 cells (`results/m3/run_status.json`, 12:14).
  Resubmitted with `scripts/tau2_submit.sh` (all 11 experiments, existing run directories): simulator **149848**
  (cs001), arrays 149849–149869 (odd), reports 149850–149870 (even), stop 149871.
- **2026-11-18: simulated-user auto-restart** (commit 8a9b02e). `slurm/simuser.sbatch` restarts a crashed service on a
  new port and rewrites the endpoint; agents (`SimStack.reconnect`) wait for it, switch over and re-run the
  interrupted episode (at most 3 times; an outage is never an attempt). Rolled out by cancelling the 12:14 batch
  (simulator 149848 crashed after 2 min) and resubmitting all 11 cells: simulator 149942, arrays 149943–149963 (odd),
  reports 149944–149964 (even). In production: 149942 crashed at 12:22:10 and served again at 12:22:35; agents
  reconnected. Both quick crashes were on cs001 GPU 0, so the simulator moved to **150003 on cs003** (same endpoint;
  agents reconnected); simulator jobs now exclude cs001. By 12:45: 0 agents stopped τ² scheduling, 13 reconnects.
  Simulator stop **150067** (cancels 150003 after all 11 arrays). Leftovers after this batch (e.g. Qwen3-4B Q4_0,
  1 episode with 2 HTTP-500 attempts) need one more `tau2_submit.sh` pass.
- **2026-11-18 22:25: grid nearly complete.** The 12:14 batch finished all but **23 τ² episodes** in 6 cells (each with
  1–2 recorded HTTP-500 / decode attempts, decision 15). Simulator 150003 crashed 21 times in 9 h on cs003 and stopped
  at the 20-restart cap: the crashes follow load, not cs001. 87 agent reconnects, 4 stops (after the simulator gave
  up). All 34 exclusions in the batch were empty simulated-user replies (decision 16). Restart cap raised to 200
  (9034904). Final pass: simulator **153714**, arrays 153715–153725 (odd), stop 153727.
- **2026-11-19 00:05: final pass left 19 τ² episodes.** It finished 4 of the 23 (now pending: SmolLM3 Q4_0 10,
  Q4_K_M 3, IQ4_XS 3; LFM2.5-2.6B Q4_0 1, Q4_K_M 1; Qwen3-4B Q4_0 1; 16 of 19 are telecom). Simulator 153714
  segfaulted (-11) **16 times in 26 min** with only ~6 agents. Several crashes came within a minute of a restart
  with no request logged, so the same episodes appear to crash it every time: seeded agents replay the same
  conversation to the same simulator request. The agents gave up after 3 reconnects per episode (6 stops,
  36 reconnects). The log also shows 56 HTTP-500 peg-native parse failures (decision 15 attempts). 
- **2026-11-19 00:40: decision 18.** The crash reproduced with a single-slot simulator and no prompt cache (154012:
  3 crashes in 4 min, each right after a malformed user-tool call returned HTTP 500), so it is request-specific.
  Applied `scripts/exclude_sim_crash.py`: **11 episodes excluded as SimulatorCrash** (at least 3 outages each in the
  agent logs). The other **8** (SmolLM3 Q4_0 3, Q4_K_M 3, IQ4_XS 2; 3 of 4 attempts used, 0–2 outages) get one more
  ordinary attempt: simulator **154052**, agents 154053/154055/154057, stop 154059. Reports rerun for the 3 now
  complete cells (154066–154068). Advisor to confirm decision 18.
- **2026-11-19 00:22: H1 grid complete (18/18 cells, 0 pending).** The final ordinary attempt (154052, then 154087
  for one episode on 3 attempts) ran with no simulator crashes; all 8 episodes failed a 4th time with HTTP 500
  (decision 15) and are excluded under the registered rule. Final SmolLM3 exclusions: Q4_0 47, Q4_K_M 34,
  IQ4_XS 40. Reports rerun for every touched cell. Next: damage previews for damage map v1 (Dec 4).
- **Risk:** gpt-oss llama-server crashes (13.5 h and 10 h on cs004; minutes on cs001 GPU 0). Now handled by the
  restart loop; if the simulator job itself ends (200 restarts or walltime), rerun the same `tau2_submit.sh` command. If it ends first, τ² episodes stay
  pending (resumable): resubmit with `scripts/tau2_submit.sh <experiments>` from the same clone.

## 3. Next phase: plan to M3 (2026-12-04)

| Item | Owner (WB2) | Target | Status |
|---|---|---|---|
| Finish LFM2.5-1.2B Q4_K_M / IQ4_XS | Ekant | 2026-11-18 | queued (121861, 122010) |
| Run H1 cells for Qwen3-4B, SmolLM3-3B, LFM2.5-2.6B | Ekant | ~2026-11-20 | submitted 2026-11-16 (120547–120564) |
| Damage preview for Qwen3-1.7B and LFM2.5-1.2B (as for 0.6B) | Vritika | after cells finish | not started |
| R1 grammars per tool-call format | Ekant | – | built (B2) |
| R2 variants (R2b layer-selective beyond 0.6B) | Yashashav | 2026-11-22 | 0.6B built (5.56 bpw); `r2b-build` jobs 112007 / 112012 stuck (DependencyNeverSatisfied) |
| QAD data (critical path) | Yashashav | – | mix manifest done |
| Label 300 pilot steps, second labeler, kappa → codebook v1 | Vritika, Ekant | – | label kit ready |
| Damage map v1 | Vritika | 2026-12-04 | not started |
| File OSF pre-registration | Vritika | this week | unblocked: decisions confirmed |
| Report Ch. 1–2 draft to advisor | team | **2026-11-24** | outline done |
| Demo recording | team | 2026-12-03 | storyboard done |
| Canvas submission / Project Expo | team | 2026-12-04 / 2026-12-11 | – |

## 4. Open issues

- **Advisor decisions** 1–17, S16-1 and S16-4: **confirmed as proposed** (recorded 2026-11-16 in
  `docs/m2_decisions.md`). Decision 17 done 2026-11-16 (replay aligned, FR-E2 now passes for all six models).
  Follow-ups: decision 13 (CPU arm on the fixed subset); Nov 13 meeting notes still to be written in `meetings/`.
- **Qwen3-1.7B Q4_0: 123 exclusions** (BF16: 3), mostly empty simulated-user replies. The deck's count (59) is
  out of date. Bears on decision 16: excluding agent-caused empty replies can hide quantization damage.
- **Cluster queue (Yashashav to cancel):** `r2b-build` 112007 / 112012 will never start, so the `kl-check`
  jobs 112008, 112009, 112013, 112014 behind them won't either.
- **Throughput:** 9 cells (≈ 22,500 episodes) remain; at two GPU nodes per user, each member submitting a share
  (`shard=i/n`) is the fastest path pending decision 14.
