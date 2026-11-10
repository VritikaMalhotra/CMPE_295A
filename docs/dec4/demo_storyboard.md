# Dec 4 "Demonstrate Early Prototype to Advisor": storyboard (draft, 2026-11-09)

Canvas assignment 7852164 (group, 25 pts, graded by Dr. Cordova-Diba; media recording or file upload; due Fri Dec 4
11:59 PM). Ask Bertin on Nov 13 whether he prefers a recording or a live demo (deck slide 12, ask 4).
Proposal: one 7–9 minute screen recording that follows **one run** from its spec to a damage-map figure.
Workbook 2's prototype definition: "an end-to-end slice … or establishes a research baseline."

| # | Shot (what is on screen) | Say (key point) | Who | Time |
|---|---|---|---|---|
| 1 | Title slide; the three of us on camera | Question: where does 4-bit damage appear in a small agent's conversation | Yashashav | 0:30 |
| 2 | `configs/experiment/<damage config>.yaml` and the C11 registry listing its run specs and experiment hash | One file describes the run; every episode has a stable id | Yashashav | 1:00 |
| 3 | `sbatch` of the Slurm array; `squeue` showing the tasks; one task cut by walltime and requeued | Runs on the CoE HPC; restartable (S10) | Ekant | 1:15 |
| 4 | A step log (JSONL) for one BFCL episode: prompt hash, raw output, decoded call, execution result | Every step is recorded, not just the final score | Ekant | 1:00 |
| 5 | The same episode on BF16 vs Q4_0 side by side: the step where Q4_0 answers in text instead of calling | What "damage" looks like at one step | Ekant | 1:00 |
| 6 | C6 classifier labelling the steps; codebook v1 classes; κ from the 300-step study | How steps get their failure classes, and how we know the labels are reliable | Vritika | 1:15 |
| 7 | Damage map v1: failure rate by step type and depth, BF16 vs 4-bit, with intervals | RQ1's first answer | Vritika | 1:15 |
| 8 | Variants and KL table (BF16 → Q8_0 → Q4_K_M/IQ4_XS → Q4_0; R2 variants) | What comes next: R1 grammars and R2 bits at M4 | Yashashav | 0:45 |
| 9 | Closing: repository link, what is pre-registered, M4 date | — | all | 0:30 |

Recording: Dec 3 evening (Zoom or OBS, 1080p). Each presenter records their own shots if we can't meet; Yashashav
assembles. Keep the raw clips in the shared Drive; upload the final MP4 (or a link) to Canvas.

Checklist before recording
- [ ] Damage map v1 figure final (Vritika, Dec 2)
- [ ] One clean BF16 / Q4_0 episode pair picked for shot 5 (Ekant)
- [ ] Terminal font ≥ 16 pt, dark theme, no tokens or passwords on screen
- [ ] Every number on screen traceable to a JSON in `results/`
