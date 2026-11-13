# T36: vision-language extension (FR-O6), feasibility note

**Status: DRAFT, for Vritika (T36 owner, from the Oct 17 team meeting) to complete.** Written 2026-11-10 by
Yashashav as a starting point. Decision 10 in `docs/m2_decisions.md` proposes FR-O6 as optional 295B scope with a
go/no-go at M6. Bertin's Oct 16 suggestion was to test whether the repairs transfer to another domain, "computer
vision as a possible example". His later "use examples for all modalities" may mean this extension, or one worked
example per benchmark and domain (deck question 7).

## What the extension would ask
The same RQ1–RQ3 questions for small vision-language agents:
- Where does 4-bit damage concentrate when some inputs are images?
- Do R1–R3 still repair it?

## What we already have (checked in the source at the pinned commit)
- **Runtime.** llama.cpp d812350 has multimodal support (`tools/mtmd`) and llama-server takes a separate
  projector file. Its projector types include those of our model families' vision variants: `QWEN2VL`,
  `QWEN25VL` and `QWEN3VL` (Qwen), `LFM2` (Liquid's LFM2 vision models) and `IDEFICS3` (the SmolVLM line)
  (`tools/mtmd/clip-impl.h`).
- **Quantization.** The language model quantizes with the same formats as now. The image projector is a
  separate file, usually kept at 16 bits, so it would be a recorded factor.
- **Testbed.** C3 sends prompts to llama-server and C11 records every model file's hash, so a projector file
  would fit both. C2 (runner), C5 (logger) and C6 (classifier) need image inputs and image hashes. C4 (simulated
  user) is text-only.

## What is missing
1. **A benchmark.** BFCL and τ² are text-only. We need a multi-turn, tool-calling task set with images and an
   official checker (to survey: candidates, licences, whether a checker exists).
2. **Prompt handlers.** Each vision-language model's chat template with images, written and checked like
   `testbed/prompt_formats.py`.
3. **Compute.** Image tokens lengthen prompts. Under the 2-node QoS, every added arm competes with the main grid.

## Proposal (for the team, then Bertin)
- Keep decision 10: optional 295B scope, go/no-go at M6.
- Before M6, a two-week spike: one small vision-language model whose projector the pinned llama.cpp supports, one
  benchmark from the survey, BF16 against Q4_0 on about 50 tasks, using the same step classes where they apply.
- Go if the benchmark has an official checker and the BF16 reference solves enough tasks to compare.

## To verify before this note is final
- [ ] Which vision-language checkpoints in the 0.6B–4B range exist for each family, and their licences.
- [ ] Candidate multimodal tool-use benchmarks, with their checkers and licences.
- [ ] llama-server's handling of images through the raw `/completion` endpoint that C3 uses.
