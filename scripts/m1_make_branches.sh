#!/usr/bin/env bash
# Build the M1 review branches from a drop folder, in your clone, under your git identity. Nothing is pushed.
#   bash m1_make_branches.sh <path-to-clone> <path-to-drop-folder>
# Branches (all from origin/main; merge in this order):
# Ownership follows Workbook v3 Table 40 and the Sprint 4 row of Table 39.
#   m1/interfaces              Yashashav  shared contracts (spec.py), pyproject, test scaffolding
#   m1/vritika-c1-c6-c10       Vritika    C1 adapter, non-thinking Qwen handler, C6 classifier, C10 toy stats
#                                         (drafted for her review)
#   m1/ekant-c2-c3-c5-c9       Ekant      C2 runner, C3 backends, C5 logger, C9 profiler, R1 grammar test
#                                         (drafted for his review; builds on Vritika's branch: C3 uses her handler)
#   m1/c11-registry-runner     Yashashav  C11 registry/run/report, configs, Slurm, Colab notebook
#   m1/repairs-r2-r3           Yashashav  R2 Q8_0 variants, R3 QAD data + export
#   m1/skeleton                all of the above merged: the branch the Colab run uses until reviews land
set -euo pipefail
REPO=$(cd "$1" && pwd); DROP=$(cd "$2" && pwd)
cd "$REPO"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "clone has uncommitted changes; stash or commit first"; exit 1; }
git fetch origin
for b in m1/interfaces m1/vritika-c1-c6-c10 m1/ekant-c2-c3-c5-c9 m1/c11-registry-runner m1/repairs-r2-r3 m1/skeleton; do
  git show-ref --verify --quiet "refs/heads/$b" && { echo "branch $b already exists; delete it or rename before re-running"; exit 1; }
done
TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_016i9tdctb7EVYczN4Wjz7p5'

put() {  # copy files from the drop into the work tree and stage them
  for f in "$@"; do mkdir -p "$(dirname "$f")"; cp "$DROP/$f" "$f"; git add "$f"; done
}

branch() {  # name, base, message, files...
  local name=$1 base=$2 msg=$3; shift 3
  git checkout -q -b "$name" "$base"
  put "$@"
  git commit -q -m "$msg$TRAILER"
  echo "created $name: $(git log -1 --oneline)"
}

branch m1/interfaces origin/main "M1 interfaces: RunSpec/StepRow/EpisodeResult contracts, pyproject

Frozen interfaces every M1 component codes against (Workbook Ch. 6): content-hashed RunSpec
(run_id, config_key), GenerateResult, StepRow, EpisodeResult. Changes by PR reviewed by all three." \
  .gitignore pyproject.toml testbed/__init__.py testbed/spec.py tests/conftest.py

branch m1/vritika-c1-c6-c10 m1/interfaces "C1 adapter, Qwen non-thinking handler, C6 classifier rules v0, C10 stats (draft for Vritika's review)

Drafted by Yashashav so the M1 skeleton could run end to end; Vritika owns these components
(Workbook Table 40) and should review, change, and merge.
- C1: BFCL multi-turn base tasks + tool schemas; first 20 ids = the M1 task set
- Non-thinking handler (Sprint 4): qwen_format ports bfcl-eval 2026.3.23's QwenFC prompt;
  byte-identical to bfcl's formatter (parity test on the C11 branch)
- C6: one failure class per step (10 classes, first rule wins; forced_quit before call-content rules)
- C10: exact McNemar, task-clustered paired bootstrap, planted-effect check (M1 exit criterion)
C4 (tau2 airline) moved to M2 with the Colab decision of 2026-10-24." \
  testbed/adapter.py testbed/qwen_format.py testbed/classifier.py testbed/stats.py tests/test_classifier.py \
  tests/test_stats.py tests/test_qwen_format.py

branch m1/ekant-c2-c3-c5-c9 m1/vritika-c1-c6-c10 "C2 runner, C3 backends, C5 logger, C9 profiler, R1 test (draft for Ekant's review)

Drafted by Yashashav so the M1 skeleton could run end to end; Ekant owns these components
(Workbook Table 40) and should review, change, and merge.
- C2: bfcl's multi-turn loop with a unique model_name per episode (bfcl keeps state in globals);
  scored like bfcl's multi_turn_runner (force_terminated only on a short turn count); malformed
  call blocks -> inference_error as in bfcl; context overflow -> scored failure; tool names outside
  the task's list are never eval-ed
- C3: llama-server (CUDA/CPU) behind generate(); /completion, cache_prompt=false, per-request seed,
  bfcl's token budget min(4096, ctx - prompt - 2) with ctx 40960
- C5: atomic per-episode JSONL -> Parquet
- C9: NVML energy delta + peak memory per episode, TTFT and tok/s from server timings
- R1: test for the existing free-text-branch grammar (testbed/grammar.py)
Built on m1/vritika-c1-c6-c10 because C3 formats prompts with her Qwen handler and C2 loads
tasks with her C1 adapter: merge hers first." \
  testbed/runner.py testbed/backends.py testbed/logger.py testbed/profiler.py tests/test_r1_grammar.py

git checkout -q -b m1/c11-registry-runner m1/interfaces
git merge -q --no-edit m1/ekant-c2-c3-c5-c9 -m "Merge drafted components for C11 integration$TRAILER"
put testbed/registry.py testbed/run.py testbed/report.py configs/config.yaml configs/experiment/skeleton.yaml \
    configs/experiment/smoke.yaml slurm/array.sbatch.tmpl slurm/report.sbatch.tmpl scripts/setup_env.sh \
    notebooks/m1_skeleton_colab.ipynb tests/test_registry.py tests/test_bfcl_parity.py
git commit -q -m "C11 config & registry: one command for the M1 experiment

python -m testbed.run +experiment=skeleton [dry_run|local=true|(slurm)]
- expands the Hydra experiment into 220 content-hashed RunSpecs, rejects duplicates
- experiment dir keyed by the spec-list hash; resumable (done iff episodes/<run_id>.json exists)
- manifest: model SHA-256 vs the Oct 8 pilot, llama.cpp + bfcl-eval versions, git commit, GPU
- one llama-server per configuration; Slurm array writer (one task per config) for M2
- report.md: accuracy, failure mix, paired comparisons, costs, planted effect, manifest
- per-config flock + done-recheck under the lock; a failed server start skips only its config
- refuses to resume into a directory whose model files changed (model_hashes.json)
- Slurm: array tasks never rewrite shared files; a dependent job writes the report
- test_bfcl_parity: prompt byte-identical to bfcl's QwenFC formatter; ground-truth replay scores
  valid through bfcl's checker; forced-quit, malformed, overflow and unknown-name cases
- Colab notebook (A100) and a CPU-only smoke experiment$TRAILER"
echo "created m1/c11-registry-runner: $(git log -1 --oneline)"

branch m1/repairs-r2-r3 m1/interfaces "R2 Q8_0 variant + R3 toy-QAD data and GGUF export with fidelity and KL checks

- make_variants.sh: verifies the pilot GGUFs' SHA-256, builds Q8_0 from the same BF16
- make_qad_data.py: fixed WikiText-2 sequences (S7's seqs.pt was not kept), hash recorded
- qad_toy.py: the S7 spike with a llama.cpp-exact Q4_0 fake quantizer (S7 used torch.round,
  half-to-even; llama.cpp rounds half up) and a --model argument
- export_qad.py: FQLinear checkpoint -> HF -> BF16 GGUF -> --pure Q4_0; checks dequantized
  weights equal q4_0_fake and KL(BF16 || QAD) <= KL(BF16 || plain Q4_0)
- test_q4_0.py: bit-exact check against llama.cpp's gguf-py Q4_0" \
  scripts/make_variants.sh train/__init__.py train/make_qad_data.py train/qad_toy.py train/export_qad.py tests/test_q4_0.py

git checkout -q -b m1/skeleton m1/c11-registry-runner
git merge -q --no-edit m1/repairs-r2-r3 -m "M1 skeleton: all components merged for the Colab run$TRAILER"
put scripts/m1_make_branches.sh
git commit -q -m "Add the script that built the M1 review branches$TRAILER"
echo "created m1/skeleton: $(git log -1 --oneline)"
git checkout -q m1/skeleton
echo
echo "Done. Review with: git log --oneline --graph --all -15"
echo "Push when ready:   git push -u origin m1/interfaces m1/vritika-c1-c6-c10 m1/ekant-c2-c3-c5-c9 m1/c11-registry-runner m1/repairs-r2-r3 m1/skeleton"
