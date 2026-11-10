# M3 / R2 model variants: runbook (Yashashav)

Workbook 2 work packages this covers, all owned by Yashashav: **J1 R2 variant construction** (Nov 9–22),
the **IQ4_XS** part of the **M3 exit** ("Q4_0 (--pure), Q4_K_M and IQ4_XS (imatrix) of every model, each
KL-checked against BF16 on both backends"), and the open Table 37a item **Singularity image**. It does not touch
Ekant's work this sprint (H1 damage runs, R1 grammars, second labeler) or Vritika's (labels, κ, damage map).
QoS `gpu3nodes` limits each *user* to 2 GPU nodes, so these jobs run from Yashashav's account and do not take
nodes from Ekant's damage runs.

Status 2026-11-09 (evening): steps 0–3 and 6 have run on the cluster. Results are in `docs/m3_results.md`: all
six models have IQ4_XS, Q5_0-pure and Q6_K-pure files, and KL against BF16 on both backends. The QAD mix is
built (`docs/qad_data.md`). Steps 4–5 (R2b) and 7 (container) are still to do. Earlier note: The llama.cpp behaviour they rely on was checked in the source at the pinned commit d812350
(see "Why these commands" below). Run step 0 before anything large.

## Step 0 — checks (login node, ~10 min)
```bash
source scripts/hpc_env.sh
.venv/bin/python -m pytest -q tests/test_m3_quant.py                     # 10 tests
apptainer --version; apptainer buildcfg | grep -i -E "fakeroot|userns"  # can we build without root? (step 7)
ls $LLAMA_DIR/build-cuda/bin/llama-imatrix $LLAMA_DIR/build-cpu/bin/llama-perplexity
```
Then the `--tensor-type` probe that steps 4–5 rest on (CPU job, ~5 min): exactly one tensor must come out Q4_0,
every other quantizable tensor BF16.
```bash
srun -A coe -p cpuqs -c 8 --mem=16G -t 00:20:00 $LLAMA_DIR/build-cpu/bin/llama-quantize --pure \
    --tensor-type '^blk\.0\.attn_q\.weight$=q4_0' --tensor-type '.=bf16' \
    $TESTBED_MODELS/Qwen3-0.6B-BF16.gguf $TESTBED_OUT/probe_tensor_type.gguf Q4_0
.venv/bin/python - $TESTBED_OUT/probe_tensor_type.gguf <<'EOF'
import collections, os, sys
sys.path.insert(0, os.path.join(os.environ["LLAMA_DIR"], "gguf-py"))
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
print(collections.Counter(t.tensor_type.name for t in r.tensors))
print([t.name for t in r.tensors if t.tensor_type.name == "Q4_0"])   # expect ['blk.0.attn_q.weight']
EOF
rm $TESTBED_OUT/probe_tensor_type.gguf
```
The CPU KL check of the existing Q4_0-pure and Q4_K_M files (step 3, second line, with only those two variants)
needs nothing from steps 1–2, so submit it for the core models right after step 0.

Also the review follow-up on Ekant's runner change (`slurm/array.sbatch.tmpl` quoting): once `m2-phase1` is merged
into `m2`, rerun `configs/experiment/s10_requeue.yaml` and `scripts/check_s10.py` (104/104, no duplicates).

## Step 1 — calibration text that shares no 13-gram with the benchmarks (FR-E5), login node
```bash
bash scripts/get_eval_text.sh       # WikiText-2 raw; the zip also has wiki.train.raw
.venv/bin/python scripts/benchmark_ngrams.py dump --out $CMPE295_GROUP/data/calib/benchmark_text.txt
.venv/bin/python scripts/benchmark_ngrams.py clean --bench $CMPE295_GROUP/data/calib/benchmark_text.txt \
    --calib $CMPE295_GROUP/data/eval/wikitext-2-raw/wiki.train.raw \
    --out $CMPE295_GROUP/data/calib/calib.txt --report results/m3/calib_ngram_check.json
```
`calib_ngram_check.json` must say `"pass": true`. The KL eval text stays WikiText-2 **test** (as in M2), so the
importance matrix never sees the text it is judged on.

## Step 2 — IQ4_XS (imatrix), Q5_0-pure, Q6_K-pure for all six models (GPU, one job per model)
```bash
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B LFM2.5-1.2B-Instruct SmolLM3-3B LFM2.5-2.6B; do
  sbatch slurm/m3_variants.sbatch $m          # a P100 holds every model at BF16 with a 512-token context (M2 KL jobs)
done
```
Smoke first: run only `Qwen3-0.6B`, check `$TESTBED_MODELS/Qwen3-0.6B-IQ4_XS.gguf.log` and the sizes, then the rest.

## Step 3 — KL against BF16 on both backends
```bash
for m in <six models>; do
  sbatch slurm/kl_check.sbatch     $m IQ4_XS Q5_0-pure Q6_K-pure                    # CUDA, new files only
  sbatch slurm/kl_check_cpu.sbatch $m Q4_0-pure Q4_K_M IQ4_XS Q5_0-pure Q6_K-pure Q8_0   # CPU, all
done
```
Expected order of mean KLD: Q8_0 < Q6_K < Q5_0 < Q4_K_M ≈ IQ4_XS < Q4_0-pure. Copy each
`kl_<model>.json` to `results/m3/kl_cuda_<model>.json` / `results/m3/kl_cpu_<model>.json`.

## Step 4 — R2b sensitivity ranking (GPU array), core models first
Core models (Workbook 2 Table 13): Qwen3-0.6B, Qwen3-1.7B, LFM2.5-1.2B-Instruct; then Qwen3-4B, SmolLM3-3B, LFM2.5-2.6B.
```bash
.venv/bin/python scripts/rank_tensors.py list --model Qwen3-0.6B         # login node: candidates
sbatch slurm/rank_tensors.sbatch Qwen3-0.6B base                          # BF16 logits on calib.txt
sbatch --dependency=afterok:<base_job> --array=0-7%2 slurm/rank_tensors.sbatch Qwen3-0.6B run
```
One score file per linear tensor in `$TESTBED_OUT/rank/<model>/scores/`. Rough cost: 0.6B has 196 candidate
tensors; each is one BF16-sized write plus a 20-chunk KL pass on a P100.

## Step 5 — build R2b and check FR-E10 (5.5 ± 0.1 bits per weight)
```bash
.venv/bin/python scripts/rank_tensors.py select --model Qwen3-0.6B --bpw 5.5
bash scripts/make_layer_selective.sh Qwen3-0.6B     # writes <model>-R2b-LS5.5.gguf + .check.json; exit 1 if off budget
```
Then KL-check it like step 3 (`R2b-LS5.5`). Copy `selection.json` and the `.check.json` to `results/m3/`.

## Step 6 — record
```bash
.venv/bin/python scripts/models_manifest.py --models-dir $TESTBED_MODELS \
    --base results/m2/models_manifest.json --out results/m3/models_manifest.json
```
Write the M3 results to `docs/m3_results.md` (same style as the M2 section of `docs/cluster.md`: job ids,
settings, table, JSON paths). `docs/cluster.md` gets at most a one-line pointer, added when the branches merge.

## Step 7 — container (Table 37a), two stages because compute nodes have no network
```bash
mkdir -p $CMPE295_GROUP/containers
apptainer build --fakeroot $CMPE295_GROUP/containers/testbed-base.sif apptainer/base.def   # login node: downloads only
sbatch slurm/build_apptainer.sbatch                                                        # compiles llama.cpp offline
apptainer exec --nv -B $PWD -B $CMPE295_GROUP $CMPE295_GROUP/containers/testbed.sif python -m pytest -q tests
```
Needs Apptainer ≥ 1.2 (`--build-arg` templating) and fakeroot. If fakeroot is not enabled, ask the
administrator, or build both stages elsewhere and copy the `.sif`.

## Choices to confirm (Yashashav; raise with the team if unsure)
1. **Q5_0 and Q6_K built with `--pure`**, like the Q4_0 control, so they sit at the nominal 5.5 / 6.56 bits per
   weight that Workbook 2 Table 19 lists and R2b's 5.5-bit budget compares like with like. The existing Q8_0 used
   llama.cpp's default recipe (8.5 bpw either way).
2. **IQ4_XS with llama.cpp's default recipe** plus the imatrix (as llama.cpp ships IQ4_XS), not `--pure`.
3. **Importance-matrix text for IQ4_XS = WikiText-2 train minus any line sharing a 13-gram with the benchmarks**
   (confirmed by Yashashav 2026-11-09). It is fixed before the IQ4_XS files are built, because the M3 damage runs
   use them; changing it later means rebuilding the files and rerunning that arm. The R2b ranking uses
   "calibration conversations" (Workbook 2 Ch. 3): a held-out slice of the function-calling corpus chosen for the
   QAD training mix, filtered the same way, passed in with `RANK_TEXT`. The corpus is still to be chosen, so
   step 4 waits for it.
4. **Ranking settings:** 20 chunks × 512 tokens (`RANK_CHUNKS`, `RANK_CTX`). Token embedding and output head stay at
   Q4_0 in R2b, as in the Q4_0 control; only 2-D block weights are candidates.

## Why these commands (llama.cpp d812350)
- `llama-quantize` applies `--tensor-type` overrides only when the base type is quantized, in command-line order,
  first match wins (`src/llama-quant.cpp`, `llama_tensor_get_type`). So "one tensor at Q4_0, everything else BF16" is
  `--pure --tensor-type '^name$=q4_0' --tensor-type '.=bf16' … Q4_0`, and R2b is `--pure --tensor-type-file
  tensor_types.txt … Q4_0` with `^name$=q8_0` lines.
- Tensors that llama.cpp never quantizes (norms; some conv weights) keep their type whatever the override says; the
  measured bits per weight in `.check.json` is the number that counts, not the prediction.
- `llama-imatrix -o <file>.gguf --chunks N` writes the GGUF imatrix format that `llama-quantize --imatrix` reads.
