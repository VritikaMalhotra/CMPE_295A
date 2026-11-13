#!/bin/bash
# Why is SmolLM3-3B's Q5_0-pure worse than its Q4_K_M (docs/m3_results.md, observation 1)? SmolLM3 ties its
# output head to token_embd. Q4_K_M keeps that tensor at Q6_K and raises 36 more tensors to Q6_K; --pure puts
# all of them at Q5_0. Three diagnostic files, each changing one thing, then KL against BF16 on the CPU backend
# with the same settings as step 3 (WikiText-2 test, ctx 512, 50 chunks):
#   Q5_0-pure-embdQ6K : Q5_0-pure, but token_embd (= the output head) at Q6_K
#   Q5_0-pure-kmQ6K   : Q5_0-pure, but every tensor Q4_K_M has at Q6_K (embd + 36) at Q6_K
#   Q4_K_M-embdQ5_0   : Q4_K_M, but token_embd at Q5_0
# Files go to a separate folder ($CMPE295_GROUP/models/diag_b4), never next to the study files.
#   sbatch slurm/b4_smollm3_diag.sbatch
#SBATCH -J b4-smollm3-diag
#SBATCH -A coe
#SBATCH -p cpuqs
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
NAME=SmolLM3-3B
D=$CMPE295_GROUP/models/diag_b4
mkdir -p "$D"
ln -sf "$TESTBED_MODELS/$NAME-BF16.gguf" "$D/$NAME-BF16.gguf"
Q=$LLAMA_DIR/build-cpu/bin/llama-quantize
T=${SLURM_CPUS_PER_TASK:-32}
# the tensors Q4_K_M stores as Q6_K, read from the study file
.venv/bin/python - "$TESTBED_MODELS/$NAME-Q4_K_M.gguf" "$LLAMA_DIR/gguf-py" > "$D/q4km_q6k_tensors.txt" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
for t in GGUFReader(sys.argv[1]).tensors:
    if t.tensor_type.name == "Q6_K":
        print(f"^{t.name.replace('.', chr(92) + '.')}$=q6_k")
PY
echo "Q4_K_M Q6_K tensors: $(wc -l < "$D/q4km_q6k_tensors.txt")"
B=$D/$NAME-BF16.gguf
[ -f "$D/$NAME-Q5_0-pure-embdQ6K.gguf" ] || "$Q" --pure --token-embedding-type q6_k "$B" "$D/$NAME-Q5_0-pure-embdQ6K.gguf" Q5_0 "$T" > "$D/Q5_0-pure-embdQ6K.log" 2>&1
[ -f "$D/$NAME-Q5_0-pure-kmQ6K.gguf" ] || "$Q" --pure --tensor-type-file "$D/q4km_q6k_tensors.txt" "$B" "$D/$NAME-Q5_0-pure-kmQ6K.gguf" Q5_0 "$T" > "$D/Q5_0-pure-kmQ6K.log" 2>&1
[ -f "$D/$NAME-Q4_K_M-embdQ5_0.gguf" ] || "$Q" --token-embedding-type q5_0 "$B" "$D/$NAME-Q4_K_M-embdQ5_0.gguf" Q4_K_M "$T" > "$D/Q4_K_M-embdQ5_0.log" 2>&1
export TESTBED_MODELS=$D LLAMA_PERPLEXITY=$LLAMA_DIR/build-cpu/bin/llama-perplexity KL_NGL=0
export KL_OUT=$TESTBED_OUT/kl_cpu_diag_b4/$NAME TESTBED_CPU_THREADS=$T
bash scripts/kl_check.sh "$NAME" Q5_0-pure-embdQ6K Q5_0-pure-kmQ6K Q4_K_M-embdQ5_0
