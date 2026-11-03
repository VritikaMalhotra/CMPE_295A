#!/usr/bin/env bash
# Fetch the KL/perplexity eval text on the login node (compute nodes have no network): WikiText-2 raw, the
# corpus llama.cpp's perplexity tool is usually run on, via the pinned tree's own download script.
#   source scripts/hpc_env.sh && bash scripts/get_eval_text.sh
# Writes $CMPE295_GROUP/data/eval/wikitext-2-raw/wiki.test.raw and its .sha256. M3 adds logged agent prompts.
set -euo pipefail
G=${CMPE295_GROUP:?source scripts/hpc_env.sh first}
SRC=${LLAMA_DIR:?}
D=$G/data/eval
mkdir -p "$D"
cd "$D"
if [ ! -f wikitext-2-raw/wiki.test.raw ]; then
  bash "$SRC/scripts/get-wikitext-2.sh"
fi
sha256sum wikitext-2-raw/wiki.test.raw | tee wikitext-2-raw/wiki.test.raw.sha256
wc -c wikitext-2-raw/wiki.test.raw
