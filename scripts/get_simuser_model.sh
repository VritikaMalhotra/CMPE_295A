#!/usr/bin/env bash
# C4: fetch the simulated-user model pinned in configs/experiment/skeleton.yaml and check its SHA-256.
#   scripts/get_simuser_model.sh <models_dir> [cache_dir]
# With a cache dir (e.g. a Drive folder) the file is copied from there when present, and stored there after
# the first download, so later sessions skip the 12 GB download.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?models dir}; CACHE=${2:-}
PY=$ROOT/.venv/bin/python
cd "$ROOT"
read -r REPO REV FILE SHA < <("$PY" - <<'PY'
import yaml
e = yaml.safe_load(open("configs/experiment/skeleton.yaml"))["experiment"]
s = next(b["simuser"] for b in e["extra_benchmarks"] if "simuser" in b)
print(s["hf_repo"], s["hf_revision"], s["file"], s["sha256"])
PY
)
mkdir -p "$OUT"
DST=$OUT/$FILE
if [ ! -f "$DST" ] && [ -n "$CACHE" ] && [ -f "$CACHE/$FILE" ]; then
  echo "copying $FILE from cache"; cp "$CACHE/$FILE" "$DST.part" && mv "$DST.part" "$DST"
fi
if [ ! -f "$DST" ]; then
  "$PY" -c "import huggingface_hub" 2>/dev/null || env -u UV_SYSTEM_PYTHON uv pip install -q --python "$PY" huggingface_hub
  "$PY" - "$REPO" "$REV" "$FILE" "$OUT" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, rev, f, out = sys.argv[1:]
print(hf_hub_download(repo, f, revision=rev, local_dir=out))
PY
fi
GOT=$(sha256sum "$DST" | cut -d' ' -f1)
if [ "$GOT" != "$SHA" ]; then echo "SHA-256 mismatch for $FILE: $GOT (expected $SHA)"; exit 1; fi
echo "$GOT  $FILE" > "$DST.sha256"
echo "ok: $FILE $GOT"
if [ -n "$CACHE" ] && [ ! -f "$CACHE/$FILE" ]; then
  mkdir -p "$CACHE"; cp "$DST" "$CACHE/$FILE.part" && mv "$CACHE/$FILE.part" "$CACHE/$FILE"
fi
