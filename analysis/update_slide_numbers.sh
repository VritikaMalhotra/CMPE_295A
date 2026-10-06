#!/bin/bash
# Rewrites the trailing "## Slide numbers" section of RESULTS.md from the CSVs.
# usage: update_slide_numbers.sh <progress_dir> [end HH:MM]
P=$1; END=$2
python3 - "$P" "$END" <<'PY'
import subprocess, sys
p, end = sys.argv[1], sys.argv[2]
cmd = [sys.executable, f"{p}/repo/analysis/slide_numbers.py", "--progress", p] + (["--end", end] if end else [])
block = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
path = f"{p}/RESULTS.md"; s = open(path).read()
i = s.find("\n## Slide numbers")
if i >= 0: s = s[:i]
s = s.rstrip("\n") + "\n\n## Slide numbers\n\n" + block
open(path, "w").write(s)
PY
