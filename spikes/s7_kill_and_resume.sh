#!/bin/bash
cd /content/work/s7
rm -rf runA runB
echo "A start $(date -Is)"
python s7_toy_qad_resume.py --out runA --data seqs.pt > runA.log 2>&1; echo "A exit $? $(date -Is)"
echo "B start $(date -Is)"
python s7_toy_qad_resume.py --out runB --data seqs.pt > runB_part1.log 2>&1 &
BP=$!
until [ -f runB/loss.jsonl ] && [ "$(wc -l < runB/loss.jsonl)" -ge 27 ]; do sleep 0.2; done
kill -9 $BP; wait $BP 2>/dev/null; echo "B killed after $(wc -l < runB/loss.jsonl) logged steps $(date -Is)"
python s7_toy_qad_resume.py --out runB --data seqs.pt > runB_part2.log 2>&1; echo "B resume exit $? $(date -Is)"
