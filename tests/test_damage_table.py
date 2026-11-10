"""H1 damage table: pairing, rates and ratio on synthetic run directories (no real labels)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(tmp, name, variant, fail_every):
    d = tmp / name
    for sub in ("episodes", "steps"):
        (d / sub).mkdir(parents=True)
    specs = []
    for t in range(20):
        for seed in (0, 1, 2):
            rid = f"{name}-{t}-{seed}"
            specs.append({"run_id": rid, "benchmark": "bfcl_multi_turn_base", "task_id": f"multi_turn_base_{t}"})
            (d / "episodes" / f"{rid}.json").write_text(json.dumps(
                {"run_id": rid, "task_id": f"multi_turn_base_{t}", "seed": seed, "valid": False}))
            rows = []
            for s in range(10):
                bad = (s % fail_every == 0)
                rows.append({"run_id": rid, "config_key": "c", "task_id": f"multi_turn_base_{t}", "seed": seed, "turn": 0, "step": s,
                             "variant": variant, "backend": "cuda", "repair": "none", "constraint": "none",
                             "stop_reason": "eos", "raw_text": "x", "n_tool_call_blocks": 1,
                             "decoded_calls": ["ls()"], "execution_results": ["Error during execution: x" if bad else "{}"],
                             "known_tools": True, "schema_errors": [], "turn_has_ground_truth": True,
                             "is_last_step_of_turn": s == 9, "forced_quit": False, "prompt_tokens": 1,
                             "completion_tokens": 1, "ttft_ms": None, "decode_ms": None, "latency_ms": 1.0})
            (d / "steps" / f"{rid}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "specs.jsonl").write_text("\n".join(json.dumps(s) for s in specs) + "\n")
    return str(d)


def test_damage_table_ratio_and_pairing(tmp_path):
    ref = _run(tmp_path, "ref", "bf16", 5)        # 2 of 10 steps fail
    q4 = _run(tmp_path, "q4", "q4_0", 2)          # 5 of 10 steps fail
    out = tmp_path / "d.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "damage_table.py"), "--model", "m",
                        "--ref", f"BF16={ref}", "--cell", f"Q4_0={q4}", "--json", str(out), "--md", str(tmp_path / "d.md")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    c = json.load(open(out))["cells"]["Q4_0"]
    assert c["pairs"] == 40                      # seeds 1-2 only (seed 0 = greedy), 20 tasks
    e = c["classes"]["exec_error"]
    assert abs(e["rate_ref_per100"] - 20) < 1e-9 and abs(e["rate_4bit_per100"] - 50) < 1e-9
    assert abs(e["ratio"] - 2.5) < 1e-9 and e["ratio_ci95"][0] <= 2.5 <= e["ratio_ci95"][1]
    assert "development data, preview" in open(tmp_path / "d.md").read()
