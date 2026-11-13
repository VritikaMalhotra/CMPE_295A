"""Report: builds from episode/step files; CPU-vs-CUDA parity section counts identical steps."""
import json
import os

from testbed.report import write_report


def _write(d, run_id, backend, texts, seed=1):
    os.makedirs(f"{d}/episodes", exist_ok=True)
    os.makedirs(f"{d}/steps", exist_ok=True)
    ep = dict(run_id=run_id, config_key=backend, task_id="multi_turn_base_0", seed=seed, variant="q4_0",
              backend=backend, repair="none", constraint="none", valid=False, error_type="x", n_turns=1,
              n_steps=len(texts), forced_quit=False, wall_s=1.0, context_overflow=False, energy_j=None,
              peak_gpu_mib=None)
    json.dump(ep, open(f"{d}/episodes/{run_id}.json", "w"))
    with open(f"{d}/steps/{run_id}.jsonl", "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps(dict(run_id=run_id, config_key=backend, task_id="multi_turn_base_0", seed=seed,
                                    turn=0, step=i, variant="q4_0", backend=backend, repair="none",
                                    constraint="none", prompt_tokens=10, completion_tokens=5, ttft_ms=1.0,
                                    decode_ms=1.0, latency_ms=2.0, stop_reason="eos", raw_text=t,
                                    n_tool_call_blocks=0, decoded_calls=[], execution_results=[], known_tools=True,
                                    schema_errors=[], turn_has_ground_truth=True, is_last_step_of_turn=False,
                                    forced_quit=False, failure_class="end_turn")) + "\n")


def test_parity_section(tmp_path):
    d = str(tmp_path)
    _write(d, "a1", "cuda", ["same", "x"])
    _write(d, "b1", "cpu", ["same", "y"])
    text = open(write_report(d)).read()
    assert "Aligned steps: 2; identical: 1 (50.0%)" in text
    assert "First step of each episode: 1 of 1 identical" in text


def test_report_lists_infrastructure_exclusions(tmp_path):
    d = str(tmp_path)
    _write(d, "a1", "cuda", ["same"])
    os.makedirs(f"{d}/retries", exist_ok=True)
    with open(f"{d}/retries/gone.jsonl", "w") as f:            # 4 failed attempts, never finished
        f.write("\n".join(json.dumps({"error": "RuntimeError"}) for _ in range(4)) + "\n")
    with open(f"{d}/retries/a1.jsonl", "w") as f:              # failed once, then finished
        f.write(json.dumps({"error": "TimeoutError"}) + "\n")
    text = open(write_report(d)).read()
    assert "Excluded after 4 infrastructure failures (1): `gone`" in text
