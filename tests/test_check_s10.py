"""scripts/check_s10.py: S10 array-requeue verdict on a synthetic experiment directory."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("check_s10", os.path.join(ROOT, "scripts", "check_s10.py"))
check_s10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_s10)


def _make(tmp_path, n_cfg=2, n_ep=3, done_skip=(), dup_step=None, requeue=True):
    d = tmp_path / "run"
    for sub in ("episodes", "steps", "slurm_logs"):
        (d / sub).mkdir(parents=True)
    rows = []
    for c in range(n_cfg):
        for e in range(n_ep):
            rid = f"r{c}{e}"
            rows.append({"run_id": rid, "config_key": f"cfg{c}", "task_id": f"t{e}", "seed": 1})
            if rid in done_skip:
                continue
            (d / "episodes" / f"{rid}.json").write_text(json.dumps({"run_id": rid}))
            steps = [{"run_id": rid, "turn": 0, "step": 0}, {"run_id": rid, "turn": 0, "step": 1}]
            if rid == dup_step:
                steps.append({"run_id": rid, "turn": 0, "step": 1})
            (d / "steps" / f"{rid}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (d / "specs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for c in range(n_cfg):
        log = [f"config cfg{c} task 900_{c} on g3, restart 0, CUDA_VISIBLE_DEVICES=0"]
        if requeue and c == 0:
            log += ["walltime near: stopping the runner and requeueing 900_0",
                    "config cfg0 task 900_0 on g5, restart 1, CUDA_VISIBLE_DEVICES=1"]
        (d / "slurm_logs" / f"900_{c}.out").write_text("\n".join(log) + "\n")
    return str(d)


def test_complete_and_requeued_passes(tmp_path):
    r = check_s10.check(_make(tmp_path))
    assert r["planned"] == 6 and r["done"] == 6 and r["configs"] == 2 and r["pass"]
    assert r["tasks_requeued_by_walltime"] == ["900_0"]
    assert r["array_tasks"]["900_0"]["hosts"] == ["g3", "g5"] and r["array_tasks"]["900_0"]["max_restart"] == 1


def test_missing_episode_fails(tmp_path):
    r = check_s10.check(_make(tmp_path, done_skip={"r11"}))
    assert r["missing"] == ["r11"] and not r["pass"]


def test_duplicate_step_rows_fail(tmp_path):
    r = check_s10.check(_make(tmp_path, dup_step="r01"))
    assert r["duplicate_step_rows"] == {"r01": 1} and not r["pass"]


def test_no_requeue_fails(tmp_path):
    r = check_s10.check(_make(tmp_path, requeue=False))
    assert r["done"] == r["planned"] and not r["tasks_requeued_by_walltime"] and not r["pass"]


def test_s10_config_plans_8_configs_and_104_episodes():
    from omegaconf import OmegaConf
    exp = OmegaConf.load(os.path.join(ROOT, "configs", "experiment", "s10_requeue.yaml")).experiment
    n_cfg = len(exp.baselines) + len(exp.repairs)
    n_ep = len(exp.baselines) * len(exp.baseline_seeds) * exp.n_tasks + len(exp.repairs) * len(exp.repair_seeds) * exp.n_tasks
    assert n_cfg == 8 and n_ep == 104
    assert all(c.backend == "cuda" for c in list(exp.baselines) + list(exp.repairs))
