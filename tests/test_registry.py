"""C11: expansion, hashing, dedup, resume."""
import json
import os

import pytest

from testbed import registry as reg
from testbed.spec import RunSpec

TASKS = [f"multi_turn_base_{i}" for i in range(20)]


def test_skeleton_expands_to_220_unique(skeleton_exp):
    specs = reg.expand(skeleton_exp, TASKS)
    assert len(specs) == 160 + 60
    assert len({s.run_id for s in specs}) == 220
    assert len({s.config_key for s in specs}) == 7


def test_run_id_is_stable_and_content_addressed(skeleton_exp):
    a = reg.expand(skeleton_exp, TASKS)
    b = reg.expand(skeleton_exp, TASKS)
    assert [s.run_id for s in a] == [s.run_id for s in b]
    s = a[0]
    assert RunSpec.from_dict(s.content()).run_id == s.run_id
    changed = RunSpec.from_dict({**s.content(), "decoding": {**s.decoding, "temperature": 0.0}})
    assert changed.run_id != s.run_id and changed.config_key != s.config_key


def test_config_key_ignores_task_and_seed(skeleton_exp):
    specs = reg.expand(skeleton_exp, TASKS)
    s = specs[0]
    other = RunSpec.from_dict({**s.content(), "task_id": "multi_turn_base_7", "seed": 99})
    assert other.config_key == s.config_key and other.run_id != s.run_id


def test_duplicate_specs_rejected(skeleton_exp):
    exp = dict(skeleton_exp)
    exp["baselines"] = exp["baselines"] + exp["baselines"][:1]
    with pytest.raises(ValueError):
        reg.expand(exp, TASKS)


def test_resume_counts_only_finished_episodes(tmp_path, skeleton_exp):
    specs = reg.expand(skeleton_exp, TASKS[:2])
    r = reg.Registry(str(tmp_path), specs)
    r.write_specs()
    assert len(r.pending()) == len(specs)
    done = specs[3].run_id
    with open(os.path.join(r.dir, "episodes", f"{done}.json"), "w") as f:
        json.dump({"run_id": done}, f)
    open(os.path.join(r.dir, "episodes", f"{specs[4].run_id}.json.part"), "w").close()   # interrupted write
    assert len(r.pending()) == len(specs) - 1 and done not in {s.run_id for s in r.pending()}
    # a changed config lands in a different experiment directory
    exp2 = {**skeleton_exp, "decoding": {**skeleton_exp["decoding"], "top_k": 40}}
    assert reg.Registry(str(tmp_path), reg.expand(exp2, TASKS[:2])).dir != r.dir


def test_manifest_flags_sha_mismatch(tmp_path, skeleton_exp):
    specs = reg.expand(skeleton_exp, TASKS[:1])
    f = tmp_path / "m.gguf"
    f.write_bytes(b"not the pilot file")
    m = reg.build_manifest(specs, {"q4_0": str(f)}, {}, expected_sha={"q4_0": "0" * 64})
    assert m["model_files"]["q4_0"]["sha256_matches_pilot"] is False
    assert m["n_specs"] == len(specs) and m["prompt_cache"] is False


def test_config_lock_is_exclusive_and_released(tmp_path):
    from testbed.run import _try_lock
    p = str(tmp_path / "locks" / "k.lock")
    a = _try_lock(p)
    assert a is not None
    # a second open file description in the same process contends like another process would
    assert _try_lock(p) is None
    a.close()
    b = _try_lock(p)
    assert b is not None
    b.close()
