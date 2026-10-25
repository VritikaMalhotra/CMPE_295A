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


# config keys of the BFCL configurations as run on Oct 24-25 (report.md, d92a85e486)
OCT25_KEYS = {"8bae173bdd21", "ec9d23f8549d", "dc08400d1b5b", "7d5318997f9f", "b373660af0bb", "74260a9a88e3",
              "80e76cc888af"}


def _tasks(exp):
    t = {exp["benchmark"]: TASKS}
    for e in exp.get("extra_benchmarks", []):
        t[e["name"]] = [str(i) for i in e["task_ids"]]
    return t


def test_adding_tau2_keeps_bfcl_run_ids(skeleton_exp):
    specs = reg.expand_all(skeleton_exp, _tasks(skeleton_exp))
    bfcl = [s for s in specs if s.benchmark == "bfcl_multi_turn_base"]
    tau2 = [s for s in specs if s.benchmark == "tau2_verified_airline"]
    assert len(bfcl) == 220 and len(tau2) == 24 + 9
    assert {s.config_key for s in bfcl} == OCT25_KEYS          # finished BFCL episodes carry over
    assert [s.run_id for s in bfcl] == [s.run_id for s in reg.expand(skeleton_exp, TASKS)]
    assert len({s.config_key for s in tau2}) == 7 and not ({s.config_key for s in tau2} & OCT25_KEYS)
    t = tau2[0]
    assert t.server["simuser"]["sha256"] and t.server["tau2"]["commit"] and t.max_steps_per_turn == 200


def test_simulator_change_changes_tau2_ids_only(skeleton_exp):
    exp2 = json.loads(json.dumps(skeleton_exp))
    exp2["extra_benchmarks"][0]["simuser"]["sha256"] = "f" * 64
    a = reg.expand_all(skeleton_exp, _tasks(skeleton_exp))
    b = reg.expand_all(exp2, _tasks(exp2))
    for x, y in zip(a, b):
        assert (x.run_id == y.run_id) == (x.benchmark == "bfcl_multi_turn_base")


def test_import_from_copies_only_matching_finished_episodes(tmp_path, skeleton_exp):
    specs = reg.expand(skeleton_exp, TASKS[:2])
    old = reg.Registry(str(tmp_path / "old"), specs)
    for s in specs[:3]:
        json.dump({"run_id": s.run_id}, open(os.path.join(old.dir, "episodes", f"{s.run_id}.json"), "w"))
        os.makedirs(os.path.join(old.dir, "steps"), exist_ok=True)
        open(os.path.join(old.dir, "steps", f"{s.run_id}.jsonl"), "w").write("{}\n")
    # one finished episode without its step file is not imported (it would have no steps)
    os.remove(os.path.join(old.dir, "steps", f"{specs[2].run_id}.jsonl"))
    new = reg.Registry(str(tmp_path / "new"), specs[:2] + specs[5:7])
    assert new.import_from(old.dir) == 2
    assert new.done_ids() == {specs[0].run_id, specs[1].run_id}
    assert new.import_from(old.dir) == 0                          # idempotent
    assert json.load(open(os.path.join(new.dir, "imports.json")))[0]["episodes"] == 2
