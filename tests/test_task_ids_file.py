"""The registered holdout task list loads, and an edited list is refused (testbed.run.load_task_ids_file)."""
import hashlib
import json
import os

import pytest

from testbed.run import load_task_ids_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = os.path.join(REPO, "results", "m3", "holdout_tasks_bfcl.json")


def test_registered_holdout_loads():
    cats = load_task_ids_file(HOLDOUT)
    assert {k: len(v) for k, v in cats.items()} == {
        "bfcl_multi_turn_base": 100, "bfcl_multi_turn_miss_func": 100, "bfcl_multi_turn_miss_param": 100}
    assert len({t for v in cats.values() for t in v}) == 300


def test_holdout_is_disjoint_from_the_development_sample():
    pytest.importorskip("bfcl_eval")
    from testbed.adapter import BFCL_CATEGORIES, sample_task_ids
    cats = load_task_ids_file(HOLDOUT)
    for c in BFCL_CATEGORIES:
        assert not set(cats[f"bfcl_{c}"]) & set(sample_task_ids(c))


def test_edited_list_is_refused(tmp_path):
    d = json.load(open(HOLDOUT))
    d["categories"]["bfcl_multi_turn_base"][0] = "multi_turn_base_999"
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="SHA-256"):
        load_task_ids_file(str(p))


def test_list_without_hash_loads(tmp_path):
    p = tmp_path / "plain.json"
    p.write_text(json.dumps({"categories": {"bfcl_multi_turn_base": ["a", "b"]}}))
    assert load_task_ids_file(str(p)) == {"bfcl_multi_turn_base": ["a", "b"]}
    assert hashlib.sha256(b"a\nb").hexdigest()       # the hash convention: newline-joined ids
