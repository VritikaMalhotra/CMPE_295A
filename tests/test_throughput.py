"""NFR-5 pilot pieces: the shard option that splits one configuration across jobs, and the overlap count."""
import datetime as dt
import importlib.util
import os

import pytest

from testbed.run import _shard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shard_parsing():
    assert _shard(None) is None and _shard("") is None
    assert _shard("2/8") == (2, 8)
    with pytest.raises(ValueError):
        _shard("8/8")


def test_shards_partition_the_specs():
    ids = [f"{i:012x}" for i in range(50)]
    parts = [[x for k, x in enumerate(sorted(ids)) if k % 4 == i] for i in range(4)]   # as run_local does
    assert sorted(sum(parts, [])) == sorted(ids) and all(len(p) in (12, 13) for p in parts)


def test_max_overlap():
    spec = importlib.util.spec_from_file_location("tr", os.path.join(ROOT, "scripts", "throughput_report.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    t = lambda m: dt.datetime(2026, 11, 8, 12, m)                  # noqa: E731
    assert tr.max_overlap([(t(0), t(10)), (t(5), t(15)), (t(10), t(20)), (t(30), t(40))]) == 2
    assert tr.max_overlap([(t(0), t(10))] * 3) == 3
