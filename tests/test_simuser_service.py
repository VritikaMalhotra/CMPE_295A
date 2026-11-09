"""Decision 4: the simulated user as its own job. Endpoint file checks, per-job C4 stores and their merge for
replays, and shards that keep a tau2 task whole."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from testbed.backends import ServerConfig
from testbed.run import _cache_tag, cache_file_name, shard_members
from testbed.simuser_cache import SimUserCache, merge_stores
from testbed.simuser_service import (check_record, endpoint_record, remove_endpoint, wait_for_endpoint,
                                     write_endpoint)

SIM = {"model": "gpt-oss-20b", "file": "gpt-oss-20b-MXFP4.gguf", "sha256": "a" * 64, "ctx": 32768}


class Health:
    def __enter__(self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        return self

    def __exit__(self, *a):
        self.srv.shutdown()


def test_server_listens_where_asked():
    a = ServerConfig(binary="ls", model_path="m.gguf", backend="cuda", ctx=4 * 32768, host="0.0.0.0", parallel=4).argv(8080)
    assert a[a.index("--host") + 1] == "0.0.0.0" and a[a.index("-np") + 1] == "4" and a[a.index("-c") + 1] == "131072"
    d = ServerConfig(binary="ls", model_path="m.gguf", backend="cuda").argv(8080)
    assert d[d.index("--host") + 1] == "127.0.0.1" and d[d.index("-np") + 1] == "1"   # the agent's server: unchanged


def test_endpoint_record_must_match_the_experiment():
    rec = endpoint_record(SIM, "http://h:1", 32768, 4)
    assert check_record(rec, SIM) == []
    assert any("sha256" in b for b in check_record(rec, {**SIM, "sha256": "b" * 64}))
    assert any("ctx" in b for b in check_record(rec, {**SIM, "ctx": 16384}))


def test_wait_for_endpoint(tmp_path):
    path = str(tmp_path / "ep.json")
    with pytest.raises(RuntimeError, match="not written"):
        wait_for_endpoint(path, SIM, timeout_s=0)
    with Health() as h:
        write_endpoint(path, endpoint_record(SIM, h.url, 32768, 4))
        assert wait_for_endpoint(path, SIM, timeout_s=5) == h.url
        with pytest.raises(RuntimeError, match="different simulator"):      # at once, no waiting
            wait_for_endpoint(path, {**SIM, "file": "other.gguf"}, timeout_s=3600)
    with pytest.raises(RuntimeError, match="unreachable"):                   # the job died without cleanup
        wait_for_endpoint(path, SIM, timeout_s=0)


def test_remove_endpoint_only_own_file(tmp_path):
    path = str(tmp_path / "ep.json")
    write_endpoint(path, {**endpoint_record(SIM, "http://h:1", 32768, 4), "pid": 1})
    remove_endpoint(path, pid=2)                     # a newer simulator job wrote it: keep
    assert os.path.exists(path)
    remove_endpoint(path, pid=1)
    assert not os.path.exists(path)


def test_cache_file_per_job():
    cfg = lambda **k: SimpleNamespace(get=lambda key, d=None: k.get(key, d))   # noqa: E731
    assert cache_file_name(_cache_tag(cfg())) == "simuser_cache.sqlite"     # plain local run: the old name
    assert _cache_tag(cfg(only_config="abc")) == "abc"
    assert _cache_tag(cfg(only_config="abc", shard="1/3")) == "abc.shard1of3"


def _store(path, replies):
    c = SimUserCache(str(path), SIM, mode="replay")
    with c._lock:
        for k, v in replies.items():
            c._db.execute("INSERT INTO replies VALUES (?,?,?,?,?,?)", (k, "simuser", "{}", json.dumps(v), 0.0, 0.0))
        c._db.commit()
    c.close()


def test_merge_stores_first_wins(tmp_path):
    _store(tmp_path / "a.sqlite", {"k1": "a1", "k2": "a2"})
    _store(tmp_path / "b.sqlite", {"k2": "b2", "k3": "b3"})
    dst = tmp_path / "m.sqlite"
    added = merge_stores([str(tmp_path / "a.sqlite"), str(tmp_path / "b.sqlite")], str(dst))
    assert list(added.values()) == [2, 1]
    c = SimUserCache(str(dst), SIM, mode="replay")
    got = {k: json.loads(v) for k, v in c._db.execute("SELECT key, response FROM replies")}
    c.close()
    assert got == {"k1": "a1", "k2": "a2", "k3": "b3"}


def _spec(bench, task, seed, cfg="c"):
    rid = f"{bench}-{task}-{seed}-{cfg}"
    return SimpleNamespace(benchmark=bench, task_id=task, seed=seed, run_id=rid)


def test_shards_partition_and_keep_tau2_tasks_whole():
    specs = [_spec("bfcl_multi_turn_base", str(t), s) for t in range(20) for s in range(3)]
    specs += [_spec("tau2_verified_retail", str(t), s, c) for t in range(9) for s in range(5) for c in "xy"]
    parts = [shard_members(specs, i, 4) for i in range(4)]
    assert sorted(sum(map(list, parts), [])) == sorted(s.run_id for s in specs)   # disjoint, complete
    for t in range(9):                                                              # a tau2 task: one shard
        ids = {s.run_id for s in specs if s.benchmark.startswith("tau2") and s.task_id == str(t)}
        assert sum(bool(ids & p) for p in parts) == 1
    # stable: finishing episodes does not move the others (computed over planned specs, not pending)
    assert shard_members(specs, 2, 4) == parts[2]
