"""C4: cache keys equal iff inputs equal; hits never reach upstream; replay makes zero calls; outages are 503."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from testbed.simuser_cache import CacheProxy, SimUserCache, request_key

SIM = {"model": "gpt-oss-20b", "sha256": "a" * 64, "ctx": 32768, "temperature": 0.0}
BODY = {"model": "simuser", "messages": [{"role": "system", "content": "be Emma"}, {"role": "user", "content": "Hi"}],
        "temperature": 0.0, "seed": 7}


class FakeUpstream:
    def __init__(self):
        self.calls = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(body)
                out = json.dumps({"id": "x", "object": "chat.completion", "model": body["model"],
                                  "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                      "role": "assistant", "content": f"reply {len(outer.calls)}"}}],
                                  "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_key_changes_exactly_with_inputs():
    k = request_key(SIM, BODY)
    assert k == request_key(dict(SIM), json.loads(json.dumps(BODY)))
    assert k == request_key(SIM, {**BODY, "stream": False, "api_key": "x", "num_retries": 3})   # transport only
    for changed in ({**BODY, "seed": 8}, {**BODY, "temperature": 0.1},
                    {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "ok"}]},
                    {**BODY, "model": "judge"}):
        assert request_key(SIM, changed) != k
    assert request_key({**SIM, "sha256": "b" * 64}, BODY) != k          # different simulator file


def test_hit_miss_and_replay(tmp_path):
    up = FakeUpstream()
    try:
        db = str(tmp_path / "c.sqlite")
        c = SimUserCache(db, SIM, upstream=up.url, inject={"chat_template_kwargs": {"reasoning_effort": "low"}})
        s1, r1 = c.complete(BODY)
        s2, r2 = c.complete(BODY)
        assert s1 == s2 == 200 and r1 == r2 and len(up.calls) == 1
        assert up.calls[0]["chat_template_kwargs"] == {"reasoning_effort": "low"}       # injected upstream
        c.complete({**BODY, "seed": 9})
        assert len(up.calls) == 2
        st = c.stats()["by_role"]["simuser"]
        assert st["miss"]["n"] == 2 and st["hit"]["n"] == 1
        c.close()
        # replay: answers from the store, never calls upstream, a miss is an error
        rp = SimUserCache(db, SIM, upstream=up.url, mode="replay",
                          inject={"chat_template_kwargs": {"reasoning_effort": "low"}})
        assert rp.complete(BODY) == (200, r1) and len(up.calls) == 2
        status, _ = rp.complete({**BODY, "seed": 10})
        assert status == 503 and len(up.calls) == 2
        rp.close()
    finally:
        up.close()


def test_outage_and_budget_are_503(tmp_path):
    c = SimUserCache(str(tmp_path / "c.sqlite"), SIM, upstream="http://127.0.0.1:9", timeout_s=2)
    status, body = c.complete(BODY)
    assert status == 503 and c.upstream_failures == 1 and "unreachable" in body["error"]["message"]
    up = FakeUpstream()
    try:
        b = SimUserCache(str(tmp_path / "b.sqlite"), SIM, upstream=up.url, max_upstream_hours=0.0)
        assert b.complete(BODY)[0] == 503 and not up.calls
    finally:
        up.close()


def test_proxy_speaks_openai_chat_completions(tmp_path):
    up = FakeUpstream()
    try:
        c = SimUserCache(str(tmp_path / "c.sqlite"), SIM, upstream=up.url)
        with CacheProxy(c) as p:
            r = requests.post(p.base_url + "/chat/completions", json=BODY, timeout=10)
            assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "reply 1"
            assert requests.post(p.base_url + "/chat/completions", json=BODY, timeout=10).json() == r.json()
            assert requests.get(p.base_url.replace("/v1", "") + "/health", timeout=5).json()["status"] == "ok"
        assert len(up.calls) == 1
        c.backup_to(str(tmp_path / "copy.sqlite"))
        assert SimUserCache(str(tmp_path / "copy.sqlite"), SIM, mode="replay").complete(BODY)[0] == 200
    finally:
        up.close()


def test_empty_reply_escalates_reasoning_and_stores_the_first_nonempty(tmp_path):
    up = FakeUpstream()
    orig = up.httpd.RequestHandlerClass.do_POST

    def do_post(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        up.calls.append(body)
        effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        content = "" if effort == "low" else f"reply at {effort}"
        out = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    up.httpd.RequestHandlerClass.do_POST = do_post
    try:
        c = SimUserCache(str(tmp_path / "c.sqlite"), SIM, upstream=up.url,
                         inject={"chat_template_kwargs": {"reasoning_effort": "low"}})
        status, r = c.complete(BODY)
        assert status == 200 and r["choices"][0]["message"]["content"] == "reply at medium" and len(up.calls) == 2
        assert c.complete(BODY)[1] == r and len(up.calls) == 2          # replay serves the stored reply
        assert c.stats()["by_role"]["simuser"]["empty_reply_0"]["n"] == 1
    finally:
        up.httpd.RequestHandlerClass.do_POST = orig
        up.close()


def test_user_context_overflow_is_stored_not_an_outage(tmp_path):
    """llama-server's 400 exceed_context_size: an episode outcome. Stored, so a replay reproduces it."""
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            out = json.dumps({"error": {"code": 400, "type": "exceed_context_size_error",
                                        "message": "request (32794 tokens) exceeds the available context size "
                                                   "(32768 tokens)"}}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        db = str(tmp_path / "c.sqlite")
        c = SimUserCache(db, SIM, upstream=f"http://127.0.0.1:{srv.server_address[1]}")
        status, body = c.complete(BODY)
        assert status == 400 and "exceed_context_size" in json.dumps(body)
        assert c.user_overflows == 1 and c.upstream_failures == 0
        c.close()
        rp = SimUserCache(db, SIM, mode="replay")
        assert rp.complete(BODY) == (status, body) and rp.user_overflows == 1 and len(calls) == 1
        rp.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_empty_after_escalation_is_counted_and_replays(tmp_path):
    """gpt-oss empty at every reasoning effort: counted, stored, and a replay counts it again."""
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            out = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        db = str(tmp_path / "c.sqlite")
        c = SimUserCache(db, SIM, upstream=f"http://127.0.0.1:{srv.server_address[1]}")
        assert c.complete(BODY)[0] == 200 and c.empty_replies == 1 and len(calls) == 3   # every escalation
        c.close()
        rp = SimUserCache(db, SIM, mode="replay")
        assert rp.complete(BODY)[0] == 200 and rp.empty_replies == 1 and len(calls) == 3
        rp.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_permanent_failure_excludes_at_once(tmp_path):
    from testbed.logger import MAX_INFRA_ATTEMPTS, record_retry, retry_count
    record_retry(str(tmp_path), "r1", ValueError("x"))
    assert retry_count(str(tmp_path), "r1") == 1
    record_retry(str(tmp_path), "r1", RuntimeError("empty"), permanent=True)
    assert retry_count(str(tmp_path), "r1") == MAX_INFRA_ATTEMPTS


def _status_upstream(status: int, text: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = text.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_upstream_500_is_counted_apart_from_outages(tmp_path):
    """An HTTP 500 answer (the simulator is up, one request failed) is counted in upstream_500s; a 503 and an
    unreachable simulator are outages only. Nothing is stored, so a later attempt asks again."""
    err = '{"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format"}}'
    srv, url = _status_upstream(500, err)
    try:
        c = SimUserCache(str(tmp_path / "a.sqlite"), SIM, upstream=url)
        status, body = c.complete(BODY)
        assert status == 503 and "upstream returned 500" in body["error"]["message"]
        assert c.upstream_failures == 1 and c.upstream_500s == 1
        assert c.complete(BODY)[0] == 503 and c.upstream_500s == 2          # not cached: asked again
    finally:
        srv.shutdown()
        srv.server_close()
    srv, url = _status_upstream(503, '{"error":{"message":"Loading model"}}')
    try:
        b = SimUserCache(str(tmp_path / "b.sqlite"), SIM, upstream=url)
        assert b.complete(BODY)[0] == 503 and b.upstream_failures == 1 and b.upstream_500s == 0
    finally:
        srv.shutdown()
        srv.server_close()
    d = SimUserCache(str(tmp_path / "d.sqlite"), SIM, upstream="http://127.0.0.1:9", timeout_s=2)
    assert d.complete(BODY)[0] == 503 and d.upstream_failures == 1 and d.upstream_500s == 0


def test_simulator_failure_kind():
    """Only when every failure of the episode was a 500 answer is it this episode's error; else an outage."""
    from types import SimpleNamespace
    from testbed.tau2_bench import SimulatorRequestFailed, SimulatorUnavailable, _simulator_failure
    c = SimpleNamespace(upstream_failures=5, upstream_500s=3)
    assert isinstance(_simulator_failure(c, 4, 2, "x"), SimulatorRequestFailed)     # 1 new failure, a 500
    assert isinstance(_simulator_failure(c, 3, 2, "x"), SimulatorUnavailable)       # 2 new, only 1 a 500
    assert isinstance(_simulator_failure(c, 5, 3, "x"), SimulatorUnavailable)       # none new


def test_judge_error_keeps_the_scored_episode_unless_outage():
    """The NL judge is diagnostic: a plain judge error or an HTTP 500 answer keeps the episode; an outage does not."""
    from types import SimpleNamespace
    from testbed.tau2_bench import _judge_error_keeps_episode
    assert _judge_error_keeps_episode(SimpleNamespace(upstream_failures=2, upstream_500s=1), 2, 1)      # no failure
    assert _judge_error_keeps_episode(SimpleNamespace(upstream_failures=3, upstream_500s=2), 2, 1)      # one 500
    assert not _judge_error_keeps_episode(SimpleNamespace(upstream_failures=3, upstream_500s=1), 2, 1)  # outage


def test_run_status_counts_used_up_attempts_as_excluded(tmp_path):
    """scripts/run_status.py: a permanent line or MAX_INFRA_ATTEMPTS attempts = excluded, as in retry_count."""
    import importlib.util
    import os
    from testbed.logger import MAX_INFRA_ATTEMPTS, record_retry
    spec = importlib.util.spec_from_file_location(
        "run_status", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_status.py"))
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    assert rs.MAX_INFRA_ATTEMPTS == MAX_INFRA_ATTEMPTS
    rd = tmp_path / "run"
    (rd / "episodes").mkdir(parents=True)
    with open(rd / "specs.jsonl", "w") as f:
        for i in range(3):
            f.write(json.dumps({"run_id": f"r{i}", "benchmark": "tau2_verified_retail", "experiment": "x",
                                "model": "m", "variant": "bf16"}) + "\n")
    for _ in range(MAX_INFRA_ATTEMPTS):
        record_retry(str(rd), "r0", RuntimeError("HTTP 500"))
    record_retry(str(rd), "r1", RuntimeError("once"))
    st = rs.status(str(rd))
    assert st["excluded"] == 1 and st["finished"] == 0
