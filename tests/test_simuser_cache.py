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
