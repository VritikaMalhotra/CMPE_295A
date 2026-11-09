"""C3 backend: token budget, request body, response mapping and server start-up (design spec 7.2).

A fake llama-server (http.server on a free port) answers /tokenize and /completion with canned replies
and records every request body, so no model or GPU is needed.
"""
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from testbed.backends import Backend, ContextOverflow, LlamaServer, ServerConfig
from testbed.spec import canonical_json, sha256_text

MSGS = [{"role": "user", "content": "list files"}]
TOOLS = [{"name": "ls", "description": "list", "parameters": {"type": "dict", "properties": {}, "required": []}}]


class FakeServer:
    """``n_tokens`` prompt tokens; ``reply`` is the /completion JSON; ``status``/``error`` force an HTTP error."""

    def __init__(self, n_tokens=10, reply=None, status=200, error=""):
        self.n_tokens, self.status, self.error = n_tokens, status, error
        self.reply = reply or {"content": "hello", "stop_type": "eos", "tokens_evaluated": n_tokens,
                               "tokens_predicted": 3, "timings": {"prompt_ms": 5.0, "predicted_ms": 7.0}}
        self.bodies = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    self._send(200, {"tokens": list(range(fake.n_tokens))})
                    return
                fake.bodies.append(body)
                if fake.status != 200:
                    self._send(fake.status, {"error": fake.error})
                else:
                    self._send(200, fake.reply)

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class _Srv:
    """Stands in for a running LlamaServer: Backend only reads ``base`` and ``cfg.ctx``."""

    def __init__(self, base, ctx):
        self.base = base
        self.cfg = ServerConfig(binary="llama-server", model_path="m.gguf", backend="cpu", ctx=ctx)


@pytest.fixture
def fake():
    servers = []

    def make(**kw):
        s = FakeServer(**kw)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()


def _backend(fs, ctx=100, **decoding):
    return Backend(_Srv(fs.base, ctx), decoding or {"n_predict": 4096})


def test_budget_and_overflow(fake):
    fs = fake(n_tokens=30)
    _backend(fs, ctx=100).generate(MSGS, TOOLS)
    assert fs.bodies[-1]["n_predict"] == 100 - 30 - 2            # min(cap, ctx - n - 2)
    _backend(fs, ctx=1000, n_predict=50).generate(MSGS, TOOLS)
    assert fs.bodies[-1]["n_predict"] == 50                       # the cap wins when it is smaller
    with pytest.raises(ContextOverflow):
        _backend(fake(n_tokens=98), ctx=100).generate(MSGS, TOOLS)  # budget 0: never sent
    with pytest.raises(ContextOverflow):
        _backend(fake(status=400, error="the request exceeds the available context size"),
                 ctx=1000).generate(MSGS, TOOLS)


def test_baseline_sends_no_grammar(fake):
    fs = fake()
    be = _backend(fs)
    be.generate(MSGS, TOOLS, constraint=None)
    assert "grammar" not in fs.bodies[-1]
    g = 'root ::= "x"'
    r = be.generate(MSGS, TOOLS, constraint=g)
    assert fs.bodies[-1]["grammar"] == g and r.grammar_used


def test_cache_prompt_always_false(fake):
    fs = fake()
    _backend(fs).generate(MSGS, TOOLS)
    assert fs.bodies[-1]["cache_prompt"] is False
    _backend(fs, n_predict=4096, cache_prompt=True).generate(MSGS, TOOLS)
    assert fs.bodies[-1]["cache_prompt"] is True                  # only when the spec sets it


def test_bad_blocks_and_stop_reason(fake):
    content = ('<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>\n'
               '<tool_call>\n{"foo": 1}\n</tool_call>')
    r = _backend(fake(reply={"content": content, "stop_type": "limit", "timings": {}})).generate(MSGS, TOOLS)
    assert r.tool_call_blocks == 2 and r.bad_call_blocks == 1
    assert [c.name for c in r.tool_calls] == ["ls"]
    assert r.stop_reason == "limit"
    r = _backend(fake(reply={"content": "x", "stopped_limit": True, "timings": {}})).generate(MSGS, TOOLS)
    assert r.stop_reason == "limit"                               # older servers: no stop_type
    r = _backend(fake(reply={"content": "x", "timings": {}})).generate(MSGS, TOOLS)
    assert r.stop_reason == "eos"


def test_prompt_hashes(fake):
    """FR-E17: the step records what was sent; same prompt and tokens -> same hashes on any backend."""
    fs = fake(n_tokens=12)
    r = _backend(fs).generate(MSGS, TOOLS)
    assert r.prompt_sha256 == sha256_text(fs.bodies[-1]["prompt"])
    assert r.prompt_tokens_sha256 == sha256_text(canonical_json(list(range(12))))
    assert r.prompt_tokens_sha256 != _backend(fake(n_tokens=13)).generate(MSGS, TOOLS).prompt_tokens_sha256


def test_server_exit_is_reported(tmp_path):
    false = shutil.which("false")
    if not false:
        pytest.skip("no `false` binary")
    log = str(tmp_path / "server.log")
    srv = LlamaServer(ServerConfig(binary=false, model_path="m.gguf", backend="cpu"), log_path=log,
                      startup_timeout_s=20)
    with pytest.raises(RuntimeError, match="server.log"):
        srv.__enter__()
    assert srv.proc.poll() is not None                            # exited, nothing left running
    srv.__exit__(None, None, None)
    assert srv.proc is None


def test_missing_binary(tmp_path):
    srv = LlamaServer(ServerConfig(binary=str(tmp_path / "nope"), model_path="m.gguf", backend="cpu"))
    with pytest.raises(FileNotFoundError):
        srv.__enter__()


def test_argv_backend_flags():
    cuda = ServerConfig(binary="b", model_path="m", backend="cuda").argv(8080)
    cpu = ServerConfig(binary="b", model_path="m", backend="cpu", threads=4).argv(8080)
    assert cuda[cuda.index("-ngl") + 1] == "99" and "-t" not in cuda
    assert cpu[cpu.index("-ngl") + 1] == "0" and cpu[cpu.index("-t") + 1] == "4"
    assert cpu[cpu.index("-np") + 1] == "1"                       # one slot: timings never shared
