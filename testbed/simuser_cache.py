"""C4 Simulated-User Cache: a content-addressed caching proxy in front of the self-hosted simulator.

Owner: Vritika Malhotra (draft by Yashashav for review).

tau2-bench's user simulator and its natural-language-assertion judge call an LLM through LiteLLM. We point
both at ``openai/<role>`` with ``api_base`` = this proxy, so the harness runs unmodified and never knows it is
cached (Workbook Ch. 6, "Proxy (simulated-user cache)"). The proxy:

* keys each ``POST /v1/chat/completions`` by SHA-256 over the simulator identity (model file SHA-256, server
  settings it injects) and the request body (conversation prefix, sampling settings, seed, tools); volatile
  transport fields are dropped first, so the key changes exactly when an input to the reply changes;
* answers hits from SQLite and forwards only misses to the upstream llama-server;
* keeps a ledger of every request (hit or miss, upstream seconds) and enforces an upstream-hour ceiling;
* in ``replay`` mode never calls upstream: a miss is an error, which is how "a replayed episode makes zero
  server calls" is checked (FR-E6).

An unreachable upstream or an exhausted budget returns HTTP 503. The runner treats that as "simulator
unavailable": the episode stays pending and is retried, never scored (Workbook Ch. 6, failure modes).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import requests

from .spec import canonical_json, sha256_text

KEY_VERSION = 2   # 2: replies are never empty (escalation below); v1 stores may hold empty replies
# transport-only fields LiteLLM or clients may add; they never change the reply
VOLATILE = {"stream", "stream_options", "user", "metadata", "num_retries", "timeout", "request_timeout",
            "api_key", "api_base", "extra_headers"}


# gpt-oss can spend its whole turn reasoning and return an empty message; tau2 then rejects the user turn
# (Oct 25: 3 of 33 episodes). On an empty reply the proxy asks again with more reasoning effort, in this
# fixed order, and stores the first non-empty reply under the original key, so replays stay identical.
ESCALATION = ({}, {"chat_template_kwargs": {"reasoning_effort": "medium"}},
              {"chat_template_kwargs": {"reasoning_effort": "high"}})


def is_empty_reply(resp: dict) -> bool:
    try:
        m = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return True
    return not (m.get("content") or "").strip() and not m.get("tool_calls")


def request_key(sim_identity: dict, body: dict) -> str:
    """The cache key: equal iff every input to the simulator's reply is equal."""
    clean = {k: v for k, v in body.items() if k not in VOLATILE}
    return sha256_text(canonical_json({"v": KEY_VERSION, "sim": sim_identity, "body": clean}))


class SimUserCache:
    """SQLite reply store + ledger. Thread-safe (one connection, one lock)."""

    def __init__(self, db_path: str, sim_identity: dict, upstream: Optional[str] = None, mode: str = "live",
                 inject: Optional[dict] = None, max_upstream_hours: Optional[float] = None,
                 timeout_s: int = 600):
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be live or replay, not {mode!r}")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self.sim_identity = dict(sim_identity)
        self.upstream = upstream.rstrip("/") if upstream else None
        self.mode = mode
        self.inject = dict(inject or {})
        self.max_upstream_hours = max_upstream_hours
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=DELETE")     # Drive-backed folders do not support WAL well
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, role TEXT, request TEXT, response TEXT,
                                                created REAL, upstream_s REAL);
            CREATE TABLE IF NOT EXISTS ledger (ts REAL, key TEXT, role TEXT, outcome TEXT, upstream_s REAL);
        """)
        self._db.commit()
        self.upstream_failures = 0          # read by the runner: >0 during an episode = simulator outage

    # ---- store -------------------------------------------------------------------------------------
    def _get(self, key: str) -> Optional[dict]:
        row = self._db.execute("SELECT response FROM replies WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _log(self, key: str, role: str, outcome: str, upstream_s: float = 0.0) -> None:
        self._db.execute("INSERT INTO ledger VALUES (?,?,?,?,?)", (time.time(), key, role, outcome, upstream_s))
        self._db.commit()

    def upstream_hours(self) -> float:
        (s,) = self._db.execute("SELECT COALESCE(SUM(upstream_s),0) FROM ledger").fetchone()
        return s / 3600.0

    def stats(self) -> dict:
        rows = self._db.execute("SELECT role, outcome, COUNT(*), COALESCE(SUM(upstream_s),0) FROM ledger "
                                "GROUP BY role, outcome").fetchall()
        out: dict = {"by_role": {}, "stored_replies": self._db.execute("SELECT COUNT(*) FROM replies").fetchone()[0],
                     "upstream_hours": self.upstream_hours(), "ceiling_hours": self.max_upstream_hours,
                     "sim": self.sim_identity}
        for role, outcome, n, s in rows:
            out["by_role"].setdefault(role, {})[outcome] = {"n": n, "upstream_s": round(s, 1)}
        return out

    # ---- the completion call -------------------------------------------------------------------------
    def complete(self, body: dict) -> tuple[int, dict]:
        """(HTTP status, JSON body) for one chat-completions request."""
        role = str(body.get("model", "?"))
        key = request_key({**self.sim_identity, "inject": self.inject}, body)
        with self._lock:
            hit = self._get(key)
            if hit is not None:
                self._log(key, role, "hit")
                return 200, hit
            if self.mode == "replay" or not self.upstream:
                self._log(key, role, "replay_miss")
                return 503, {"error": {"message": "simulated-user cache miss in replay mode", "key": key}}
            if self.max_upstream_hours is not None and self.upstream_hours() >= self.max_upstream_hours:
                self._log(key, role, "budget_stop")
                return 503, {"error": {"message": "simulator budget ceiling reached"}}
        dt = 0.0
        for step, extra in enumerate(ESCALATION):
            fwd = {k: v for k, v in body.items() if k not in VOLATILE}
            fwd.update(self.inject)
            fwd.update(extra)
            t0 = time.time()
            try:
                r = requests.post(self.upstream + "/v1/chat/completions", json=fwd, timeout=self.timeout_s)
                dt += time.time() - t0
                if r.status_code != 200:
                    with self._lock:
                        self.upstream_failures += 1
                        self._log(key, role, f"upstream_{r.status_code}", dt)
                    return 503, {"error": {"message": f"upstream returned {r.status_code}: {r.text[:300]}"}}
                resp = r.json()
            except requests.RequestException as e:
                with self._lock:
                    self.upstream_failures += 1
                    self._log(key, role, "upstream_unreachable", time.time() - t0)
                return 503, {"error": {"message": f"simulator unreachable: {type(e).__name__}"}}
            if not is_empty_reply(resp):
                break
            with self._lock:
                self._log(key, role, f"empty_reply_{step}", 0.0)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?,?,?,?,?,?)",
                             (key, role, canonical_json(fwd), json.dumps(resp), time.time(), dt))
            self._log(key, role, "miss", dt)
        return 200, resp

    def backup_to(self, path: str) -> None:
        """Consistent copy of the store (SQLite backup API), written to ``path`` via a temp file + rename."""
        tmp = f"{path}.{os.getpid()}.part"
        with self._lock:
            dst = sqlite3.connect(tmp)
            try:
                self._db.backup(dst)
            finally:
                dst.close()
        os.replace(tmp, path)

    def close(self) -> None:
        with self._lock:
            self._db.close()


class _Handler(BaseHTTPRequestHandler):
    cache: SimUserCache = None  # set per server

    def log_message(self, *a):  # quiet
        pass

    def _send(self, status: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._send(200, {"status": "ok", "mode": self.cache.mode})
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": "simuser", "object": "model"},
                                                               {"id": "judge", "object": "model"}]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "only /v1/chat/completions is proxied"}})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": {"message": "bad JSON"}})
        status, obj = self.cache.complete(body)
        self._send(status, obj)


class CacheProxy:
    """Runs the proxy on 127.0.0.1 in a daemon thread: ``with CacheProxy(cache) as p: p.base_url``."""

    def __init__(self, cache: SimUserCache, port: int = 0):
        handler = type("H", (_Handler,), {"cache": cache})
        self.cache = cache
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self) -> "CacheProxy":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def report(db_path: str) -> dict:
    c = SimUserCache(db_path, sim_identity={}, mode="replay")
    try:
        return c.stats()
    finally:
        c.close()
