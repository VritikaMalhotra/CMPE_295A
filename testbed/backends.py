"""C3 Inference Backends: llama-server on CUDA and CPU behind one ``generate`` call.

Owner: Ekant Kapgate (draft by Yashashav for review).

Protocol pinned from the Oct 8 pilot (v2 harness): raw ``/completion`` with the Qwen3 prompt
built in ``qwen_format``, non-thinking suffix, ``cache_prompt=false`` (prompt-cache reuse changed
61/400 greedy Q4_0 outputs), greedy decoding unless the spec says otherwise, one slot.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests

from . import qwen_format
from .spec import GenerateResult, ToolCall


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerConfig:
    binary: str                    # path to llama-server (CUDA or CPU build)
    model_path: str
    backend: str                   # "cuda" | "cpu"
    ctx: int = 40960
    threads: Optional[int] = None  # CPU threads; None = llama.cpp default
    seed: int = 0
    extra_args: tuple = ()

    def argv(self, port: int) -> list[str]:
        a = [self.binary, "-m", self.model_path, "-c", str(self.ctx), "-np", "1",
             "--host", "127.0.0.1", "--port", str(port), "--seed", str(self.seed), "--no-webui"]
        a += ["-ngl", "99"] if self.backend == "cuda" else ["-ngl", "0"]
        if self.threads:
            a += ["-t", str(self.threads)]
        return a + list(self.extra_args)


class LlamaServer:
    """Starts one llama-server process; use as a context manager."""

    def __init__(self, cfg: ServerConfig, log_path: Optional[str] = None, startup_timeout_s: int = 300):
        self.cfg = cfg
        self.port = _free_port()
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self.base = f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "LlamaServer":
        if not (os.path.exists(self.cfg.binary) or shutil.which(self.cfg.binary)):
            raise FileNotFoundError(f"llama-server not found: {self.cfg.binary}")
        log = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cfg.argv(self.port), stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(self.base + "/health", timeout=2).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise TimeoutError("llama-server did not become healthy")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def props(self) -> dict:
        try:
            return requests.get(self.base + "/props", timeout=5).json()
        except Exception:
            return {}


class ContextOverflow(RuntimeError):
    """The prompt no longer fits the context window (bfcl's budget would be <= 0)."""


class Backend:
    """``generate(messages, tools, constraint)`` over a running LlamaServer."""

    def __init__(self, server: LlamaServer, decoding: dict, thinking: bool = False, timeout_s: int = 900):
        self.server = server
        self.decoding = dict(decoding)
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.ctx = server.cfg.ctx

    def count_tokens(self, prompt: str) -> int:
        r = requests.post(self.server.base + "/tokenize", json={"content": prompt, "add_special": False},
                          timeout=self.timeout_s)
        r.raise_for_status()
        return len(r.json()["tokens"])

    def generate(self, messages: list[dict], tools: list[dict], constraint: Optional[str] = None,
                 seed: int = 0) -> GenerateResult:
        """``constraint`` is a GBNF grammar string (R1) or None.

        Token budget follows bfcl's local handler: min(cap, ctx - prompt_tokens - 2), cap = decoding.n_predict
        (bfcl: 4096). A prompt that leaves no budget raises ContextOverflow.
        """
        prompt = qwen_format.format_prompt(messages, tools, thinking=self.thinking)
        n_prompt = self.count_tokens(prompt)
        budget = min(int(self.decoding.get("n_predict", 4096)), self.ctx - n_prompt - 2)
        if budget <= 0:
            raise ContextOverflow(f"prompt {n_prompt} tokens, context {self.ctx}")
        body = {
            "prompt": prompt,
            "n_predict": budget,
            "temperature": float(self.decoding.get("temperature", 0.0)),
            "top_p": float(self.decoding.get("top_p", 1.0)),
            "top_k": int(self.decoding.get("top_k", 0)),
            "min_p": float(self.decoding.get("min_p", 0.0)),
            "seed": int(seed),
            "cache_prompt": bool(self.decoding.get("cache_prompt", False)),
            "stream": False,
        }
        if constraint:
            body["grammar"] = constraint
        t0 = time.time()
        r = requests.post(self.server.base + "/completion", json=body, timeout=self.timeout_s)
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 400 and "context" in r.text:
            raise ContextOverflow(r.text[:200])
        r.raise_for_status()
        j = r.json()
        raw = j.get("content", "")
        _, cleaned = qwen_format.split_think(raw)
        calls, n_blocks = qwen_format.extract_tool_calls(cleaned)
        good = [c for c in calls if isinstance(c, dict) and "name" in c and "arguments" in c]
        timings = j.get("timings") or {}
        return GenerateResult(
            text=cleaned,
            raw_text=raw,
            tool_calls=[ToolCall(c["name"], c["arguments"]) for c in good],
            tool_call_blocks=n_blocks,
            prompt_tokens=int(j.get("tokens_evaluated", timings.get("prompt_n", n_prompt)) or n_prompt),
            completion_tokens=int(j.get("tokens_predicted", timings.get("predicted_n", 0)) or 0),
            ttft_ms=timings.get("prompt_ms"),
            decode_ms=timings.get("predicted_ms"),
            latency_ms=latency_ms,
            stop_reason=str(j.get("stop_type") or ("limit" if j.get("stopped_limit") else "eos")),
            grammar_used=bool(constraint),
            bad_call_blocks=len(calls) - len(good),
            n_predict=budget,
        )
