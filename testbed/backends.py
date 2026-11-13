"""C3 Inference Backends: llama-server on CUDA and CPU behind one ``generate`` call.

Protocol pinned from the Oct 8 pilot (v2 harness): raw ``/completion`` with the model's prompt format
(``prompt_formats``: Qwen3 through ``qwen_format``, unchanged; SmolLM3 and LFM2.5 from their chat templates),
non-thinking suffix where the template has one, ``cache_prompt=false`` (prompt-cache reuse changed
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
from .prompt_formats import QWEN3, PromptFormat
from .spec import GenerateResult, ToolCall, canonical_json, sha256_text


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
    host: str = "127.0.0.1"        # 0.0.0.0 only for the shared simulated-user job (other nodes connect)
    parallel: int = 1              # slots; ctx is split evenly between them

    def argv(self, port: int) -> list[str]:
        a = [self.binary, "-m", self.model_path, "-c", str(self.ctx), "-np", str(self.parallel),
             "--host", self.host, "--port", str(port), "--seed", str(self.seed), "--no-webui"]
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

    def __init__(self, server: LlamaServer, decoding: dict, thinking: bool = False, timeout_s: int = 900,
                 fmt: PromptFormat = QWEN3):
        self.server = server
        self.fmt = fmt
        self.decoding = dict(decoding)
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.ctx = server.cfg.ctx

    def tokenize(self, prompt: str) -> list[int]:
        body = {"content": prompt, "add_special": False}
        if self.fmt.send_tokens:                    # these ids are what the model sees: parse the template's special
            body["parse_special"] = True            # tokens explicitly (llama-server's default, made explicit)
        r = requests.post(self.server.base + "/tokenize", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        return list(r.json()["tokens"])

    def count_tokens(self, prompt: str) -> int:
        return len(self.tokenize(prompt))

    def generate(self, messages: list[dict], tools: list[dict], constraint: Optional[str] = None,
                 seed: int = 0) -> GenerateResult:
        """``constraint`` is a GBNF grammar string (R1) or None.

        Token budget follows bfcl's local handler: min(cap, ctx - prompt_tokens - 2), cap = decoding.n_predict
        (bfcl: 4096). A prompt that leaves no budget raises ContextOverflow.
        """
        prompt = self.fmt.format_prompt(messages, tools, thinking=self.thinking)
        token_ids = self.tokenize(prompt)
        n_prompt = len(token_ids)
        budget = min(int(self.decoding.get("n_predict", 4096)), self.ctx - n_prompt - 2)
        if budget <= 0:
            raise ContextOverflow(f"prompt {n_prompt} tokens, context {self.ctx}")
        body = {
            "prompt": token_ids if self.fmt.send_tokens else prompt,
            "n_predict": budget,
            "temperature": float(self.decoding.get("temperature", 0.0)),
            "top_p": float(self.decoding.get("top_p", 1.0)),
            "top_k": int(self.decoding.get("top_k", 0)),
            "min_p": float(self.decoding.get("min_p", 0.0)),
            "seed": int(seed),
            "cache_prompt": bool(self.decoding.get("cache_prompt", False)),
            "stream": False,
        }
        if self.fmt.preserve:          # control tokens the parser needs (LFM2.5 call markers): keep them in content
            body["preserved_tokens"] = list(self.fmt.preserve)
        if constraint:
            body["grammar"] = constraint
        t0 = time.time()
        r = requests.post(self.server.base + "/completion", json=body, timeout=self.timeout_s)
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 400 and "context" in r.text:
            raise ContextOverflow(r.text[:200])
        r.raise_for_status()
        res = parse_completion(r.json(), n_prompt=n_prompt, budget=budget, constraint=constraint,
                               latency_ms=latency_ms, fmt=self.fmt)
        res.prompt_sha256 = sha256_text(prompt)
        res.prompt_tokens_sha256 = sha256_text(canonical_json(token_ids))
        return res


def parse_completion(j: dict, n_prompt: int, budget: int, constraint: Optional[str],
                     latency_ms: float, fmt: PromptFormat = QWEN3) -> GenerateResult:
    """llama-server's /completion JSON -> GenerateResult (also used by replay tests with recorded text)."""
    raw = fmt.raw_prefix + j.get("content", "")
    _, cleaned = fmt.split_think(raw)
    calls, n_blocks = fmt.extract_tool_calls(cleaned)
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
