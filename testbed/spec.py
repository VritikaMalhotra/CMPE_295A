"""Frozen interfaces for the M1 walking skeleton (Workbook Ch. 6).

Three contracts every component codes against:

* ``RunSpec``   - one fully specified run (model variant x backend x repair x task x seed).
                  Produced by C11, consumed by C2. Its ``run_id`` is a hash of its content,
                  so the same spec always gets the same id and duplicates are impossible.
* ``GenerateResult`` - what C3's ``generate(messages, tools, constraint)`` returns.
* ``StepRow``   - one log row per model step, written by C5, labeled by C6, read by C10.

Change these only by pull request reviewed by all three members.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Backend = Literal["cuda", "cpu"]
Constraint = Literal["none", "free", "forced"]  # R1 grammar modes (free = default R1, forced = ablation)
Repair = Literal["none", "r1", "r2", "r3"]

SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> str:
    """Key-sorted, whitespace-free JSON: the input to every hash in the testbed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunSpec:
    """One episode to run. Every field that can change an output is part of the hash."""

    experiment: str                 # e.g. "skeleton"
    model: str                      # e.g. "qwen3-0.6b"
    variant: str                    # GGUF variant key, e.g. "bf16", "q4_k_m", "q4_0", "q8_0", "r3_qad_q4_0"
    backend: Backend
    repair: Repair
    constraint: Constraint
    benchmark: str                  # e.g. "bfcl_multi_turn_base"
    task_id: str                    # e.g. "multi_turn_base_0"
    seed: int
    decoding: dict = field(default_factory=dict)   # temperature, top_p, top_k, n_predict, ...
    server: dict = field(default_factory=dict)     # llama-server flags that change outputs (ctx, cache_prompt, ...)
    thinking: bool = False
    max_steps_per_turn: int = 20                   # bfcl-eval MAXIMUM_STEP_LIMIT
    schema_version: int = SCHEMA_VERSION

    def content(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def run_id(self) -> str:
        """12 hex chars of SHA-256 over the canonical JSON of every field."""
        return sha256_text(canonical_json(self.content()))[:12]

    @property
    def config_key(self) -> str:
        """Hash of everything except task and seed: groups the episodes of one configuration."""
        c = self.content()
        for k in ("task_id", "seed"):
            c.pop(k)
        return sha256_text(canonical_json(c))[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class GenerateResult:
    """Return value of C3 ``Backend.generate``."""

    text: str                         # raw completion text, think block removed
    raw_text: str                     # completion exactly as returned
    tool_calls: list[ToolCall]        # parsed <tool_call> blocks that are valid JSON objects
    tool_call_blocks: int             # number of <tool_call> blocks seen, parsed or not
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: Optional[float]          # llama-server timings.prompt_ms
    decode_ms: Optional[float]        # llama-server timings.predicted_ms
    latency_ms: float                 # wall clock for the HTTP call
    stop_reason: str                  # "eos" | "limit" | "word" | ...
    grammar_used: bool
    bad_call_blocks: int = 0          # blocks that parse as JSON but are not {"name", "arguments"} objects
    n_predict: int = 0                # token budget given to this request


@dataclass
class StepRow:
    """One model step inside one turn of one episode (C5 row; C6 fills ``failure_class``)."""

    run_id: str
    config_key: str
    task_id: str
    seed: int
    turn: int
    step: int
    variant: str
    backend: str
    repair: str
    constraint: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: Optional[float]
    decode_ms: Optional[float]
    latency_ms: float
    stop_reason: str
    raw_text: str
    n_tool_call_blocks: int
    decoded_calls: list[str]          # python call strings handed to the bfcl executor
    execution_results: list[str]
    known_tools: bool                 # every call names a tool in the task's tool list
    schema_errors: list[str]          # per-call schema problems (missing required, unknown arg, type)
    turn_has_ground_truth: bool       # ground truth for this turn is non-empty
    is_last_step_of_turn: bool
    forced_quit: bool
    failure_class: str = ""           # filled by C6
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class EpisodeResult:
    run_id: str
    config_key: str
    task_id: str
    seed: int
    variant: str
    backend: str
    repair: str
    constraint: str
    valid: bool                       # bfcl multi_turn_checker verdict
    error_type: str
    n_turns: int
    n_steps: int
    forced_quit: bool
    wall_s: float
    context_overflow: bool = False    # the prompt outgrew the context window; scored as a failure
    energy_j: Optional[float] = None  # C9, CUDA only
    peak_gpu_mib: Optional[float] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
