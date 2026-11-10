"""Prompt format per model family: the model handler the backend and the parsers use (C3).

Chosen from ``RunSpec.model`` (already part of every run id), so adding a family changes no existing run id or
prompt: every Qwen3 model still goes through ``qwen_format`` exactly as before.

| Model key prefix | Module | Tool-call format | Source of the format |
|---|---|---|---|
| ``qwen3-`` | ``qwen_format`` | ``<tool_call>{json}</tool_call>`` | bfcl-eval 2026.3.23 ``QwenFCHandler`` |
| ``smollm3-`` | ``smollm3_format`` | ``<tool_call>{json}</tool_call>`` | the model's chat template (bfcl has no handler) |
| ``lfm2.5-1.2b`` / ``lfm2.5-2.6b`` | ``lfm2_format`` | Pythonic ``<|tool_call_start|>[f(a=1)]<|tool_call_end|>`` | the model's chat template (bfcl has no handler) |

Each format offers ``format_prompt(messages, tools, thinking)``, ``split_think(text)`` and
``extract_tool_calls(text) -> (calls, n_blocks)``, plus ``call_markers`` (the strings that mean "the model
tried to call a tool", used by the classifier's ``malformed`` rule) and ``syntax`` ("json" or "pythonic",
which picks the R1 grammar).

Two transport details differ from Qwen3 (whose path is unchanged):
- ``send_tokens``: the backend sends llama-server the token ids of the rendered string (``/tokenize`` without
  added special tokens) instead of the string, so the model sees exactly the template's tokens. Otherwise the
  server's own BOS rule would decide: LFM2.5-1.2B's GGUF adds a BOS, LFM2.5-2.6B's does not, and SmolLM3's
  template has none.
- ``raw_prefix``: text the prompt already opened for the answer (LFM2.5-2.6B: ``<think>``). It is put back in
  front of the generated text, so the logged ``raw_text`` is the whole assistant turn and the classifier's
  think-block rule sees both tags.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from . import lfm2_format, qwen_format, smollm3_format


@dataclass(frozen=True)
class PromptFormat:
    name: str
    format_prompt: Callable[..., str]
    split_think: Callable[[str], tuple[str, str]]
    extract_tool_calls: Callable[[str], tuple[list, int]]
    call_markers: tuple[str, ...]
    syntax: str
    send_tokens: bool = False
    raw_prefix: str = ""


QWEN3 = PromptFormat("qwen3", qwen_format.format_prompt, qwen_format.split_think, qwen_format.extract_tool_calls,
                     ("<tool_call>",), "json")
SMOLLM3 = PromptFormat("smollm3", smollm3_format.format_prompt, smollm3_format.split_think,
                       smollm3_format.extract_tool_calls, ("<tool_call>",), "json", send_tokens=True)
LFM25_1P2B = PromptFormat("lfm2.5-1.2b", partial(lfm2_format.format_prompt, size="1.2b"), lfm2_format.split_think,
                          lfm2_format.extract_tool_calls, (lfm2_format.CALL_START,), "pythonic", send_tokens=True)
LFM25_2P6B = PromptFormat("lfm2.5-2.6b", partial(lfm2_format.format_prompt, size="2.6b"), lfm2_format.split_think,
                          lfm2_format.extract_tool_calls, (lfm2_format.CALL_START,), "pythonic",
                          send_tokens=True, raw_prefix="<think>")

ALL_CALL_MARKERS = ("<tool_call>", lfm2_format.CALL_START)


def for_model(model: str) -> PromptFormat:
    """The format for a model key such as ``qwen3-0.6b``, ``smollm3-3b`` or ``lfm2.5-1.2b-instruct``."""
    m = (model or "").lower()
    if m.startswith("qwen3"):
        return QWEN3
    if m.startswith("smollm3"):
        return SMOLLM3
    if m.startswith("lfm2.5-1.2b"):
        return LFM25_1P2B
    if m.startswith("lfm2.5-2.6b"):
        return LFM25_2P6B
    raise ValueError(f"no prompt format for model {model!r} (known: qwen3-*, smollm3-*, lfm2.5-1.2b*, lfm2.5-2.6b*)")
