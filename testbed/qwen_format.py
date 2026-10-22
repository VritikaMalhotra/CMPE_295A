"""Qwen3 function-calling prompt format, ported from bfcl-eval 2026.3.23
(bfcl_eval/model_handler/local_inference/qwen_fc.py, QwenFCHandler._format_prompt and
_extract_tool_calls) so the testbed prompts are byte-identical to the Oct 8 pilot's.

Non-thinking mode appends the empty think block, as the pilot's v2 harness did.
"""
from __future__ import annotations

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
NON_THINKING_SUFFIX = "<think>\n\n</think>\n\n"


def format_prompt(messages: list[dict], tools: list[dict], thinking: bool = False) -> str:
    out = ""
    if tools:
        out += "<|im_start|>system\n"
        if messages and messages[0]["role"] == "system":
            out += messages[0]["content"] + "\n\n"
        out += ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
        for tool in tools:
            out += f"\n{json.dumps(tool)}"
        out += ('\n</tools>\n\nFor each function call, return a json object with function name and arguments '
                'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
                '<args-json-object>}\n</tool_call><|im_end|>\n')
    elif messages and messages[0]["role"] == "system":
        out += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

    last_query_index = len(messages) - 1
    for offset, m in enumerate(reversed(messages)):
        idx = len(messages) - 1 - offset
        c = m.get("content")
        if m["role"] == "user" and isinstance(c, str) and not (
            c.startswith("<tool_response>") and c.endswith("</tool_response>")
        ):
            last_query_index = idx
            break

    for idx, m in enumerate(messages):
        role, content = m["role"], m.get("content", "")
        if role == "user" or (role == "system" and idx != 0):
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif role == "assistant":
            reasoning = m.get("reasoning_content") or ""
            if not reasoning and "</think>" in content:
                parts = content.split("</think>")
                reasoning = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = parts[-1].lstrip("\n")
            if idx > last_query_index:
                if idx == len(messages) - 1 or reasoning:
                    out += (f"<|im_start|>{role}\n<think>\n" + reasoning.strip("\n") + "\n</think>\n\n"
                            + content.lstrip("\n"))
                else:
                    out += f"<|im_start|>{role}\n{content}"
            else:
                out += f"<|im_start|>{role}\n{content}"
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                # bfcl compares by value (tool_call == tool_calls[0]), so a repeat of the first call
                # gets no newline either; copied on purpose to keep prompts byte-identical
                if (tc == tcs[0] and content) or tc != tcs[0]:
                    out += "\n"
                tc = tc.get("function", tc)
                args = tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"])
                out += '<tool_call>\n{"name": "' + tc["name"] + '", "arguments": ' + args + "}\n</tool_call>"
            out += "<|im_end|>\n"
        elif role == "tool":
            prev_role = messages[idx - 1]["role"] if idx > 0 else None
            next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None
            if idx == 0 or prev_role != "tool":
                out += "<|im_start|>user"
            out += f"\n<tool_response>\n{content}\n</tool_response>"
            if idx == len(messages) - 1 or next_role != "tool":
                out += "<|im_end|>\n"

    out += "<|im_start|>assistant\n"
    if not thinking:
        out += NON_THINKING_SUFFIX
    return out


def split_think(text: str) -> tuple[str, str]:
    """(reasoning, cleaned) exactly as bfcl's _parse_query_response_prompting does."""
    if "</think>" in text:
        parts = text.split("</think>")
        return parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n"), parts[-1].lstrip("\n")
    return "", text


def extract_tool_calls(text: str) -> tuple[list[dict], int]:
    """Every <tool_call> block that parses as JSON (as bfcl keeps them) and the number of blocks seen."""
    blocks = TOOL_CALL_RE.findall(text)
    calls = []
    for b in blocks:
        try:
            calls.append(json.loads(b))   # as bfcl: every block that parses, dict or not
        except Exception:
            continue
    loose = text.count("<tool_call>")
    return calls, max(len(blocks), loose)


def to_python_calls(calls: list[dict]) -> list[str]:
    """bfcl's convert_to_function_call: {name, arguments} -> "name(k=repr(v),...)"."""
    out = []
    for c in calls:
        args = c.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        out.append(f"{c['name']}({','.join(f'{k}={v!r}' for k, v in args.items())})")
    return out
