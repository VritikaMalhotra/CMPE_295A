"""LFM2.5 (Liquid AI) function-calling prompt format: LFM2.5-1.2B-Instruct and LFM2.5-2.6B.

bfcl-eval 2026.3.23 has no handler for LFM2.5, so this follows the models' own ``chat_template.jinja``
(Hugging Face checkpoints in ``$MODEL_SRC_DIR``; revisions in ``results/m2/models_manifest.json``), written
the same way as ``qwen_format``: a plain-Python rendering of the template, checked byte for byte against the
template in ``tests/test_prompt_formats.py`` (golden strings from ``scripts/render_chat_template.py``).

Format (both sizes):
- system turn: the system message, then ``List of tools: [<tool json>, <tool json>]`` (``tojson``, so
  non-ASCII stays as is);
- assistant tool calls are **Pythonic**: ``<|tool_call_start|>[name(a='x', b=1)]<|tool_call_end|>``;
- tool results are their own ``<|im_start|>tool`` turns, content as is.

The two sizes differ in thinking:
- 1.2B-Instruct keeps a ``<think>`` block only on the last assistant message and opens the answer with a plain
  ``<|im_start|>assistant\\n``;
- 2.6B keeps thinking on every assistant message after the last user message and **always** opens the answer
  with ``<think>``: the template has no non-thinking mode, so ``thinking: false`` cannot switch it off. The
  testbed follows the template (open point for the advisor, docs/m2_decisions.md).

The string starts with the BOS token ``<|startoftext|>``, as the template writes it. The two GGUF files disagree
on adding a BOS themselves (``tokenizer.ggml.add_bos_token`` true for 1.2B, false for 2.6B), so the backend
sends the token ids of this string (``PromptFormat.send_tokens``) and the server adds nothing.
"""
from __future__ import annotations

import ast
import json
import re

BOS = "<|startoftext|>"
CALL_START, CALL_END = "<|tool_call_start|>", "<|tool_call_end|>"
PRESERVE = ("<|tool_call_start|>", "<|tool_call_end|>", "<think>", "</think>")   # kept in the server's output
TOOL_CALL_RE = re.compile(re.escape(CALL_START) + r"(.*?)" + re.escape(CALL_END), re.DOTALL)
VARIANTS = ("lfm2.5-1.2b", "lfm2.5-2.6b")


def _tojson(x) -> str:
    """Hugging Face's ``tojson`` filter: json.dumps with ensure_ascii=False and default separators."""
    return json.dumps(x, ensure_ascii=False)


def _arg_value(v) -> str:
    """The template's ``format_arg_value``: quoted, escaped strings; JSON for lists/dicts; str() otherwise."""
    if isinstance(v, str):
        return "'" + (v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")) + "'"
    if isinstance(v, (dict, list, tuple)):
        return _tojson(v)
    return str(v)


def render_tool_calls(tool_calls: list[dict]) -> str:
    parts = []
    for tc in tool_calls:
        f = tc.get("function", tc)
        args = f.get("arguments")
        if isinstance(args, str):
            if args.strip() not in ("", "{}", "null"):
                raise ValueError("tool call arguments must be a mapping, not a JSON string")
            args = {}
        parts.append(f["name"] + "(" + ", ".join(f"{k}={_arg_value(v)}" for k, v in (args or {}).items()) + ")")
    return CALL_START + "[" + ", ".join(parts) + "]" + CALL_END


def _content(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return _tojson(c)
    out = ""
    for item in c:
        if isinstance(item, str):
            out += item
        elif isinstance(item, dict) and item.get("type") == "text":
            out += str(item.get("text") or "")
        else:
            out += _tojson(item)
    return out


def format_prompt(messages: list[dict], tools: list[dict], thinking: bool = False, size: str = "1.2b",
                  preserve_thinking: bool = False) -> str:
    """``size`` is "1.2b" (LFM2.5-1.2B-Instruct) or "2.6b" (LFM2.5-2.6B); ``thinking`` has no effect on
    either template (see the module docstring)."""
    if size not in ("1.2b", "2.6b"):
        raise ValueError(f"unknown LFM2.5 size {size!r}")
    system = ""
    if messages and messages[0]["role"] == "system":
        if messages[0].get("content"):
            system = _content(messages[0]["content"])
        messages = messages[1:]
    if tools:
        system += ("\n" if system else "") + "List of tools: [" + ", ".join(
            t if isinstance(t, str) else _tojson(t) for t in tools) + "]"
    out = BOS + (f"<|im_start|>system\n{system}<|im_end|>\n" if system else "")
    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    for i, m in enumerate(messages):
        out += f"<|im_start|>{m['role']}\n"
        if m["role"] == "assistant":
            keep = preserve_thinking or (i == last_assistant if size == "1.2b" else i > last_user)
            th = m.get("thinking") or m.get("reasoning") or m.get("reasoning_content")
            th = th if isinstance(th, str) else ""
            if th and keep:
                out += "<think>" + th + "</think>"
            content = _content(m.get("content")) if m.get("content") else ""
            if not keep and "</think>" in content:
                content = content.split("</think>")[-1].strip()
            out += content
            if m.get("tool_calls"):
                out += render_tool_calls(m["tool_calls"])
            out += "<|im_end|>\n"
        else:
            if m.get("content"):
                out += _content(m["content"])
            out += "<|im_end|>\n"
    out += "<|im_start|>assistant\n" + ("<think>" if size == "2.6b" else "")
    return out


def split_think(text: str) -> tuple[str, str]:
    """(reasoning, answer). The 2.6B answer starts inside the ``<think>`` the prompt opened, so its raw text
    has a closing ``</think>`` only; the split is the same as ``qwen_format.split_think``."""
    if "</think>" in text:
        parts = text.split("</think>")
        return parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n"), parts[-1].lstrip("\n")
    return "", text


_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _text(s: str) -> str:
    """Python string syntax keeps the escapes ``\\ud83d\\ude80`` as two lone surrogates, which UTF-8 cannot encode.
    A pair is read as the character it encodes, as JSON reads it; an unpaired surrogate is not text, so it is an
    error and the block does not parse."""
    if not any("\ud800" <= c <= "\udfff" for c in s):
        return s
    try:
        return s.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError as e:
        raise ValueError("unpaired surrogate in a string argument") from e


def _literal(node):
    """ast.literal_eval plus JSON's true/false/null (the template writes list and dict arguments as JSON)."""
    if isinstance(node, ast.Constant):
        return _text(node.value) if isinstance(node.value, str) else node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _literal(node.operand)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError("bad unary operand")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("dict unpacking")
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"not a literal: {ast.dump(node)[:60]}")


def _call_name(f) -> str:
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return _call_name(f.value) + "." + f.attr
    raise ValueError("call target is not a name")


_NUL_RE = re.compile(r"(?<!\\)((?:\\\\)*)\x00")    # a NUL after an even number of backslashes


def parse_calls(block: str) -> list[dict]:
    """``[f(a='x'), g()]`` (or a single ``f(a='x')``) -> [{"name", "arguments"}]. Keyword arguments only:
    a positional argument cannot be mapped to a parameter name, so the block does not parse.

    Python's parser rejects a NUL byte anywhere in the source, even inside a string literal, where the R1 grammar
    (and free decoding) allow it as an ordinary character. A NUL that no backslash escapes is therefore written as
    the escape ``\\x00`` first, which reads back as the same character; outside a string it still fails to parse."""
    tree = ast.parse(_NUL_RE.sub(lambda m: m.group(1) + "\\x00", block.strip()), mode="eval").body
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls = []
    for n in nodes:
        if not isinstance(n, ast.Call) or n.args or any(k.arg is None for k in n.keywords):
            raise ValueError("not a keyword-only call")
        calls.append({"name": _call_name(n.func), "arguments": {k.arg: _literal(k.value) for k in n.keywords}})
    return calls


def extract_tool_calls(text: str) -> tuple[list[dict], int]:
    """Every call in every ``<|tool_call_start|>…<|tool_call_end|>`` block that parses, and a block count in
    the units the classifier compares with the parsed calls: a block that parses counts its calls, a block that
    does not (or an unclosed start marker) counts 1. So ``count > len(calls)`` exactly when something did not
    parse, as with ``qwen_format.extract_tool_calls``."""
    calls: list[dict] = []
    count = 0
    blocks = TOOL_CALL_RE.findall(text)
    for b in blocks:
        try:
            got = parse_calls(b)
        except (SyntaxError, ValueError, RecursionError):
            count += 1
            continue
        calls += got
        count += max(len(got), 1)
    count += max(0, text.count(CALL_START) - len(blocks))
    return calls, count
