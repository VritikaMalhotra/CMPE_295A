#!/usr/bin/env python
"""Golden prompts for tests/test_prompt_formats.py, rendered from each model's own chat_template.jinja.

The rendering environment is the one Hugging Face ``apply_chat_template`` builds
(transformers ``chat_template_utils._compile_jinja_template``): an immutable sandbox with trim_blocks and
lstrip_blocks, the loopcontrols extension, a ``{% generation %}`` tag that renders its body, a ``tojson`` filter
that keeps non-ASCII text, and the globals ``raise_exception`` and ``strftime_now`` (pinned to the date the
testbed pins, see testbed/smollm3_format.py). With ``--check-transformers`` it also renders every case through
``transformers.PreTrainedTokenizerFast`` from the checkpoint directory and asserts the two agree.

    python scripts/render_chat_template.py --smollm3 $MODEL_SRC_DIR/SmolLM3-3B \\
        --lfm-1p2b $MODEL_SRC_DIR/LFM2.5-1.2B-Instruct --lfm-2p6b $MODEL_SRC_DIR/LFM2.5-2.6B \\
        --out tests/golden/prompt_formats.json [--check-transformers]

Each argument is a checkpoint directory (``chat_template.jinja`` + ``tokenizer_config.json``) or a bare
``.jinja`` file (then the BOS token is given with ``--lfm-bos``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testbed import smollm3_format  # noqa: E402

TODAY = smollm3_format.TODAY


class _Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def _env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_Generation, loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise ValueError(msg)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: TODAY      # every case uses "%d %B %Y"
    return env


def _load(path: str, bos: str | None) -> tuple[str, str]:
    if os.path.isdir(path):
        tmpl = open(os.path.join(path, "chat_template.jinja")).read()
        tc = json.load(open(os.path.join(path, "tokenizer_config.json")))
        b = tc.get("bos_token")
        bos = b.get("content") if isinstance(b, dict) else (b or bos or "")
    else:
        tmpl = open(path).read()
    return tmpl, bos or ""


TOOLS_FS = [
    {"type": "function", "function": {"name": "ls", "description": "List the files in the current directory.",
                                      "parameters": {"type": "object", "properties": {
                                          "a": {"type": "boolean", "description": "Show hidden files."}},
                                          "required": []}}},
    {"type": "function", "function": {"name": "mv", "description": "Move a file to a folder — keeps its name.",
                                      "parameters": {"type": "object", "properties": {
                                          "source": {"type": "string"}, "destination": {"type": "string"}},
                                          "required": ["source", "destination"]}}},
]
TOOLS_AIR = [
    {"type": "function", "function": {"name": "update_reservation_baggages", "description": "Update bags.",
                                      "parameters": {"type": "object", "properties": {
                                          "reservation_id": {"type": "string"}, "total_baggages": {"type": "integer"},
                                          "nonfree_baggages": {"type": "integer"},
                                          "payment": {"type": "object"}},
                                          "required": ["reservation_id"]}}},
]

CASES = {
    # BFCL-shaped: no system message, two turns, one and two calls per step, tool results, a closing reply
    "bfcl_two_turns": {
        "tools": TOOLS_FS,
        "messages": [
            {"role": "user", "content": "Move final_report.pdf into temp."},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "ls", "arguments": {"a": True}}],
             "reasoning_content": ""},
            {"role": "tool", "name": "ls(a=True)", "content": '{"current_directory_content": ["final_report.pdf", "temp"]}'},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "mv", "arguments": {"source": "final_report.pdf", "destination": "temp"}},
                {"name": "ls", "arguments": {}}], "reasoning_content": ""},
            {"role": "tool", "name": "mv(...)", "content": "None"},
            {"role": "tool", "name": "ls()", "content": '{"current_directory_content": ["temp"]}'},
            {"role": "assistant", "content": "I moved the file into temp.", "reasoning_content": ""},
            {"role": "user", "content": "Now list what's in temp, d'accord?"},
        ],
    },
    # tau2-shaped: a policy system message, a text reply, a call with nested, numeric, boolean and quoted values
    "tau2_policy": {
        "tools": TOOLS_AIR,
        "messages": [
            {"role": "system", "content": "# Airline policy\nYou help users change bookings.\n- Be precise."},
            {"role": "user", "content": "Hi, I want to add a bag."},
            {"role": "assistant", "content": "Could you give me your reservation id?", "reasoning_content": ""},
            {"role": "user", "content": "It's ZFA04Y. Pay with \"gift_card_1\".\nThanks"},
            {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [
                {"name": "update_reservation_baggages", "arguments": {
                    "reservation_id": "ZFA04Y", "total_baggages": 2, "nonfree_baggages": 0,
                    "payment": {"payment_id": "gift_card_1", "amount": 0.0, "refund": False, "note": None},
                    "comment": "it's \"fine\"\nok"}}]},
            {"role": "tool", "content": '{"reservation_id": "ZFA04Y", "status": "updated"}'},
        ],
    },
    # thinking kept or dropped by position (LFM2.5 1.2B: last assistant only; 2.6B: after the last user turn)
    "with_reasoning": {
        "tools": TOOLS_FS,
        "messages": [
            {"role": "user", "content": "Show hidden files."},
            {"role": "assistant", "content": "", "reasoning_content": "The user wants ls -a.",
             "tool_calls": [{"name": "ls", "arguments": {"a": True}}]},
            {"role": "tool", "content": '{"current_directory_content": [".env"]}'},
            {"role": "assistant", "content": "There is one hidden file, .env.", "reasoning_content": "One file."},
            {"role": "user", "content": "Thanks. Anything else?"},
            {"role": "assistant", "content": "", "reasoning_content": "Check again.",
             "tool_calls": [{"name": "ls", "arguments": {}}]},
            {"role": "tool", "content": '{"current_directory_content": []}'},
        ],
    },
}


def _template_messages(fmt: str, messages: list[dict]) -> list[dict]:
    """The message list the template sees. SmolLM3's template ignores ``tool_calls``, so a call goes into the
    content as the testbed writes it (smollm3_format choice 2)."""
    out = copy.deepcopy(messages)
    if fmt == "smollm3":
        for m in out:
            if m["role"] == "assistant" and m.get("tool_calls"):
                m["content"] = ((m.get("content") or "") + "\n" if m.get("content") else "") + \
                    smollm3_format.call_text(m.pop("tool_calls"))
    return out


def render(fmt: str, tmpl: str, bos: str, case: dict) -> str:
    kwargs = dict(messages=_template_messages(fmt, case["messages"]), tools=case["tools"],
                  add_generation_prompt=True, bos_token=bos, eos_token="<|im_end|>")
    if fmt == "smollm3":
        kwargs["enable_thinking"] = False
    text = _env().from_string(tmpl).render(**kwargs)
    if fmt.startswith("lfm"):
        assert text.startswith(bos), "LFM2.5 template no longer starts with the BOS token"
    return text


def check_transformers(fmt: str, path: str, case: dict, expected: str, bos: str) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    import transformers.utils.chat_template_utils as ctu      # pin the date the same way
    kw = {"enable_thinking": False} if fmt == "smollm3" else {}
    orig = ctu.strftime_now if hasattr(ctu, "strftime_now") else None
    if orig:
        ctu.strftime_now = lambda fmt_: TODAY
    try:
        got = tok.apply_chat_template(_template_messages(fmt, case["messages"]), tools=case["tools"],
                                      add_generation_prompt=True, tokenize=False, **kw)
    finally:
        if orig:
            ctu.strftime_now = orig
    if fmt == "smollm3" and not orig:
        print("note: transformers has no strftime_now hook; SmolLM3 date line not pinned in this check")
        return
    assert got == expected, f"{fmt}: transformers rendering differs from the sandbox rendering"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smollm3", required=True)
    ap.add_argument("--lfm-1p2b", required=True)
    ap.add_argument("--lfm-2p6b", required=True)
    ap.add_argument("--lfm-bos", default="<|startoftext|>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-transformers", action="store_true")
    a = ap.parse_args()
    out = {"today": TODAY, "templates": {}, "cases": {}}
    for fmt, path in (("smollm3", a.smollm3), ("lfm2.5-1.2b", a.lfm_1p2b), ("lfm2.5-2.6b", a.lfm_2p6b)):
        tmpl, bos = _load(path, a.lfm_bos if fmt.startswith("lfm") else None)
        out["templates"][fmt] = {"sha256": hashlib.sha256(tmpl.encode()).hexdigest(), "bos_token": bos}
        for name, case in CASES.items():
            text = render(fmt, tmpl, bos, case)
            if a.check_transformers and os.path.isdir(path):
                check_transformers(fmt, path, case, text, bos)
            out["cases"].setdefault(name, {"messages": case["messages"], "tools": case["tools"], "expected": {}})
            out["cases"][name]["expected"][fmt] = text
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {a.out}: {len(CASES)} cases x 3 formats")


if __name__ == "__main__":
    main()
