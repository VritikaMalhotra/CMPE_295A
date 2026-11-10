"""SmolLM3-3B (Hugging Face) function-calling prompt format.

bfcl-eval 2026.3.23 has no handler for SmolLM3, so this follows the model's own ``chat_template.jinja``
(checkpoint in ``$MODEL_SRC_DIR/SmolLM3-3B``; revision in ``results/m2/models_manifest.json``), written the same
way as ``qwen_format`` and checked byte for byte against the template in ``tests/test_prompt_formats.py``.

What the template does, and the three places where it leaves a choice to the caller:
- The system turn is a metadata header (knowledge cutoff, **today's date**, reasoning mode), the custom
  instructions (the system message, or SmolLM's default line), then the tools in ``<tools></tools>`` and the
  ``<tool_call>{"name": …, "arguments": …}</tool_call>`` instruction. Tools are printed with Jinja's ``string``
  filter, i.e. Python ``repr`` of each dict, not JSON.
- Non-thinking mode (``/no_think``): every assistant turn and the generation prompt carry an empty
  ``<think>\\n\\n</think>\\n``.
- Tool results become **user** turns with the result text as is.

Choices (recorded; the advisor can overrule):
1. **Date.** The template prints ``strftime_now("%d %B %Y")``, so the prompt would change every day. The testbed
   pins it to ``TODAY`` (the day the references were frozen), so prompts and run ids are reproducible.
2. **Tool calls in the history.** The template prints only an assistant message's ``content`` and ignores
   ``tool_calls``. A call the model made is therefore written back as the text the system prompt asks for:
   ``<tool_call>\\n{"name": …, "arguments": …}\\n</tool_call>``, one block per call, joined by a newline.
3. **Parsing** accepts optional whitespace inside ``<tool_call>`` (the Qwen parser requires exact newlines).
"""
from __future__ import annotations

import json
import re

TODAY = "10 November 2026"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
EMPTY_THINK = "<think>\n\n</think>\n"
DEFAULT_INSTRUCTIONS = "You are a helpful AI assistant named SmolLM, trained by Hugging Face."
THINK_INSTRUCTIONS = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves "
    "thoroughly exploring questions through a systematic thinking process before providing the final precise and "
    "accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, "
    "reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please "
    "structure your response into two main sections: Thought and Solution using the specified format: <think> "
    "Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. "
    "Each step should include detailed considerations such as analysing questions, summarizing relevant findings, "
    "brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting "
    "previous steps. In the Solution section, based on various attempts, explorations, and reflections from the "
    "Thought section, systematically present the final solution that you deem correct. The Solution section "
    "should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.")
TOOLS_HEAD = ("You may call one or more functions to assist with the user query.\nYou are provided with function "
              "signatures within <tools></tools> XML tags:\n\n<tools>\n")
TOOLS_TAIL = ("</tools>\n\nFor each function call, return a json object with function name and arguments within "
              "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
              "<args-json-object>}\n</tool_call>")


def call_text(tool_calls: list[dict]) -> str:
    """Choice 2: a call written back into the history in the format the system prompt asks for."""
    out = []
    for tc in tool_calls:
        f = tc.get("function", tc)
        args = f["arguments"] if isinstance(f["arguments"], str) else json.dumps(f["arguments"])
        out.append('<tool_call>\n{"name": "' + f["name"] + '", "arguments": ' + args + "}\n</tool_call>")
    return "\n".join(out)


def format_prompt(messages: list[dict], tools: list[dict], thinking: bool = False, today: str = TODAY) -> str:
    mode = "/think" if thinking else "/no_think"
    out = "<|im_start|>system\n"
    custom = None
    sysmsg = None
    if messages and messages[0]["role"] == "system":
        sysmsg = messages[0].get("content") or ""
        if "/no_think" in sysmsg:
            mode = "/no_think"
        elif "/think" in sysmsg:
            mode = "/think"
        custom = sysmsg.replace("/no_think", "").replace("/think", "").rstrip()
    if sysmsg is not None and "/system_override" in sysmsg:
        out += custom.replace("/system_override", "").rstrip() + "<|im_end|>\n"
    else:
        out += "## Metadata\n\nKnowledge Cutoff Date: June 2025\n"
        out += f"Today Date: {today}\n"
        out += f"Reasoning Mode: {mode}\n\n"
        out += "## Custom Instructions\n\n"
        if custom:
            out += custom + "\n\n"
        elif mode == "/think":
            out += THINK_INSTRUCTIONS + "\n\n"
        else:
            out += DEFAULT_INSTRUCTIONS + "\n\n"
        if tools:
            out += "### Tools\n\n" + TOOLS_HEAD + "".join(str(t) + "\n" for t in tools) + TOOLS_TAIL
            out += "\n\n<|im_end|>\n"
    # the template's main loop renders every message, the system message included only as a header above
    for m in messages:
        content = m.get("content") if isinstance(m.get("content"), str) else ""
        if m["role"] == "user":
            out += "<|im_start|>user\n" + content + "<|im_end|>\n"
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                content = (content + "\n" if content else "") + call_text(m["tool_calls"])
            if mode == "/think":
                out += "<|im_start|>assistant\n" + content.lstrip("\n") + "<|im_end|>\n"
            else:
                out += "<|im_start|>assistant\n" + EMPTY_THINK + content.lstrip("\n") + "<|im_end|>\n"
        elif m["role"] == "tool":
            out += "<|im_start|>user\n" + content + "<|im_end|>\n"
    out += "<|im_start|>assistant\n" + ("" if mode == "/think" else EMPTY_THINK)
    return out


def split_think(text: str) -> tuple[str, str]:
    if "</think>" in text:
        parts = text.split("</think>")
        return parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n"), parts[-1].lstrip("\n")
    return "", text


def extract_tool_calls(text: str) -> tuple[list[dict], int]:
    """Every <tool_call> block that parses as JSON and the number of blocks seen (as qwen_format)."""
    blocks = TOOL_CALL_RE.findall(text)
    calls = []
    for b in blocks:
        try:
            calls.append(json.loads(b))
        except Exception:
            continue
    return calls, max(len(blocks), text.count("<tool_call>"))
