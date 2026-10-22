"""Qwen3 non-thinking handler (prompt format + tool-call parsing): unit checks.
Byte-level parity with bfcl's QwenFC formatter is in test_bfcl_parity.py."""
from testbed.qwen_format import extract_tool_calls, format_prompt, split_think, to_python_calls

TOOLS = [{"name": "cd", "description": "Change directory.",
          "parameters": {"type": "dict", "properties": {"folder": {"type": "string", "description": "x"}},
                         "required": ["folder"]}}]


def test_extract_and_python_calls():
    text = '<tool_call>\n{"name": "cd", "arguments": {"folder": "docs"}}\n</tool_call>\n' \
           '<tool_call>\n{"name": "cd", "arguments": {broken}\n</tool_call>'
    calls, n = extract_tool_calls(text)
    assert n == 2 and len(calls) == 1
    assert to_python_calls(calls) == ["cd(folder='docs')"]


def test_non_dict_block_is_kept_as_bfcl_does():
    calls, n = extract_tool_calls('<tool_call>\n[1, 2]\n</tool_call>')
    assert n == 1 and calls == [[1, 2]]          # the runner turns this into bfcl's inference_error


def test_prompt_is_non_thinking_and_lists_tools():
    p = format_prompt([{"role": "user", "content": "go to docs"}], TOOLS, thinking=False)
    assert '"name": "cd"' in p and p.endswith("<think>\n\n</think>\n\n")
    assert split_think("<think>\nhm\n</think>\n\nhi") == ("hm", "hi")
