"""scripts/evaluator_agreement.py, decision 17: the FR-E2 replay counts bad tool-call blocks exactly as the runner."""
import importlib.util
import os

import pytest

from testbed import prompt_formats
from testbed.backends import parse_completion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("evaluator_agreement",
                                              os.path.join(ROOT, "scripts", "evaluator_agreement.py"))
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)

CASES = [
    (prompt_formats.QWEN3, '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>\n'
                           '<tool_call>\n{"foo": 1}\n</tool_call>', 1),
    (prompt_formats.QWEN3, '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>', 0),
    (prompt_formats.SMOLLM3, '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>', 0),
    # SmolLM3 multi_turn_base_139 seed 3: the earlier tool result copied as a call
    (prompt_formats.SMOLLM3, '<tool_call>\n{"current_working_directory": "alex"}\n</tool_call>', 1),
    (prompt_formats.LFM25_1P2B, "<|tool_call_start|>[cd(folder='temp')]<|tool_call_end|>", 0),
    (prompt_formats.QWEN3, "no call here", 0),
]


@pytest.mark.parametrize("fmt,content,bad", CASES)
def test_replay_counts_bad_blocks_like_the_runner(fmt, content, bad):
    g = parse_completion({"content": content}, n_prompt=5, budget=50, constraint=None, latency_ms=1.0, fmt=fmt)
    assert g.bad_call_blocks == bad
    assert ea.bad_call_blocks(fmt, g.raw_text) == g.bad_call_blocks       # the replay reads the logged raw text


def test_crash_entry_is_not_a_list():
    # bfcl's _evaluate_single_multi_turn_entry scores any non-list model result as multi_turn:inference_error
    assert isinstance(ea.INFERENCE_CRASH, str)
