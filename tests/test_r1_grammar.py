"""R1: the free-text-branch grammar's structure (testbed/grammar.py)."""
from testbed.grammar import task_grammar, task_grammar_free

TOOLS = [{"name": "cd", "description": "Change directory.",
          "parameters": {"type": "dict", "properties": {"folder": {"type": "string", "description": "x"}},
                         "required": ["folder"]}}]


def test_r1_grammar_has_free_branch():
    strict, free = task_grammar(TOOLS), task_grammar_free(TOOLS)
    assert strict.startswith("root ::= block")
    assert "root ::= calls | free" in free and "free ::=" in free and '"\\"cd\\""' in free


def test_grammar_table_per_call_syntax():
    """Qwen3/SmolLM3 keep the JSON R1 grammars exactly; LFM2.5 gets the Pythonic ones."""
    from testbed.grammar import grammars_for, task_grammar, task_grammar_free
    j = grammars_for("json")
    assert j["free"] is task_grammar_free and j["forced"] is task_grammar and j["none"] is None
    fns = [{"name": "ls", "parameters": {"type": "dict", "properties": {"a": {"type": "boolean"}}, "required": []}}]
    p = grammars_for("pythonic")
    assert p["forced"](fns).startswith("root ::= block\nblock ::= \"<|tool_call_start|>[\"")
    assert p["free"](fns).startswith("root ::= (block | free)")
    assert grammars_for("pythonic", think_prefix=True)["free"](fns).startswith("root ::= think (block | free)")
