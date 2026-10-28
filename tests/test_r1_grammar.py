"""R1: the free-text-branch grammar's structure (testbed/grammar.py)."""
from testbed.grammar import task_grammar, task_grammar_free

TOOLS = [{"name": "cd", "description": "Change directory.",
          "parameters": {"type": "dict", "properties": {"folder": {"type": "string", "description": "x"}},
                         "required": ["folder"]}}]


def test_r1_grammar_has_free_branch():
    strict, free = task_grammar(TOOLS), task_grammar_free(TOOLS)
    assert strict.startswith("root ::= block")
    assert "root ::= calls | free" in free and "free ::=" in free and '"\\"cd\\""' in free
