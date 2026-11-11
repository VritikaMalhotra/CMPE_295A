"""R1 property test (design spec 7.2): llama.cpp's own grammar engine accepts every schema-valid call and
rejects calls with a wrong name, a missing required key or an extra key; the free-text branch accepts text
not starting with "<" or whitespace and rejects text starting with "<".

Needs llama.cpp's test-gbnf-validator at the pinned commit (slurm/build_gbnf_validator.sbatch) in
$LLAMA_GBNF_VALIDATOR; skipped otherwise.
"""
import json
import os
import subprocess
import tempfile

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from testbed.grammar import task_grammar, task_grammar_free  # noqa: E402

VALIDATOR = os.environ.get("LLAMA_GBNF_VALIDATOR", "")
pytestmark = pytest.mark.skipif(not (VALIDATOR and os.access(VALIDATOR, os.X_OK)),
                                reason="set LLAMA_GBNF_VALIDATOR to llama.cpp's test-gbnf-validator")

# BFCL-style tool docs (type names as bfcl writes them: dict, float, ...), covering every type R1 maps
TOOLS = [
    {"name": "cd", "description": "Change directory.",
     "parameters": {"type": "dict", "properties": {"folder": {"type": "string"}}, "required": ["folder"]}},
    {"name": "ls", "description": "List files.",
     "parameters": {"type": "dict", "properties": {"a": {"type": "boolean"}}, "required": []}},
    {"name": "book_flight", "description": "Book a flight.",
     "parameters": {"type": "dict", "required": ["travel_from", "travel_to", "travel_cost"],
                    "properties": {"access_token": {"type": "string"}, "travel_from": {"type": "string"},
                                   "travel_to": {"type": "string"}, "travel_cost": {"type": "float"},
                                   "travel_class": {"type": "string", "enum": ["economy", "business", "first"]},
                                   "passengers": {"type": "integer"}}}},
    {"name": "post_tweet", "description": "Post a tweet.",
     "parameters": {"type": "dict", "required": ["content"],
                    "properties": {"content": {"type": "string"},
                                   "tags": {"type": "array", "items": {"type": "string"}},
                                   "mentions": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "set_car", "description": "Nested settings.",
     "parameters": {"type": "dict", "required": ["settings"],
                    "properties": {"settings": {"type": "dict", "required": ["temp"],
                                                "properties": {"temp": {"type": "integer"},
                                                               "fan": {"type": "boolean"}}}}}},
]
BY_NAME = {t["name"]: t for t in TOOLS}

# limits of llama.cpp's JSON grammar: integers of at most 16 digits, no exponent with a leading zero (1e-05)
INTS = st.integers(min_value=-10**15, max_value=10**15)
FLOATS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False).map(lambda x: round(x, 3))
# DEL (0x7F) is legal raw in JSON but llama.cpp's char rule excludes it; json.dumps escapes all other controls
TEXT = st.text(st.characters(blacklist_categories=("Cs",), blacklist_characters="\x7f"), max_size=30)


def value_for(p):
    if "enum" in p:
        return st.sampled_from(p["enum"])
    t = p.get("type")
    if t == "string":
        return TEXT
    if t == "integer":
        return INTS
    if t == "float":
        return FLOATS
    if t == "boolean":
        return st.booleans()
    if t == "array":
        return st.lists(value_for(p["items"]), max_size=4)
    if t == "dict":
        return args_for(p)
    raise ValueError(t)


def args_for(params):
    """A dict with every required key and any subset of the optional ones, in the grammar's order
    (required keys in schema order, then optional keys in schema order)."""
    props, req = params["properties"], params.get("required", [])
    opt = [k for k in props if k not in req]

    @st.composite
    def build(draw):
        out = {k: draw(value_for(props[k])) for k in props if k in req}
        for k in opt:
            if draw(st.booleans()):
                out[k] = draw(value_for(props[k]))
        return out
    return build()


@st.composite
def calls(draw):
    t = draw(st.sampled_from(TOOLS))
    return {"name": t["name"], "arguments": draw(args_for(t["parameters"]))}


def block(call, ensure_ascii):
    return "<tool_call>\n" + json.dumps(call, ensure_ascii=ensure_ascii) + "\n</tool_call>"


def valid(grammar, text):
    with tempfile.TemporaryDirectory() as d:
        g, i = os.path.join(d, "g.gbnf"), os.path.join(d, "in.txt")
        open(g, "w").write(grammar)
        with open(i, "w", encoding="utf-8") as f:
            f.write(text)
        # the validator echoes the input byte by byte around the error, which can split a UTF-8 character
        out = subprocess.run([VALIDATOR, g, i], capture_output=True, timeout=30).stdout.decode("utf-8", "replace")
    assert "valid" in out, out
    return "is valid" in out


STRICT, FREE = task_grammar(TOOLS), task_grammar_free(TOOLS)
SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(st.lists(calls(), min_size=1, max_size=3), st.booleans())
def test_r1_accepts_valid_calls(cs, ascii_only):
    text = "\n".join(block(c, ascii_only) for c in cs)
    assert valid(STRICT, text), text
    assert valid(FREE, text), text


@SETTINGS
@given(calls(), st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12))
def test_r1_rejects_unknown_name(c, name):
    if name in BY_NAME:
        return
    assert not valid(STRICT, block({**c, "name": name}, True))


@SETTINGS
@given(calls(), st.data())
def test_r1_rejects_missing_required_key(c, data):
    req = BY_NAME[c["name"]]["parameters"].get("required", [])
    if not req:
        return
    drop = data.draw(st.sampled_from(req))
    args = {k: v for k, v in c["arguments"].items() if k != drop}
    assert not valid(STRICT, block({**c, "arguments": args}, True))


@SETTINGS
@given(calls(), st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12))
def test_r1_rejects_extra_key(c, key):
    if key in BY_NAME[c["name"]]["parameters"]["properties"]:
        return
    assert not valid(STRICT, block({**c, "arguments": {**c["arguments"], key: 1}}, True))


@SETTINGS
@given(st.text(st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=60))
def test_free_branch(text):
    starts_ok = text[0] not in "<\n\t "
    assert valid(FREE, text) == starts_ok, text
    assert not valid(STRICT, text)


# ---- Pythonic format (LFM2.5): the same properties, calls written as the chat template writes them ----------
from testbed.grammar import pythonic_task_grammar, pythonic_task_grammar_free  # noqa: E402
from testbed.lfm2_format import extract_tool_calls, render_tool_calls  # noqa: E402

PY_STRICT, PY_FREE = pythonic_task_grammar(TOOLS), pythonic_task_grammar_free(TOOLS)
PY_THINK = pythonic_task_grammar_free(TOOLS, think_prefix=True)


@SETTINGS
@given(st.lists(calls(), min_size=1, max_size=3))
@example([{"name": "cd", "arguments": {"folder": "\x00"}}])   # found by hypothesis (suite job 112511)
def test_r1_pythonic_accepts_valid_calls(cs):
    text = render_tool_calls(cs)
    assert valid(PY_STRICT, text), text
    assert valid(PY_FREE, text), text
    assert valid(PY_THINK, "I should list the files first.</think>" + text), text
    assert extract_tool_calls(text)[0] == cs          # and what the grammar allows parses back unchanged


@SETTINGS
@given(calls(), st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12))
def test_r1_pythonic_rejects_unknown_name(c, name):
    if name in BY_NAME:
        return
    assert not valid(PY_STRICT, render_tool_calls([{**c, "name": name}]))


@SETTINGS
@given(calls(), st.data())
def test_r1_pythonic_rejects_missing_required_key(c, data):
    req = BY_NAME[c["name"]]["parameters"].get("required", [])
    if not req:
        return
    drop = data.draw(st.sampled_from(req))
    args = {k: v for k, v in c["arguments"].items() if k != drop}
    assert not valid(PY_STRICT, render_tool_calls([{**c, "arguments": args}]))


@SETTINGS
@given(calls(), st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12))
def test_r1_pythonic_rejects_extra_key(c, key):
    if key in BY_NAME[c["name"]]["parameters"]["properties"]:
        return
    assert not valid(PY_STRICT, render_tool_calls([{**c, "arguments": {**c["arguments"], key: 1}}]))


@SETTINGS
@given(st.text(st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=60))
def test_pythonic_free_branch(text):
    starts_ok = text[0] not in "<\n\t "
    assert valid(PY_FREE, text) == starts_ok, text
    assert not valid(PY_STRICT, text)
