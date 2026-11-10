
import json_schema_to_grammar as j2g
TYPE_MAP = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer", "int": "integer",
            "string": "string", "boolean": "boolean", "bool": "boolean", "array": "array", "tuple": "array", "list": "array"}
def to_js(p):
    out = {}
    t = p.get("type")
    if t in TYPE_MAP: out["type"] = TYPE_MAP[t]
    if "enum" in p: out["enum"] = p["enum"]
    if "properties" in p:
        out["properties"] = {k: to_js(v) for k, v in p["properties"].items()}
        out["required"] = [r for r in p.get("required", []) if r in p["properties"]]
        out["additionalProperties"] = False
    if "items" in p: out["items"] = to_js(p["items"])
    return out
def call_schema(fn):
    return {"type": "object", "properties": {"name": {"const": fn["name"]}, "arguments": to_js(fn["parameters"])},
            "required": ["name", "arguments"], "additionalProperties": False}
def task_grammar(functions):
    conv = j2g.SchemaConverter(prop_order={"name": 0, "arguments": 1}, allow_fetch=False, dotall=False, raw_pattern=False)
    schema = call_schema(functions[0]) if len(functions) == 1 else {"anyOf": [call_schema(f) for f in functions]}
    conv.visit(schema, "call")
    return 'root ::= block ("\\n" block)*\nblock ::= "<tool_call>\\n" call "\\n</tool_call>"\n' + conv.format_grammar()


def task_grammar_free(functions):
    # spec-style R1: either one or more schema-valid calls, or free text that does not start with "<" or whitespace
    g = task_grammar(functions).replace('root ::= block ("\\n" block)*', 'root ::= calls | free\ncalls ::= block ("\\n" block)*', 1)
    return g.rstrip('\n') + '\nfree ::= [^<\\n\\t ] free-char*\nfree-char ::= [^\\x00]\n'


# ---- R1 for the Pythonic tool-call format (LFM2.5; testbed/lfm2_format.py) ---------------------------------
# One block "<|tool_call_start|>[f(a=1), g(b='x')]<|tool_call_end|>". Keyword arguments follow the schema:
# required ones first, then optional ones, each in schema order (as the JSON grammar orders properties).
# Values are written as the chat template writes them: strings single-quoted with \\ \' \n \r escapes,
# numbers and True/False as Python literals, lists and dicts as JSON (the template's tojson).
PY_CALL_START, PY_CALL_END = "<|tool_call_start|>", "<|tool_call_end|>"
_PY_RULES = r'''py-str ::= "'" py-char* "'"
py-char ::= [^'\\\n\r] | "\\" ['"\\nrt]
py-int ::= "-"? ([0-9] | [1-9] [0-9]{1,15})
py-num ::= py-int ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
py-bool ::= "True" | "False"
'''


def _rule_name(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9-]+", "-", s).strip("-") or "x"


def _py_value(conv, p: dict, name: str, extra: list) -> str:
    """The rule for one keyword value; nested lists/dicts become JSON rules through the converter."""
    js = to_js(p)
    if "enum" in p:
        alts = " | ".join(conv._format_literal("'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'")
                          for v in p["enum"])
        extra.append(f"{name} ::= {alts}")
        return name
    t = js.get("type")
    if t == "string":
        return "py-str"
    if t == "integer":
        return "py-int"
    if t == "number":
        return "py-num"
    if t == "boolean":
        return "py-bool"
    if t in ("array", "object"):
        return conv.visit(js, name)
    return conv.visit({}, name)                          # untyped: any JSON value


def _py_args(names: list[str], required: set, kv: dict) -> str:
    """Comma-separated keyword arguments: required always, optional maybe, no stray separators."""
    req = [n for n in names if n in required]
    opt = [n for n in names if n not in required]
    if req:
        out = ' ", " '.join(kv[n] for n in req)
        for n in opt:
            out += f' (", " {kv[n]})?'
        return out
    if not opt:
        return ""

    def tail(i):
        return "".join(f' (", " {kv[n]})?' for n in opt[i + 1:])
    return "(" + " | ".join(f"{kv[n]}{tail(i)}" for i, n in enumerate(opt)) + ")?"


def pythonic_task_grammar(functions, think_prefix: bool = False):
    """Forced R1 for the Pythonic format: one block of one or more schema-valid calls. ``think_prefix``
    (LFM2.5-2.6B, whose answer starts inside <think>) allows reasoning text and "</think>" first."""
    conv = j2g.SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    extra: list[str] = []
    call_rules = []
    for f in functions:
        fn = _rule_name(f["name"])
        params = f.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or []) & set(props)
        kv = {}
        for k, p in props.items():
            r = f"{fn}-{_rule_name(k)}"
            val = _py_value(conv, p, r + "-val", extra)
            extra.append(f"{r}-kv ::= {conv._format_literal(k + '=')} {val}")
            kv[k] = f"{r}-kv"
        args = _py_args([k for k in props if k in required] + [k for k in props if k not in required], required, kv)
        extra.append(f"call-{fn} ::= {conv._format_literal(f['name'] + '(')} {args} \")\"")
        call_rules.append(f"call-{fn}")
    head = "root ::= " + ("think " if think_prefix else "") + "block\n"
    head += f'block ::= {conv._format_literal(PY_CALL_START + "[")} call (", " call)* {conv._format_literal("]" + PY_CALL_END)}\n'
    head += "call ::= " + " | ".join(call_rules) + "\n"
    if think_prefix:
        head += 'think ::= ([^<] | "<" [^/])* "</think>"\n'
    return head + _PY_RULES + "\n".join(extra) + "\n" + conv.format_grammar()


def pythonic_task_grammar_free(functions, think_prefix: bool = False):
    """Spec-style R1 for the Pythonic format: schema-valid calls, or free text not starting with "<" or
    whitespace (the same free branch as the JSON grammar)."""
    g = pythonic_task_grammar(functions, think_prefix)
    first = g.split("\n", 1)
    body = "root ::= " + ("think " if think_prefix else "") + "(block | free)"
    return body + "\n" + first[1].rstrip("\n") + '\nfree ::= [^<\\n\\t ] free-char*\nfree-char ::= [^\\x00]\n'


def grammars_for(syntax: str, think_prefix: bool = False) -> dict:
    """Constraint name -> grammar builder for a prompt format's call syntax ("json" or "pythonic")."""
    if syntax == "json":
        return {"none": None, "free": task_grammar_free, "forced": task_grammar}
    if syntax == "pythonic":
        return {"none": None,
                "free": lambda fns: pythonic_task_grammar_free(fns, think_prefix),
                "forced": lambda fns: pythonic_task_grammar(fns, think_prefix)}
    raise ValueError(f"unknown call syntax {syntax!r}")
