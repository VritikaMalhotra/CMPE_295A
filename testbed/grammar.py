
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
