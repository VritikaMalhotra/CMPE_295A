"""C1 Benchmark Adapter: BFCL multi-turn tasks with tool schemas in one JSON-schema form.

Mirrors bfcl-eval 2026.3.23 exactly for the prompt side (``populate_test_cases_with_predefined_functions``
+ the Python-syntax hint), so scores stay comparable with the official evaluator:
  * tools = every function doc of every involved class, in bfcl's own format (``type: dict`` etc.),
    with " Note that the provided function is in Python 3 syntax." appended to each description;
  * ``excluded_function`` is ignored, as bfcl itself ignores it for multi_turn_base;
  * ground truth from ``data/possible_answer``.
``json_schema()`` gives the normalized JSON-schema view (dict->object, float->number) used by
the R1 grammar and by C6's schema checks.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

PY_HINT = " Note that the provided function is in Python 3 syntax."
TYPE_MAP = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer",
            "int": "integer", "string": "string", "boolean": "boolean", "bool": "boolean", "array": "array",
            "tuple": "array", "list": "array", "any": None}


def bfcl_data_dir() -> str:
    import bfcl_eval
    return os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def _jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=None)
def _func_docs(class_name: str) -> tuple:
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    path = os.path.join(bfcl_data_dir(), "multi_turn_func_doc", MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name])
    return tuple(json.dumps(d) for d in _jsonl(path))


def json_schema(p: dict) -> dict:
    """bfcl parameter schema -> standard JSON schema (recursive)."""
    out: dict = {}
    t = TYPE_MAP.get(p.get("type"), None)
    if t:
        out["type"] = t
    if "enum" in p:
        out["enum"] = p["enum"]
    if "properties" in p:
        out["properties"] = {k: json_schema(v) for k, v in p["properties"].items()}
        out["required"] = [r for r in p.get("required", []) if r in p["properties"]]
    if "items" in p:
        out["items"] = json_schema(p["items"])
    return out


@dataclass
class Task:
    id: str
    benchmark: str
    turns: list[list[dict]]          # bfcl "question": a list of turns, each a list of messages
    initial_config: dict
    involved_classes: list[str]
    tools: list[dict]                # bfcl-format function docs as shown to the model
    ground_truth: list[list[str]]    # per turn, python call strings
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def tool_names(self) -> set[str]:
        return {t["name"] for t in self.tools}

    def tool_schema(self, name: str) -> dict | None:
        for t in self.tools:
            if t["name"] == name:
                return json_schema(t["parameters"])
        return None

    def checker_entry(self) -> dict:
        """The test_entry dict bfcl's multi_turn_checker expects."""
        e = copy.deepcopy(self.raw)
        e["function"] = copy.deepcopy(self.tools)
        return e


def load_bfcl_multi_turn(category: str = "multi_turn_base", ids: list[str] | None = None,
                         limit: int | None = None) -> list[Task]:
    data = os.path.join(bfcl_data_dir(), f"BFCL_v4_{category}.json")
    answers = {a["id"]: a["ground_truth"] for a in _jsonl(os.path.join(bfcl_data_dir(), "possible_answer",
                                                                        f"BFCL_v4_{category}.json"))}
    tasks = []
    for e in _jsonl(data):
        if ids is not None and e["id"] not in ids:
            continue
        tools = []
        for cls in e["involved_classes"]:
            for d in _func_docs(cls):
                doc = json.loads(d)
                doc["description"] = doc["description"] + PY_HINT
                tools.append(doc)
        tasks.append(Task(id=e["id"], benchmark=f"bfcl_{category}", turns=e["question"],
                          initial_config=e.get("initial_config", {}), involved_classes=e["involved_classes"],
                          tools=tools, ground_truth=answers[e["id"]], raw=e))
    if ids is not None:
        order = {i: n for n, i in enumerate(ids)}
        tasks.sort(key=lambda t: order[t.id])
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def skeleton_task_ids(n: int = 20, category: str = "multi_turn_base") -> list[str]:
    """The M1 task set: the first n ids of the category, in file order (fixed, documented in the manifest)."""
    data = os.path.join(bfcl_data_dir(), f"BFCL_v4_{category}.json")
    return [e["id"] for e in _jsonl(data)[:n]]
