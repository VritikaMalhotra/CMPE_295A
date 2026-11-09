"""C1 Benchmark Adapter: BFCL multi-turn tasks with tool schemas in one JSON-schema form.

Mirrors bfcl-eval 2026.3.23 exactly for the prompt side (``load_dataset_entry`` ->
``populate_test_cases_with_predefined_functions`` + ``add_language_specific_hint_to_function_doc``, and the
holdout-turn message of ``BaseHandler.inference_multi_turn_prompting``), so scores stay comparable with the
official evaluator:
  * tools = every function doc of every involved class, in bfcl's own format (``type: dict`` etc.),
    with " Note that the provided function is in Python 3 syntax." appended to each description;
  * ``excluded_function`` is ignored, as bfcl itself ignores it for every multi-turn category;
  * multi_turn_miss_func: each name in ``missed_function`` (turn index -> names) is removed from the tool
    list (first doc with that name, as bfcl pops it) *before* the hint is added, so held-out docs carry no
    hint. The holdout turn has no user message in the data; bfcl fills it with one user message,
    ``DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=<held-out docs>)`` (the Python
    repr of the doc list), and in prompting mode the system prompt's tool list never changes. We put that
    message into ``Task.turns`` here, so the runner needs no category-specific code;
  * multi_turn_miss_param needs nothing extra: its ground truth has an empty turn where a parameter is missing;
  * ground truth from ``data/possible_answer``; each category is scored by bfcl's checker under its own name.
``json_schema()`` gives the normalized JSON-schema view (dict->object, float->number) used by
the R1 grammar and by C6's schema checks.
"""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache

PY_HINT = " Note that the provided function is in Python 3 syntax."
BFCL_CATEGORIES = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param")
SAMPLE_SEED = 295            # fixed seed of the 100-per-category BFCL sample (Workbook Ch. 3: 300 BFCL tasks)
SAMPLE_PER_CATEGORY = 100
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
    category: str = "multi_turn_base"
    # miss_func only: turn index (str, as in bfcl) -> held-out function docs (bfcl format, no hint)
    missed_function: dict[str, list[dict]] = field(default_factory=dict)

    @property
    def all_tools(self) -> list[dict]:
        """The prompt's tools plus the held-out ones (all methods bfcl's executor can run for this task)."""
        return self.tools + [d for docs in self.missed_function.values() for d in docs]

    @property
    def tool_names(self) -> set[str]:
        """Names the executor may run. Includes held-out functions: bfcl executes them (they are methods of
        the involved classes), before and after the turn that announces them."""
        return {t["name"] for t in self.all_tools}

    def tool_schema(self, name: str) -> dict | None:
        for t in self.all_tools:
            if t["name"] == name:
                return json_schema(t["parameters"])
        return None

    def checker_entry(self) -> dict:
        """The test_entry dict bfcl's multi_turn_checker expects (as bfcl's eval passes it)."""
        e = copy.deepcopy(self.raw)
        e["function"] = copy.deepcopy(self.tools)
        if self.missed_function:
            e["missed_function"] = copy.deepcopy(self.missed_function)
        return e


def holdout_message(docs: list[dict]) -> dict:
    """bfcl's user message for a miss_func holdout turn (prompting mode)."""
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING
    return {"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=docs)}


def load_bfcl_multi_turn(category: str = "multi_turn_base", ids: list[str] | None = None,
                         limit: int | None = None) -> list[Task]:
    data = os.path.join(bfcl_data_dir(), f"BFCL_v4_{category}.json")
    answers = {a["id"]: a["ground_truth"] for a in _jsonl(os.path.join(bfcl_data_dir(), "possible_answer",
                                                                        f"BFCL_v4_{category}.json"))}
    tasks = []
    for e in _jsonl(data):
        if ids is not None and e["id"] not in ids:
            continue
        docs = [json.loads(d) for cls in e["involved_classes"] for d in _func_docs(cls)]
        missed: dict[str, list[dict]] = {}
        for turn_idx, names in e.get("missed_function", {}).items():   # bfcl: pop before adding the hint
            missed[turn_idx] = []
            for name in names:
                for i, doc in enumerate(docs):
                    if doc["name"] == name:
                        missed[turn_idx].append(docs.pop(i))
                        break
        for doc in docs:
            doc["description"] = doc["description"] + PY_HINT
        turns = copy.deepcopy(e["question"])
        for turn_idx, held in missed.items():
            if turns[int(turn_idx)]:
                raise ValueError(f"{e['id']}: holdout turn {turn_idx} has a user message (bfcl asserts it has none)")
            turns[int(turn_idx)] = [holdout_message(held)]
        tasks.append(Task(id=e["id"], benchmark=f"bfcl_{category}", turns=turns,
                          initial_config=e.get("initial_config", {}), involved_classes=e["involved_classes"],
                          tools=docs, ground_truth=answers[e["id"]], raw=e, category=category,
                          missed_function=missed))
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


def sample_task_ids(category: str, n: int = SAMPLE_PER_CATEGORY, seed: int = SAMPLE_SEED) -> list[str]:
    """Fixed-seed sample of n task ids from one BFCL multi-turn category, returned in file order.
    Each category draws from its own stream (seeded with "<seed>:<category>"; string seeds are hashed with
    SHA-512, so the draw is the same on every machine and Python 3 version), so the three samples are
    independent rather than the same 100 indices."""
    data = os.path.join(bfcl_data_dir(), f"BFCL_v4_{category}.json")
    ids = [e["id"] for e in _jsonl(data)]
    if n > len(ids):
        raise ValueError(f"{category} has {len(ids)} tasks, cannot sample {n}")
    chosen = set(random.Random(f"{seed}:{category}").sample(ids, n))
    return [i for i in ids if i in chosen]


def bfcl_sample(n: int = SAMPLE_PER_CATEGORY, seed: int = SAMPLE_SEED,
                categories: tuple[str, ...] = BFCL_CATEGORIES) -> dict[str, list[str]]:
    """The study's BFCL task pool: n per category (default 3 x 100 = 300), keyed by benchmark name."""
    return {f"bfcl_{c}": sample_task_ids(c, n, seed) for c in categories}
