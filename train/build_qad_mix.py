#!/usr/bin/env python3
"""QAD training mix and R2b calibration conversations (Workbook 2 C8 data step, work package K; FR-E5).

Workbook 2 Ch. 3: "The training mix is public function-calling and general text, filtered to share no 13-gram
with any benchmark task", about 50 million tokens per run; the R2b ranking uses "calibration conversations that
are disjoint from both benchmarks". Sources and budgets (chosen by Yashashav 2026-11-09; licences in
docs/qad_data.md), counted in Qwen3-0.6B tokens of the plain rendering below (the planning tokenizer; each model's
own count comes from the pack step with its chat template):

    toucan                 Agent-Ark/Toucan-1.5M, SFT config        Apache-2.0   15M  (+ 2,000 held-out multi-turn
                                                                                       conversations for R2b)
    nemotron_tool_calling  nvidia/Nemotron-Agentic-v1, tool_calling CC-BY-4.0     5M
    toolace                Team-ACE/ToolACE                         Apache-2.0    5M
    fineweb_edu            HuggingFaceFW/fineweb-edu, sample-10BT   ODC-By 1.0   20M
    cosmopedia_v2          HuggingFaceTB/smollm-corpus, cosmopedia  ODC-By 1.0    5M

Steps (the qad-data env, which has pyarrow, huggingface_hub and transformers):
    python train/build_qad_mix.py fetch --out $CMPE295_GROUP/data/qad_mix                 # login node: downloads
    python train/build_qad_mix.py check-toolace --out $CMPE295_GROUP/data/qad_mix --n 500  # parser yield
    sbatch slurm/qad_mix.sbatch                                                            # build (CPU job)

Every chat record is normalised to OpenAI-style messages (system/user/assistant/tool; assistant tool calls as
{"name", "arguments"}) plus its tool schemas. Dropped on the way: Nemotron's reasoning_content, ToolACE's system
prompt (BFCL's own prompt wording; its tool list is kept as tools). A record is dropped if any 13-gram of its
messages or tool schemas is a benchmark 13-gram (scripts/benchmark_ngrams.py dump, which includes the BFCL and
tau2 prompt text), or if its content duplicates an earlier record. Packing into [N, seq_len] token tensors is
left to the training pipeline (Sprint 7).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 295
RANK_HOLDOUT = 2000

SOURCES = {
    "toucan": {"repo": "Agent-Ark/Toucan-1.5M", "rev": "0df3cf37f2abefb380370cfb02eabea2a35ae782",
               "files": [f"SFT/train-0000{i}-of-00003.parquet" for i in range(3)], "license": "Apache-2.0",
               "kind": "chat", "budget": 15_000_000, "subsets": ["multi-turn", "single-turn-diversify"]},
    "nemotron_tool_calling": {"repo": "nvidia/Nemotron-Agentic-v1", "rev": "650d590978ca35c8f1ecea2faf136e5fac421b62",
                              "files": ["data/tool_calling.jsonl"], "license": "CC-BY-4.0", "kind": "chat",
                              "budget": 5_000_000},
    "toolace": {"repo": "Team-ACE/ToolACE", "rev": "6bda777c88d21e5a204703c1ee45597a8fa4f734", "files": ["data.json"],
                "license": "Apache-2.0", "kind": "chat", "budget": 5_000_000},
    "fineweb_edu": {"repo": "HuggingFaceFW/fineweb-edu", "rev": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                    "files": ["sample/10BT/000_00000.parquet"], "license": "ODC-By-1.0", "kind": "text",
                    "budget": 20_000_000},
    "cosmopedia_v2": {"repo": "HuggingFaceTB/smollm-corpus", "rev": "3ba9d605774198c5868892d7a8deda78031a781f",
                      "files": ["cosmopedia-v2/train-00000-of-00104.parquet"], "license": "ODC-By-1.0", "kind": "text",
                      "budget": 5_000_000},
}


def _load_ngrams():
    spec = importlib.util.spec_from_file_location("benchmark_ngrams", os.path.join(ROOT, "scripts", "benchmark_ngrams.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ng = _load_ngrams()


class ParseError(ValueError):
    pass


# ---------------------------------------------------------------- ToolACE: Python-style call lists ----------
def split_top(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside quotes and brackets. Quotes may be ' or " with backslash escapes."""
    out, cur, depth, quote, esc = [], [], 0, None, False
    for ch in s:
        if quote:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise ParseError("unbalanced brackets")
        elif ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if quote or depth:
        raise ParseError("unterminated quote or bracket")
    out.append("".join(cur))
    return out


_JSON_WORDS = re.compile(r"\b(true|false|null)\b")


def parse_value(v: str):
    """A Python or JSON literal; anything else stays a string (the generator sometimes writes bare words)."""
    v = v.strip()
    for attempt in (lambda x: ast.literal_eval(x), json.loads,
                    lambda x: ast.literal_eval(_JSON_WORDS.sub(lambda m: {"true": "True", "false": "False",
                                                                           "null": "None"}[m.group(1)], x))):
        try:
            return attempt(v)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return v


_KEY = re.compile(r"[A-Za-z_][\w.\- ]*")


def parse_py_calls(s: str) -> list[dict] | None:
    """'[Market Trends API(trend_type="MARKET_INDEXES", country="us"), f(x=1)]' -> list of {name, arguments}.
    None if `s` is not a call list (plain assistant text). Raises ParseError on a malformed call list."""
    t = s.strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    body = t[1:-1].strip()
    if not re.match(r"^[^\[\]{}()\"'=,]+\(", body):
        return None
    calls = []
    for part in split_top(body):
        part = part.strip()
        i = part.find("(")
        if i <= 0 or not part.endswith(")"):
            raise ParseError(f"not a call: {part[:60]!r}")
        name = part[:i].strip()
        args = {}
        inner = part[i + 1:-1].strip()
        if inner:
            for kv in split_top(inner):
                k, eq, v = kv.partition("=")
                k = k.strip()
                if not eq or not _KEY.fullmatch(k):
                    raise ParseError(f"bad argument {kv[:60]!r} in {name!r}")
                args[k] = parse_value(v)
        calls.append({"name": name, "arguments": args})
    return calls


TOOLACE_TOOLS_MARK = "Here is a list of functions in JSON format that you can invoke:"


def _dict_to_object(x):
    if isinstance(x, dict):
        return {k: ("object" if k == "type" and v == "dict" else _dict_to_object(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_dict_to_object(v) for v in x]
    return x


def toolace_record(rec: dict) -> tuple[list[dict], list[dict]]:
    sysp = rec.get("system") or ""
    i = sysp.find(TOOLACE_TOOLS_MARK)
    if i < 0:
        raise ParseError("no tool list in the system prompt")
    j = sysp.find("[", i)
    try:
        funcs, _ = json.JSONDecoder().raw_decode(sysp[j:])
    except (json.JSONDecodeError, ValueError) as e:
        raise ParseError(f"tool list is not JSON: {e}") from None
    tools = [{"type": "function", "function": _dict_to_object(f)} for f in funcs]
    msgs = []
    for turn in rec["conversations"]:
        role, val = turn["from"], turn["value"]
        if role == "assistant":
            calls = parse_py_calls(val)
            msgs.append({"role": "assistant", "content": "", "tool_calls": calls} if calls
                        else {"role": "assistant", "content": val})
        elif role in ("user", "tool"):
            msgs.append({"role": role, "content": val})
        else:
            raise ParseError(f"unknown role {role!r}")
    return msgs, tools


# ---------------------------------------------------------------- Toucan and Nemotron ----------------------
def toucan_record(row: dict) -> tuple[list[dict], list[dict]]:
    raw = json.loads(row["messages"])
    tools = json.loads(row["tools"]) if row.get("tools") else []
    msgs: list[dict] = []
    for m in raw:
        role, content = m.get("role"), m.get("content")
        if role == "tool_call":
            try:
                c = ast.literal_eval(content) if isinstance(content, str) else content
            except (ValueError, SyntaxError) as e:
                raise ParseError(f"tool_call: {e}") from None
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            call = {"name": c["name"], "arguments": args}
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif role == "tool_response":
            msgs.append({"role": "tool", "content": content if isinstance(content, str) else json.dumps(content)})
        elif role in ("system", "user", "assistant"):
            msgs.append({"role": role, "content": content or ""})
        else:
            raise ParseError(f"unknown role {role!r}")
    return msgs, tools


def nemotron_record(obj: dict) -> tuple[list[dict], list[dict]]:
    msgs = []
    for m in obj["messages"]:
        role, content = m["role"], m.get("content") or ""
        if role == "system" and not content.strip():
            continue
        out = {"role": role, "content": content if isinstance(content, str) else json.dumps(content)}
        if role == "assistant" and m.get("tool_calls"):
            out["tool_calls"] = []
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                args = f.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                out["tool_calls"].append({"name": f["name"], "arguments": args})
        msgs.append(out)                     # reasoning_content is not carried over
    return msgs, obj.get("tools") or []


# ---------------------------------------------------------------- rendering, filtering --------------------
def render_plain(msgs: list[dict]) -> str:
    """Role-prefixed text with no special tokens: the R2b ranking text (llama-perplexity reads chat-template
    markers as plain text, so a model's own template would add junk tokens) and the planning token count."""
    lines = []
    for m in msgs:
        if m.get("content"):
            lines.append(f"{m['role']}: {m['content']}")
        for c in m.get("tool_calls") or []:
            lines.append("assistant: <tool_call>" + json.dumps(c, ensure_ascii=False) + "</tool_call>")
    return "\n".join(lines)


def truncate_tool_turns(msgs: list[dict], max_tokens: int, encode, decode) -> tuple[list[dict], int]:
    """D-1 (R2b ranking window): every tool reply cut to its first ``max_tokens`` tokens of the planning
    tokenizer, so the ranking window covers more conversations. Returns (messages, replies cut)."""
    out, cut = [], 0
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            ids = encode(m["content"])
            if len(ids) > max_tokens:
                m = {**m, "content": decode(ids[:max_tokens])}
                cut += 1
        out.append(m)
    return out, cut


def build_rank_text(out: str, bench_path: str, tokenizer: str, max_tool_tokens: int) -> dict:
    """rank_text_t<N>.txt from the held-out conversations: the same rendering as rank_text.txt, tool replies
    truncated; the whole file is checked again for 13-grams shared with the benchmarks."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]          # noqa: E731
    dec = lambda ids: tok.decode(ids)                                        # noqa: E731
    rank_path = os.path.join(out, "rank_conversations.jsonl")
    convs = [json.loads(line) for line in open(rank_path) if line.strip()]
    parts, cut, replies = [], 0, 0
    for r in convs:
        msgs, c = truncate_tool_turns(r["messages"], max_tool_tokens, enc, dec)
        cut += c
        replies += sum(1 for m in r["messages"] if m.get("role") == "tool")
        parts.append(render_plain(msgs))
    text = "\n\n".join(parts)
    path = os.path.join(out, f"rank_text_t{max_tool_tokens}.txt")
    with open(path, "w") as fh:
        fh.write(text)
    with open(bench_path) as fh:
        bench = ng.ngram_set(fh.read())
    overlap = len(bench & ng.ngram_set(text))
    rec = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": rank_path,
           "source_sha256": _sha_file(rank_path), "max_tool_tokens": max_tool_tokens, "planning_tokenizer": tokenizer,
           "conversations": len(convs), "tool_replies": replies, "tool_replies_cut": cut,
           "text": path, "text_sha256": _sha_file(path), "text_tokens_qwen3": len(enc(text)),
           "n_overlapping_13grams": overlap, "pass": overlap == 0}
    with open(path[:-4] + ".json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))
    return rec


def record_check_text(msgs: list[dict], tools: list[dict]) -> str:
    return render_plain(msgs) + "\n" + "\n".join(ng._strings(tools))


def overlaps(text: str, bench: set) -> bool:
    return any(h in bench for h in ng.ngrams(ng.tokens(text)))


def content_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seeded_order(n: int, seed: int = SEED, salt: str = "") -> list[int]:
    idx = list(range(n))
    random.Random(f"{seed}:{salt}").shuffle(idx)
    return idx


# ---------------------------------------------------------------- I/O (pyarrow, hub, tokenizer) -----------
def raw_dir(out: str, name: str) -> str:
    return os.path.join(out, "raw", SOURCES[name]["repo"].replace("/", "__"))


def fetch(out: str) -> None:
    from huggingface_hub import hf_hub_download
    for name, s in SOURCES.items():
        for f in s["files"]:
            p = hf_hub_download(s["repo"], f, repo_type="dataset", revision=s["rev"], local_dir=raw_dir(out, name))
            print(f"{name}: {p} ({os.path.getsize(p) / 1e9:.2f} GB)")


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def iter_toucan(out: str, subsets: list[str], salt: str):
    import pyarrow.parquet as pq
    rows = []
    for f in SOURCES["toucan"]["files"]:
        t = pq.read_table(os.path.join(raw_dir(out, "toucan"), f), columns=["uuid", "subset_name", "messages", "tools"])
        rows += [r for r in t.to_pylist() if r["subset_name"] in subsets]
    rows.sort(key=lambda r: r["uuid"])                       # file order must not matter
    for i in seeded_order(len(rows), salt=salt):
        r = rows[i]
        yield f"toucan:{r['uuid']}", r["subset_name"], lambda r=r: toucan_record(r)


def iter_nemotron(out: str):
    path = os.path.join(raw_dir(out, "nemotron_tool_calling"), SOURCES["nemotron_tool_calling"]["files"][0])
    offsets, pos = [], 0
    with open(path, "rb") as fh:                              # index line starts; never hold the 5 GB file
        for line in fh:
            if line.strip():
                offsets.append(pos)
            pos += len(line)
    fh = open(path, "rb")
    for i in seeded_order(len(offsets), salt="nemotron"):
        fh.seek(offsets[i])
        obj = json.loads(fh.readline())
        yield f"nemotron:{obj.get('uuid', i)}", "tool_calling", lambda o=obj: nemotron_record(o)


def iter_toolace(out: str):
    d = json.load(open(os.path.join(raw_dir(out, "toolace"), "data.json")))
    for i in seeded_order(len(d), salt="toolace"):
        yield f"toolace:{i}", "", lambda r=d[i]: toolace_record(r)


def iter_text(out: str, name: str):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(raw_dir(out, name), SOURCES[name]["files"][0]), columns=["text"])
    texts = t.column("text").to_pylist()
    for i in seeded_order(len(texts), salt=name):
        yield f"{name}:{i}", texts[i]


def check_toolace(out: str, n: int) -> dict:
    d = json.load(open(os.path.join(raw_dir(out, "toolace"), "data.json")))
    ok, fails, calls, raw_args = 0, [], 0, 0
    for i in seeded_order(len(d), salt="toolace")[:n]:
        try:
            msgs, _ = toolace_record(d[i])
            ok += 1
            for m in msgs:
                for c in m.get("tool_calls") or []:
                    calls += 1
                    raw_args += sum(isinstance(v, str) and v[:1] in "[{" for v in c["arguments"].values())
        except ParseError as e:
            fails.append({"index": i, "error": str(e)[:160]})
    rep = {"checked": n, "parsed": ok, "yield": ok / n, "tool_calls": calls,
           "args_left_as_bracketed_strings": raw_args, "failures_sample": fails[:15]}
    print(json.dumps(rep, indent=1))
    return rep


def build(out: str, bench_path: str, tokenizer: str) -> dict:
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(tokenizer)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    bench_text = open(bench_path).read()
    bench = ng.ngram_set(bench_text)
    seen: set[str] = set()
    report, selected = {}, []

    def take(name, it, budget, kind, limit_records=None, rec_out=None):
        st = {"candidates": 0, "parse_failed": 0, "empty": 0, "benchmark_13gram": 0, "duplicate": 0,
              "kept": 0, "tokens": 0, "subsets": {}}
        for rid, subset, payload in it:
            if (budget is not None and st["tokens"] >= budget) or (limit_records and st["kept"] >= limit_records):
                break
            st["candidates"] += 1
            if kind == "chat":
                try:
                    msgs, tools = payload()
                except (ParseError, KeyError, TypeError, json.JSONDecodeError):
                    st["parse_failed"] += 1
                    continue
                body = {"messages": msgs, "tools": tools}
                text, check = render_plain(msgs), record_check_text(msgs, tools)
                count_text = text + "\n" + json.dumps(tools, ensure_ascii=False)
            else:
                body = {"text": payload}
                text = check = count_text = payload
            if not text.strip():
                st["empty"] += 1
                continue
            if overlaps(check, bench):
                st["benchmark_13gram"] += 1
                continue
            h = content_hash(body)
            if h in seen:
                st["duplicate"] += 1
                continue
            seen.add(h)
            n = ntok(count_text)
            st["kept"] += 1
            st["tokens"] += n
            st["subsets"][subset or "-"] = st["subsets"].get(subset or "-", 0) + 1
            (rec_out if rec_out is not None else selected).append(
                {"id": rid, "source": name, "subset": subset, "kind": kind, "n_tokens_qwen3": n, **body})
        report[name] = st
        print(f"{name}: {st}  ({time.time() - t0:.0f} s)", flush=True)

    rank: list[dict] = []
    # The holdout goes first, so its content hashes are in `seen` and the training draw skips them (counted as
    # "duplicate" under toucan); the manifest also checks the two id sets are disjoint.
    take("toucan_rank_holdout", iter_toucan(out, ["multi-turn"], "rank"), None, "chat", RANK_HOLDOUT, rank)
    take("toucan", iter_toucan(out, SOURCES["toucan"]["subsets"], "train"), SOURCES["toucan"]["budget"], "chat")
    take("nemotron_tool_calling", iter_nemotron(out), SOURCES["nemotron_tool_calling"]["budget"], "chat")
    take("toolace", iter_toolace(out), SOURCES["toolace"]["budget"], "chat")
    for name in ("fineweb_edu", "cosmopedia_v2"):
        take(name, ((rid, "", t) for rid, t in iter_text(out, name)), SOURCES[name]["budget"], "text")

    order = seeded_order(len(selected), salt="mix")
    train_path, rank_path = os.path.join(out, "train.jsonl"), os.path.join(out, "rank_conversations.jsonl")
    with open(train_path, "w") as fh:
        for i in order:
            fh.write(json.dumps(selected[i], ensure_ascii=False) + "\n")
    with open(rank_path, "w") as fh:
        for r in rank:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rank_text = "\n\n".join(render_plain(r["messages"]) for r in rank)
    rank_txt_path = os.path.join(out, "rank_text.txt")
    with open(rank_txt_path, "w") as fh:
        fh.write(rank_text)
    rank_after = len(bench & ng.ngram_set(rank_text))      # whole-file check, as for calib.txt
    ids_train = {r["id"] for r in selected}
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED,
        "planning_tokenizer": tokenizer, "bench_text": bench_path, "bench_text_sha256": _sha_file(bench_path),
        "sources": {k: {**s, "raw_sha256": {f: _sha_file(os.path.join(raw_dir(out, k), f)) for f in s["files"]}}
                    for k, s in SOURCES.items()},
        "filter": report,
        "train": {"path": train_path, "sha256": _sha_file(train_path), "records": len(selected),
                  "tokens_qwen3": sum(r["n_tokens_qwen3"] for r in selected)},
        "rank": {"path": rank_path, "sha256": _sha_file(rank_path), "records": len(rank),
                 "text": rank_txt_path, "text_sha256": _sha_file(rank_txt_path),
                 "text_tokens_qwen3": ntok(rank_text), "n_overlapping_13grams": rank_after,
                 "disjoint_from_train": not (ids_train & {r["id"] for r in rank})},
        "pass": rank_after == 0 and not (ids_train & {r["id"] for r in rank}),
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({k: manifest[k] for k in ("train", "rank", "pass")}, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "check-toolace", "build", "rank-text"):
        p = sub.add_parser(c)
        p.add_argument("--out", required=True)
        if c == "check-toolace":
            p.add_argument("--n", type=int, default=500)
        if c in ("build", "rank-text"):
            p.add_argument("--bench", required=True)
            p.add_argument("--tokenizer", required=True)
        if c == "rank-text":
            p.add_argument("--max-tool-tokens", type=int, default=256)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.out)
    elif a.cmd == "check-toolace":
        check_toolace(a.out, a.n)
    elif a.cmd == "rank-text":
        return 0 if build_rank_text(a.out, a.bench, a.tokenizer, a.max_tool_tokens)["pass"] else 1
    else:
        return 0 if build(a.out, a.bench, a.tokenizer)["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
