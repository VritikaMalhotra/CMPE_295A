"""Old vs new LFM2.5 parser on every logged step: does any tool-call block parse differently?

Usage: python3 -I equiv_scan.py <old lfm2_format.py> <new lfm2_format.py> <out.json> <run dir> [<run dir> ...]
Reads <run dir>/steps/*.jsonl and <run dir>/configs.txt. Compares parse_calls() block by block (result, or the
exception type), over every <|tool_call_start|>...<|tool_call_end|> block in raw_text. Also counts raw_text with a
NUL character and with a surrogate escape (\\ud800-\\udfff written as text).
"""
import glob, importlib.util, json, os, re, sys, time

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

old, new = load(sys.argv[1], "lfm_old"), load(sys.argv[2], "lfm_new")
SURR_ESC = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")

def outcome(mod, block):
    try:
        return ("ok", repr(mod.parse_calls(block)))
    except Exception as e:
        return ("err", type(e).__name__)

out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "old": sys.argv[1], "new": sys.argv[2], "runs": {}}
for rd in sys.argv[4:]:
    keys = [l.strip() for l in open(os.path.join(rd, "configs.txt")) if l.strip()]
    per = {}
    for f in sorted(glob.glob(os.path.join(rd, "steps", "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            ck = r["config_key"]
            d = per.setdefault(ck, {"task": keys.index(ck) if ck in keys else None, "steps": 0, "episodes": set(),
                                    "blocks": 0, "nul_steps": 0, "surrogate_escape_steps": 0, "differ": []})
            d["steps"] += 1; d["episodes"].add(r["run_id"])
            t = r["raw_text"]
            if "\x00" in t: d["nul_steps"] += 1
            if SURR_ESC.search(t): d["surrogate_escape_steps"] += 1
            for b in old.TOOL_CALL_RE.findall(t):
                d["blocks"] += 1
                a, c = outcome(old, b), outcome(new, b)
                if a != c:
                    d["differ"].append({"run_id": r["run_id"], "task_id": r["task_id"], "seed": r["seed"],
                                        "turn": r["turn"], "step": r["step"], "old": a, "new": c})
    for d in per.values():
        d["episodes"] = len(d["episodes"])
    out["runs"][rd] = per

json.dump(out, open(sys.argv[3], "w"), indent=1)
for rd, per in out["runs"].items():
    print(rd)
    for ck, d in sorted(per.items(), key=lambda kv: (kv[1]["task"] is None, kv[1]["task"])):
        print(f"  task {d['task']} {ck}: steps {d['steps']} episodes {d['episodes']} blocks {d['blocks']} "
              f"nul {d['nul_steps']} surr_esc {d['surrogate_escape_steps']} differ {len(d['differ'])}")
