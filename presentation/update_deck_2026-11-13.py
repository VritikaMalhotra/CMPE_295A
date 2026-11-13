#!/usr/bin/env python3
"""Fri 2026-11-13 advisor deck: add the feedback slides (agents, architecture, data, tests, which tasks fail,
examples, damage preview), rebuild the status slide and the closing questions. Every result and episode count is read
from a JSON file in the repository or from the run-status file. Fixed setup facts (model sizes, simulator settings,
GPU memory, sample sizes, bootstrap resamples) are written in the slide text.

    python presentation/update_deck_2026-11-13.py --base <deck as of 2c72ca9> --out presentation/<deck>.pptx \\
        --status results/m3/run_status.json [--figs docs/figs]

Style matches the existing deck: CONTENT layout, rounded cards (bg2 fill, accent6 outline), 14 pt text (tx1 bold
heads, tx2 body), 10 pt source notes in accent5 anchored above the footer.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


FACTS = {}            # filled in main() from code, configs and JSON (results and counts; setup facts are in the text)


def J(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def fmt_int(n):
    return f"{n:,}"


# ---------------------------------------------------------------- drawing helpers ----------------------------
def _scheme_fill(sp, val=None, rgb=None):
    spPr = sp._element.spPr
    for tag in ("a:solidFill", "a:noFill"):
        for e in spPr.findall(qn(tag)):
            spPr.remove(e)
    fill = etree.SubElement(spPr, qn("a:solidFill"))
    if rgb:
        etree.SubElement(fill, qn("a:srgbClr"), val=rgb)
    else:
        etree.SubElement(fill, qn("a:schemeClr"), val=val)
    # keep fill before the outline
    ln = spPr.find(qn("a:ln"))
    if ln is not None:
        spPr.remove(fill)
        ln.addprevious(fill)


def card(slide, x, y, w, h, name, fill=None, rgb=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.name = name
    sp.adjustments[0] = 0.04 if min(w, h) > 1.2 else 0.12
    sp.shadow.inherit = False
    _scheme_fill(sp, val=fill or "bg2", rgb=rgb)
    sp.line.width = Pt(0.75)
    ln = sp._element.spPr.find(qn("a:ln"))
    for e in list(ln):
        ln.remove(e)
    f = etree.SubElement(ln, qn("a:solidFill"))
    etree.SubElement(f, qn("a:schemeClr"), val="accent6")
    sp.text_frame.text = ""
    return sp


def text(slide, x, y, w, h, name, paras, size=14, anchor="t"):
    """paras: list of paragraphs; each a list of (text, {"b": bool, "color": scheme, "size": pt}) runs, or a str."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    bp = tf._txBody.find(qn("a:bodyPr"))
    for k in ("lIns", "tIns", "rIns", "bIns"):
        bp.set(k, "0")
    bp.set("anchor", anchor)
    first = True
    for para in paras:
        if isinstance(para, str):
            para = [(para, {})]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(4)
        for t, o in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = bool(o.get("b"))
            rPr = r._r.get_or_add_rPr()
            f = etree.SubElement(rPr, qn("a:solidFill"))
            etree.SubElement(f, qn("a:schemeClr"), val=o.get("color", "tx1"))
    return tb


def head_body(slide, x, y, w, h, name, head, body, size=14):
    paras = [[(head, {"b": True})]]
    for b in ([body] if isinstance(body, str) else body):
        paras.append([(b, {"color": "tx2"})])
    return text(slide, x, y, w, h, name, paras, size)


def source(slide, s):
    return text(slide, 0.6, 6.55, 12.1, 0.35, "Source note", [[(s, {"color": "accent5", "size": 10})]], 10, "b")


def stat(slide, x, y, w, name, number, label, color="tx1"):
    text(slide, x, y, w, 0.75, name + " number", [[(number, {"b": True, "size": 32, "color": color})]], 32)
    text(slide, x, y + 0.75, w, 0.7, name + " label", [[(label, {"color": "tx2", "size": 13})]], 13)


def table(slide, x, y, w, rows, col_w, name, size=12, head_fill="tx1"):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.3 * len(rows)))
    shp.name = name
    t = shp.table
    tbl = shp._element.graphic.graphicData.tbl
    tbl.tblPr.set("firstRow", "1")
    tbl.tblPr.set("bandRow", "0")
    style = tbl.tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        tbl.tblPr.remove(style)
    for i, cw in enumerate(col_w):
        t.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = t.cell(ri, ci)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            tf = c.text_frame
            tf.text = ""
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.bold = ri == 0
            rPr = r._r.get_or_add_rPr()
            f = etree.SubElement(rPr, qn("a:solidFill"))
            etree.SubElement(f, qn("a:schemeClr"), val="bg1" if ri == 0 else "tx1")
            tcPr = c._tc.get_or_add_tcPr()
            fill = etree.SubElement(tcPr, qn("a:solidFill"))
            etree.SubElement(fill, qn("a:schemeClr"), val=head_fill if ri == 0 else ("bg1" if ri % 2 else "bg2"))
    return shp


_SLDNUM = None


def new_slide(prs, title):
    global _SLDNUM
    if _SLDNUM is None:
        for sh in prs.slides[2].shapes:
            if sh.is_placeholder and sh.placeholder_format.type == 13:
                _SLDNUM = copy.deepcopy(sh._element)
    s = prs.slides.add_slide(prs.slide_layouts.get_by_name("CONTENT"))
    if _SLDNUM is not None:
        s.shapes._spTree.append(copy.deepcopy(_SLDNUM))
    for ph in s.placeholders:
        if ph.placeholder_format.type == 1:
            ph.text_frame.text = title
    return s


def clear_body(slide):
    """Remove every shape except the title and slide-number placeholders."""
    for sh in list(slide.shapes):
        if sh.is_placeholder and sh.placeholder_format.type in (1, 13):
            continue
        sh._element.getparent().remove(sh._element)


def set_notes(slide, text_):
    slide.notes_slide.notes_text_frame.text = text_


def set_title(slide, title):
    for ph in slide.placeholders:
        if ph.placeholder_format.type == 1:
            r = ph.text_frame.paragraphs[0].runs
            if r:
                r[0].text = title
                for extra in r[1:]:
                    extra.text = ""
            else:
                ph.text_frame.text = title


def move(prs, slide, index):
    lst = prs.slides._sldIdLst
    sid = [e for e in lst if e.get(qn("r:id")) == prs.part.relate_to(slide.part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide")][0]
    lst.remove(sid)
    lst.insert(index, sid)


# ---------------------------------------------------------------- slides ----------------------------------------
def slide_agent_loop(prs):
    s = new_slide(prs, "Our agent: one tool-calling model in a reason → act → observe loop")
    # loop: three boxes and arrows, left half
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    boxes = [("Reason", "the model reads the conversation and the tool schemas", 0.8, 1.55),
             ("Act", "it calls one or more tools, or answers in text", 0.8, 3.15),
             ("Observe", "each tool result is appended; next step", 0.8, 4.75)]
    for i, (h, b, x, y) in enumerate(boxes):
        card(s, x, y, 5.0, 1.25, f"Loop {h} card", fill="bg2")
        head_body(s, x + 0.25, y + 0.18, 4.5, 1.0, f"Loop {h} text", h, b)
    for y in (2.8, 4.4):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(3.3), Inches(y), Inches(3.3), Inches(y + 0.35))
        c.line.width = Pt(1.5)
        c.line.color.rgb = __import__("pptx").dml.color.RGBColor(0x5B, 0x6B, 0x7B)
        c.name = "Loop arrow"
        ln = c.line._get_or_add_ln()
        etree.SubElement(ln, qn("a:tailEnd"), type="triangle")
    card(s, 6.3, 1.55, 6.4, 1.55, "Category card")
    head_body(s, 6.55, 1.7, 5.9, 1.3, "Category text", "Category: a single tool-calling (function-calling) agent",
              "ReAct-style loop (Workbook 2 Ch. 1). The small model under test is the only agent; it decides every "
              "action.")
    card(s, 6.3, 3.25, 6.4, 1.55, "Purpose card")
    head_body(s, 6.55, 3.4, 5.9, 1.3, "Purpose text", "Purpose: an on-device assistant that acts through APIs",
              "Files, car controls, travel, trading, messaging; airline, retail and telecom customer service. "
              "0.6B–4B models that fit a laptop or phone at 4 bits.")
    card(s, 6.3, 4.95, 6.4, 1.35, "Step card")
    head_body(s, 6.55, 5.1, 5.9, 1.1, "Step text", "One step = one generation",
              f"Chat template with tools → llama-server → parse calls → execute → log; ≤ {FACTS['max_steps']} steps per "
              f"turn, ctx {FACTS['ctx']:,}, ≤ {FACTS['n_predict']:,} new tokens.")
    source(s, "docs/agents_and_data.md §1; testbed/runner.py (C2), testbed/backends.py (C3), testbed/prompt_formats.py.")
    return s


def slide_agent_parts(prs):
    s = new_slide(prs, "One agent; the simulated user, tools and judges are the environment")
    parts = [("Agent (the model under test)", "Qwen3, SmolLM3 or LFM2.5 as a GGUF file on llama-server, CUDA or CPU. "
              "Its prompt format is its own chat template."),
             ("Tool executors", "BFCL: the benchmark's Python API classes. τ²: the domain database environment. "
              "Deterministic."),
             ("User", "BFCL: scripted turns from the task. τ²: an LLM simulated user (gpt-oss-20b, temperature 0, one "
              "cluster job), the same for every model and format."),
             ("Judge", "Official checkers, unmodified: bfcl-eval's multi-turn checker and τ²'s database-state "
              "evaluator.")]
    for i, (h, b) in enumerate(parts):
        x = 0.6 + (i % 2) * 6.15
        y = 1.35 + (i // 2) * 1.75
        card(s, x, y, 5.95, 1.6, f"Part {i + 1} card")
        head_body(s, x + 0.25, y + 0.15, 5.45, 1.35, f"Part {i + 1} text", h, b)
    card(s, 0.6, 4.95, 12.1, 1.35, "Not multi-agent card", rgb="FDF1EA")
    head_body(s, 0.85, 5.1, 11.6, 1.1, "Not multi-agent text", "Not a multi-agent system",
              "Only the agent changes between configurations. The simulated user is a second LLM, but it is part "
              "of the benchmark, fixed and cached (C4), so a replay of an episode needs no simulator call.")
    source(s, "docs/agents_and_data.md §1 (C1, C4, C6 components; FR-E2 agreement in results/m2/evaluator_agreement.json).")
    return s


def slide_architecture(prs, man, parity, thr):
    s = new_slide(prs, "Architecture: what each layer is for, with our numbers")
    layers = [("Workload", "C1 benchmark adapter · C4 simulated user", f"Same {FACTS['tasks']} tasks for every cell"),
              ("Execution", "C2 runner · C3 backend · C7 repairs · C8 QAD", "Real llama.cpp 4-bit kernels"),
              ("Measurement", "C5 step logger · C6 classifier · C9 cost", "Every step logged and labelled"),
              ("Analysis + control", "C10 stats · C11 config, registry, Slurm", "Resumable; run id = hash")]
    for i, (h, c, why) in enumerate(layers):
        y = 1.35 + i * 1.2
        card(s, 0.6, y, 4.6, 1.05, f"Layer {i + 1} card")
        text(s, 0.8, y + 0.12, 4.2, 0.85, f"Layer {i + 1} text",
             [[(h, {"b": True})], [(c, {"color": "tx2", "size": 12})], [(why, {"color": "accent5", "size": 12})]], 12)
    rows = {r["level"]: r for r in thr["levels"]} if "levels" in thr else {}
    card(s, 5.4, 1.35, 3.55, 4.95, "Hardware card")
    a100 = rows.get("a100_n4", {})
    p100 = rows.get("p100_n1", {})
    text(s, 5.6, 1.5, 3.2, 4.7, "Hardware text", [
        [("Hardware (CoE HPC, Slurm)", {"b": True})],
        [(f"A100 40 GB: {a100['episodes_per_hour']:,.0f} episodes/h ({a100['n']} tasks, at most {a100['max_concurrent']} at once); {a100['episode_wall_s_median']:.1f} s per 0.6B BFCL episode" if a100 else "A100 40 GB", {"color": "tx2", "size": 12})],
        [(f"P100 12 GB: {p100['episodes_per_hour']:,.0f} episodes/h on one GPU; {p100['episode_wall_s_median']:.1f} s per episode" if p100 else "P100 12 GB", {"color": "tx2", "size": 12})],
        [("H100 80 GB: QAD training (R3)", {"color": "tx2", "size": 12})],
        [("CPU nodes: the CPU backend arm, KL on CPU, R", {"color": "tx2", "size": 12})],
        [("QoS: 2 GPU nodes per user, so our pace is set by nodes, not GPUs", {"b": True, "size": 12})],
    ], 12)
    card(s, 9.15, 1.35, 3.55, 2.4, "Backend card")
    par = [(m, d) for m, d in parity]
    text(s, 9.35, 1.5, 3.2, 2.2, "Backend text",
         [[("CUDA vs CPU, same GGUF", {"b": True})]] +
         [[(f"{m}: {b:.1f}% identical at BF16, {q:.1f}% at Q4_0", {"color": "tx2", "size": 12})] for m, (b, q) in par] +
         [[("Below 99%: backend stays a factor", {"b": True, "size": 12})]], 12)
    card(s, 9.15, 3.9, 3.55, 2.4, "Memory card")
    q06 = man["models"]["Qwen3-4B"]["files"]
    text(s, 9.35, 4.05, 3.2, 2.2, "Memory text", [
        [("Memory per format", {"b": True})],
        [(f"Qwen3-4B: BF16 {q06['BF16']['file_bytes'] / 2**30:.2f} GiB, Q4_K_M {q06['Q4_K_M']['file_bytes'] / 2**30:.2f} GiB, "
          f"Q4_0 {q06['Q4_0-pure']['file_bytes'] / 2**30:.2f} GiB", {"color": "tx2", "size": 12})],
        [(f"Bits per weight: Q4_0 {q06['Q4_0-pure']['bpw_file']:.2f}, IQ4_XS {q06['IQ4_XS']['bpw_file']:.2f}, "
          f"Q4_K_M {q06['Q4_K_M']['bpw_file']:.2f}, Q8_0 {q06['Q8_0']['bpw_file']:.2f}", {"color": "tx2", "size": 12})],
    ], 12)
    source(s, "docs/architecture.md; results/m2/throughput.json, results/m2/parity_qwen3-*.json, "
              "results/m3/models_manifest.json.")
    return s


def slide_data(prs, api_rows, tau_rows):
    s = new_slide(prs, "Data: 8 BFCL API domains and 3 τ² customer-service domains")
    rows = [["BFCL API class (domain)", "Functions", "Tasks (base / miss_func / miss_param)"]] + api_rows
    table(s, 0.6, 1.35, 6.6, rows, [3.0, 1.0, 2.6], "BFCL table", 11)
    rows2 = [["τ² domain", "Base", "Write tasks run", "Agent / user tools"]] + tau_rows
    table(s, 7.45, 1.35, 5.25, rows2, [1.35, 0.8, 1.45, 1.65], "Tau table", 11)
    card(s, 7.45, 3.0, 5.25, 3.3, "Data note card")
    text(s, 7.65, 3.15, 4.9, 3.05, "Data note text", [
        [(f"{FACTS['tasks']} tasks × (4 sampled + 1 greedy) = {FACTS['tasks'] * 5:,} episodes per cell", {"b": True})],
        [(f"BFCL: a fixed-seed {FACTS['bfcl_per_cat']} of {FACTS['bfcl_full']} tasks per category. miss_func "
          "withholds a function; miss_param leaves out a value (asking is right).", {"color": "tx2", "size": 12})],
        [("τ²: only tasks whose gold actions write to the database (an agent that does nothing would pass the rest).",
          {"color": "tx2", "size": 12})],
        [(f"Other data: WikiText-2 test (KL), WikiText-2 train minus 13-gram overlaps (imatrix), a "
          f"{FACTS['qad_tokens'] / 1e6:.1f}M-token QAD mix ({FACTS['qad_sources']} Hugging Face sources), all filtered "
          f"against the benchmark text.", {"color": "tx2", "size": 12})],
    ], 12)
    source(s, "docs/agents_and_data.md §2–4 (counts from bfcl-eval 2026.3.23 and tau2-bench-verified 864350a data).")
    return s


def slide_classes(prs, wtf):
    s = new_slide(prs, "What measures the agent: verdicts and step classes")
    ref = wtf["configs"][wtf["reference"]["label"]]["failure_classes"]
    allr = wtf.get("reference_all", {}).get("failure_classes", ref)          # BF16 on all its sampled episodes
    rows = [["Failure class (codebook v0)", "Qwen3-0.6B BF16, per 100 steps"]]
    for c in sorted(allr, key=lambda c: -(allr[c]["mean"] or 0)):
        v = allr[c]
        rows.append([c, f"{v['mean']:.1f} ± {v['sd']:.1f}" if v.get("sd") is not None else f"{v['mean']:.1f}"])
    table(s, 0.6, 1.35, 5.6, rows, [3.4, 2.2], "Classes table", 12)
    card(s, 6.45, 1.35, 6.25, 2.3, "Episode card")
    head_body(s, 6.7, 1.5, 5.75, 2.0, "Episode text", "Episode verdict: the official checkers",
              ["bfcl-eval multi-turn checker (state + response) and τ²'s database evaluator.",
               f"FR-E2: our re-scoring agrees on {FACTS['fre2_agree']:,} of {FACTS['fre2_n']:,} Qwen3 reference "
               "episodes."])
    card(s, 6.45, 3.8, 6.25, 2.5, "Step card")
    head_body(s, 6.7, 3.95, 5.75, 2.25, "Step text", "Step class: what went wrong, and where",
              [f"{FACTS['n_classes']} classes, first matching rule wins (format before content). Primary outcome: "
               "failures per step (decision 2).", "Not covered in v0: wrong tool, wrong value, policy violation (need the reference "
               "trajectory); raters note them for v1."])
    source(s, "docs/codebook_v0.md; results/m3/which_tasks_fail_qwen3-0.6b.json (sampled seeds, mean ± SD over seeds; "
              "development data); results/m2/evaluator_agreement.json.")
    return s


def slide_which_fail(prs, fig, wtf):
    s = new_slide(prs, "Which tasks fail: BF16 vs 4-bit (development data)")
    s.shapes.add_picture(fig, Inches(0.6), Inches(1.3), width=Inches(12.1))
    labs = list(wtf["configs"])
    source(s, f"analysis/which_tasks_fail.py; results/m3/which_tasks_fail_qwen3-0.6b.json; {wtf['common_pairs']:,} "
              f"common (task, seed) pairs, configurations {', '.join(labs)}. Development data, preview.")
    return s


def slide_damage(prs, dmg):
    s = new_slide(prs, "First 4-bit damage numbers, Qwen3-0.6B (preview)")
    cells = list(dmg["cells"])
    rows = [["Failure class", "BF16 per 100 steps"] + [f"{c}: per 100 steps · ratio [95% CI]" for c in cells]]
    keys = ["any_failure"] + sorted({k for c in cells for k in dmg["cells"][c]["classes"] if k != "any_failure"},
                                    key=lambda k: -max(dmg["cells"][c]["classes"].get(k, {}).get("rate_4bit_per100", 0)
                                                       for c in cells))
    first = dmg["cells"][cells[0]]["classes"]
    for k in keys[:9]:
        row = ["any failure" if k == "any_failure" else k,
               f"{first[k]['rate_ref_per100']:.1f}" if k in first else "–"]
        for c in cells:
            r = dmg["cells"][c]["classes"].get(k)
            if not r:
                row.append("–")
                continue
            ci = r.get("ratio_ci95")
            ratio = "–" if r.get("ratio") is None else (f"×{r['ratio']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
            row.append(f"{r['rate_4bit_per100']:.1f} · {ratio}")
        rows.append(row)
    w = [3.0, 1.9] + [(12.1 - 4.9) / len(cells)] * len(cells)
    table(s, 0.6, 1.35, 12.1, rows, w, "Damage table", 12)
    # computed summary: classes whose CI lies entirely above or below 1
    up, down = [], []
    for c in cells:
        for k, r in dmg["cells"][c]["classes"].items():
            ci = r.get("ratio_ci95")
            if ci and ci[0] > 1:
                up.append(f"{k} ×{r['ratio']:.1f} ({c})")
            elif ci and ci[1] < 1:
                down.append(f"{k} ×{r['ratio']:.2f} ({c})")
    card(s, 0.6, 4.75, 12.1, 1.55, "Damage summary card", rgb="FDF1EA")
    text(s, 0.85, 4.88, 11.6, 1.35, "Damage summary text", [
        [("95% CI above 1: ", {"b": True, "size": 13}), ("; ".join(up) or "none", {"color": "tx2", "size": 13})],
        [("95% CI below 1: ", {"b": True, "size": 13}), ("; ".join(down) or "none", {"color": "tx2", "size": 13})],
        [("Development data from the three complete cells; not the registered test. Fewer exec_error and "
          "tool_error_payload steps under 4-bit can mean fewer calls made at all.", {"color": "accent5", "size": 11})]], 13)
    pairs = ", ".join(f"{c}: {dmg['cells'][c]['pairs']:,} pairs" for c in cells)
    left = [("any failure" if k == "any_failure" else k) for k in keys[9:]]
    source(s, f"analysis/damage_table.py; results/m3/damage_preview_qwen3-0.6b.json. Ratio = 4-bit rate / BF16 rate per "
              f"step on paired episodes ({pairs}); task-cluster bootstrap, 10,000 resamples."
              + (f" Not shown (lowest rates): {', '.join(left)}; see the JSON." if left else ""))
    return s


def _steps_paras(ep, label):
    head = f"{label}: verdict {'pass' if ep['valid'] else 'fail'}" + (f" ({ep['error_type']})" if ep["error_type"] else "")
    paras = [[(head, {"b": True, "size": 13})]]
    for st in ep["steps"]:
        out = st["output"].replace("\n", " ")
        out = out if len(out) <= 150 else out[:150] + " […]"
        paras.append([(f"t{st['turn']}.s{st['step']} {st['label']}: ", {"b": True, "size": 10}),
                      (out, {"color": "tx2", "size": 10})])
        if st["results"]:
            res = " | ".join(st["results"])
            res = res if len(res) <= 110 else res[:110] + " […]"
            paras.append([("→ " + res, {"color": "accent5", "size": 10})])
    return paras


def slide_example(prs, ex, dom, title):
    s = new_slide(prs, title)
    d = ex["domains"][dom]
    card(s, 0.6, 1.35, 5.95, 4.95, "Reference example card")
    text(s, 0.8, 1.5, 5.55, 4.7, "Reference example text", _steps_paras(d["ref"], ex["ref_label"]), 10)
    if d.get("cell"):
        card(s, 6.75, 1.35, 5.95, 4.95, "Cell example card")
        text(s, 6.95, 1.5, 5.55, 4.7, "Cell example text", _steps_paras(d["cell"], ex["cell_label"]), 10)
    source(s, f"analysis/worked_examples.py: copied from the step logs (seed 1, sampled), runs {d['ref_run']} and "
              f"{d.get('cell_run')}; task {d['task']}; texts cut mechanically. Development data.")
    return s


def slide_status(slide, st, runs):
    clear_body(slide)
    set_title(slide, st["title"])
    f = lambda t: t.format(runs=runs)                  # noqa: E731  counts come from run_status.json
    cols = [("M2 (references + plan)", [f(t) for t in st["m2"]]), ("M3 (damage map v1, Dec 4)", [f(t) for t in st["m3"]]),
            ("Dec 4 (Canvas)", [f(t) for t in st["dec4"]])]
    for i, (h, items) in enumerate(cols):
        x = 0.6 + i * 4.1
        card(slide, x, 1.35, 3.9, 4.95, f"Status {i + 1} card")
        text(slide, x + 0.2, 1.5, 3.5, 4.7, f"Status {i + 1} text",
             [[(h, {"b": True, "size": 15})]] + [[(t, {"color": "tx2", "size": 13})] for t in items], 13)
    source(slide, st["source"])


def slide_questions(slide, qs):
    clear_body(slide)
    set_title(slide, "What we would like from you today")
    rows = max(5, -(-len(qs) // 2))                     # two columns; 11 or 12 asks take a sixth row
    step = 1.0 if rows == 5 else 0.9
    for i, q in enumerate(qs):
        col, row = divmod(i, rows)
        x = 0.8 + col * 6.1
        y = 1.35 + row * step
        from pptx.enum.shapes import MSO_SHAPE
        c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.5), Inches(0.5))
        c.name = f"Ask {i + 1} circle"
        _scheme_fill(c, val="accent1")
        c.line.fill.background()
        tf = c.text_frame
        tf.word_wrap = False                            # "10" must stay on one line in the circle
        tf.margin_left = tf.margin_right = 0
        tf.text = str(i + 1)
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER    # word_wrap off drops the shape's default centring
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        r = tf.paragraphs[0].runs[0]
        r.font.size = Pt(14 if i < 9 else 12)
        r.font.bold = True
        r.font.color.rgb = __import__("pptx").dml.color.RGBColor(0xFF, 0xFF, 0xFF)
        tb = text(slide, x + 0.7, y - 0.02, 5.2, min(0.9, step - 0.05), f"Ask {i + 1} text", [[(q, {"color": "bg1", "size": 14})]], 14)


class _Runs(dict):
    """runs[experiment][field] for templates. A run missing from the status file is an error, not a silent 0."""
    def __missing__(self, k):
        raise KeyError(f"run {k!r} is not in the run-status file; refresh it or fix the deck text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--content", default=os.path.join(ROOT, "presentation", "deck_content_2026-11-13.json"))
    ap.add_argument("--status", default="results/m3/run_status.json")
    ap.add_argument("--figs", default=os.path.join(ROOT, "docs", "figs"))
    a = ap.parse_args()
    prs = Presentation(a.base)
    content = json.load(open(a.content))
    runs = _Runs(J(a.status)["runs"])
    fre2 = J("results/m2/evaluator_agreement.json")
    runs["_fre2"] = {"agree": fre2["agree"], "n": fre2["n"]}
    fre2_lfm = "results/m2/evaluator_agreement_lfm2.5-1.2b.json"    # A2 (wk11-ekant; on the integration branch)
    if os.path.exists(os.path.join(ROOT, fre2_lfm)):                 # absent: a template that uses it raises
        f2 = J(fre2_lfm)
        runs["_fre2_lfm12b"] = {"agree": f2["agree"], "n": f2["n"]}
    fre2_smol = "results/m2/evaluator_agreement_smollm3-3b.json"    # A2 (wk11-ekant d43676f; FAIL, decision 17)
    if os.path.exists(os.path.join(ROOT, fre2_smol)):
        f2 = J(fre2_smol)
        runs["_fre2_smol"] = {"agree": f2["agree"], "n": f2["n"]}
    fre2_26b = "results/m2/evaluator_agreement_lfm2.5-2.6b.json"    # A2 (wk11-ekant f92d3ef; PASS)
    if os.path.exists(os.path.join(ROOT, fre2_26b)):
        f2 = J(fre2_26b)
        runs["_fre2_lfm26b"] = {"agree": f2["agree"], "n": f2["n"]}
    a3 = "results/m2/mixed_effects_qwen3-0.6b.json"    # A3 real-data run (wk11-vritika; present on the integration branch)
    runs["_a3"] = {"text": "real-data run waits for the first finished 4-bit cell"}
    if os.path.exists(os.path.join(ROOT, a3)):
        me = J(a3)
        mods = me["result"]["models"]
        rr = next(t for t in mods["A"]["fixed"] if t["term"].startswith("configuration"))
        fit = "models A–C fit" if all(m["status"] == "ok" for m in mods.values()) else "not all of models A–C fit"
        runs["_a3"] = {"text": f"real data (BF16 vs {me['variants'][1].upper()}, {me['n_pairs']:,} pairs): {fit}, "
                               f"A's rate ratio {rr['exp_estimate']:.2f} [{rr['exp_ci_low']:.2f}, "
                               f"{rr['exp_ci_high']:.2f}]"}
    fill = lambda t: t.format(runs=runs)       # noqa: E731
    man = J("results/m3/models_manifest.json")
    thr = J("results/m2/throughput.json")
    parity = []
    for m in ("0.6b", "1.7b", "4b"):
        p = J(f"results/m2/parity_qwen3-{m}.json")
        parity.append((f"Qwen3-{m.upper()}", (100 * p["variants"]["bf16"]["output_identical"],
                                                   100 * p["variants"]["q4_0"]["output_identical"])))
    dom = J("results/m3/benchmark_domains.json")
    import sys
    import yaml
    sys.path.insert(0, ROOT)
    from testbed.classifier import CLASSES
    cfgy = yaml.safe_load(open(os.path.join(ROOT, "configs", "experiment", "ref_qwen3_0p6b.yaml")))["experiment"]
    qad = J("results/m3/qad_mix_manifest.json")
    FACTS.update({
        "max_steps": cfgy["max_steps_per_turn"], "ctx": cfgy["server"]["ctx"], "n_predict": cfgy["decoding"]["n_predict"],
        "tasks": sum(c["tasks_sampled"] for c in dom["bfcl"]["categories"].values())
                 + sum(v["write_tasks"] for v in dom["tau2"].values()),
        "bfcl_per_cat": dom["bfcl"]["per_category"],
        "bfcl_full": next(iter(dom["bfcl"]["categories"].values()))["tasks_in_bfcl"],
        "qad_tokens": qad["train"]["tokens_qwen3"], "qad_sources": len(qad["sources"]),
        "n_classes": len(CLASSES), "fre2_agree": runs["_fre2"]["agree"], "fre2_n": runs["_fre2"]["n"]})
    api_rows = [[f"{c}", str(v["functions"]), " / ".join(str(v["tasks"][k]) for k in
                 ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param"))]
                for c, v in dom["bfcl"]["api_classes"].items()]
    tau_rows = [[d, str(v["base_tasks"]), str(v["write_tasks"]), f"{v['agent_tools']} / {v['user_tools']}"]
                for d, v in dom["tau2"].items()]
    wtf = J("results/m3/which_tasks_fail_qwen3-0.6b.json")
    dmg = J("results/m3/damage_preview_qwen3-0.6b.json")
    ex = J("results/m3/worked_examples_qwen3-0.6b.json")
    slides = list(prs.slides)
    for rt in content.get("retext", []):       # one-run text of a base-deck shape, replaced whole
        sh = next(s for s in slides[rt["slide"]].shapes if s.name == rt["shape"])
        para = sh.text_frame.paragraphs[0]
        if len(sh.text_frame.paragraphs) != 1 or len(para.runs) != 1 or para.runs[0].text != rt["old"]:
            raise ValueError(f"base slide {rt['slide']} shape {rt['shape']!r} is not the expected text")
        para.runs[0].text = fill(rt["new"])
    slide_status(slides[1], content["status"], runs)
    slide_questions(slides[11], [fill(q) for q in content["questions"]])   # counts in a question come from run_status
    set_notes(slides[1], fill(content["notes"]["status"]))
    set_notes(slides[11], fill(content["notes"]["questions"]))
    for sh in slides[9].shapes:                # plan slide: the open item it named is done
        if sh.name == "Open text":
            p0 = sh.text_frame.paragraphs[0]
            p0.runs[0].text = fill(content["plan_open"][0])
            for r in p0.runs[1:]:
                r.text = ""
            if len(p0.runs) > 1:
                p0.runs[1].text = fill(content["plan_open"][1])
    new = [slide_agent_loop(prs), slide_agent_parts(prs), slide_architecture(prs, man, parity, thr),
           slide_data(prs, api_rows, tau_rows), slide_classes(prs, wtf)]
    for sl, key in zip(new, ("agent_loop", "agent_parts", "architecture", "data", "classes")):
        set_notes(sl, fill(content["notes"][key]))
    for i, s in enumerate(new):            # after the status slide
        move(prs, s, 2 + i)
    tail = [slide_which_fail(prs, os.path.join(a.figs, "which_tasks_fail_qwen3-0.6b.png"), wtf), slide_damage(prs, dmg)]
    set_notes(tail[0], fill(content["notes"]["which_fail"]))
    set_notes(tail[1], fill(content["notes"]["damage"]))
    for e in content.get("examples", []):
        if e["domain"] in ex["domains"]:
            tail.append(slide_example(prs, ex, e["domain"], e["title"]))
            set_notes(tail[-1], fill(content["notes"]["example"]))
    order = list(prs.slides)
    pos = order.index(slides[7]) + 1       # after the power slide, before the decisions slide
    for i, s in enumerate(tail):
        move(prs, s, pos + i)
    for idx, text_ in content.get("retitle", {}).items():
        set_title(slides[int(idx)], text_)
    prs.save(a.out)
    print("wrote", a.out, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
