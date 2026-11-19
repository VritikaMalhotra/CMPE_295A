#!/usr/bin/env python3
"""Fill the Ch. 1-2 report template (docs/dec4/CMPE295A_Report_Ch1-2_outline_2026-11-09.docx) with the draft text.

Sources, all in this folder: front.md (abstract, glossary), ch1.md, ch2.md, the two architecture figures, and the
APA reference list in workbook_md/references.md (keys cited as [@key]). Output: the filled .docx. The template's
styles are kept: IndentedParagraph body, Heading2/3, Bullet, tablecaption and FigureCaption (both auto-numbered),
ReferenceEntry. Fields (table of contents, lists of figures and tables) are updated when the file is opened in Word.

    python3 build_report.py --template ../CMPE295A_Report_Ch1-2_outline_2026-11-09.docx \
        --refs ../../../../workbook_md/references.md --out ../CMPE295A_Report_Ch1-2_draft_2026-11-19.docx
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import tempfile
import zipfile
from xml.sax.saxutils import escape

import refs as refmod

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_W = 8640                      # DXA: 12240 page - 2160 left - 1440 right
CITED: dict[str, None] = {}        # keys in order of first citation
REFS: dict = {}
SUFFIX: dict[str, str] = {}        # key -> year with a/b suffix


# ---------- inline text -> runs ----------
def _run(text: str, b=False, i=False, size=None) -> str:
    rpr = ("<w:b/>" if b else "") + ("<w:i/>" if i else "") + (f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' if size else "")
    return f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def cite(m: re.Match) -> str:
    keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
    parts = []
    for k in keys:
        if k not in REFS:
            raise KeyError(f"unknown citation key {k}")
        CITED.setdefault(k)
        parts.append(f"{REFS[k]['cite']}, {SUFFIX.get(k, REFS[k]['year'])}")
    return "(" + "; ".join(parts) + ")"


def runs(text: str, size=None, bold_all=False) -> str:
    text = re.sub(r"\[(@[^\]]+)\]", cite, text)
    out = []
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            out.append(_run(tok[2:-2], b=True, size=size))
        elif tok.startswith("*") and len(tok) > 2:
            out.append(_run(tok[1:-1], i=True, b=bold_all, size=size))
        else:
            out.append(_run(tok, b=bold_all, size=size))
    return "".join(out)


def para(style: str, text: str = "", inner: str | None = None) -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{inner if inner is not None else runs(text)}</w:p>'


# ---------- tables ----------
def table(rows: list[list[str]]) -> str:
    ncol = len(rows[0])
    # width ~ sqrt(longest cell), at least the longest word (about 100 DXA per character at 9 pt)
    weight = [max(len(r[c]) for r in rows) ** 0.5 for c in range(ncol)]
    floor = [max(len(w) for r in rows for w in re.sub(r"[*\[\]]", "", r[c]).split() or [""]) * 100 + 160
             for c in range(ncol)]
    widths = [int(TEXT_W * w / sum(weight)) for w in weight]
    for _ in range(5):
        short = [c for c in range(ncol) if widths[c] < floor[c]]
        if not short:
            break
        for c in short:
            widths[c] = floor[c]
        free = [c for c in range(ncol) if c not in short]
        rest = TEXT_W - sum(widths[c] for c in short)
        tw = sum(weight[c] for c in free)
        for c in free:
            widths[c] = int(rest * weight[c] / tw)
    widths[-1] += TEXT_W - sum(widths)
    border = "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                     for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    x = [f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:tblBorders>{border}</w:tblBorders>'
         '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="72" w:type="dxa"/><w:right w:w="72" w:type="dxa"/>'
         '</w:tblCellMar></w:tblPr><w:tblGrid>' + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"]
    for ri, r in enumerate(rows):
        hdr = ri == 0
        x.append("<w:tr><w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if hdr else "") + "</w:trPr>")
        for c, cell in enumerate(r):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E2"/>' if hdr else ""
            x.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[c]}" w:type="dxa"/>{shade}</w:tcPr>'
                     '<w:p><w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/><w:ind w:left="0" w:firstLine="0"/></w:pPr>'
                     f'{runs(cell, size=18, bold_all=hdr)}</w:p></w:tc>')
        x.append("</w:tr>")
    x.append("</w:tbl>")
    return "".join(x)


# ---------- images ----------
MEDIA: list[tuple[str, str]] = []  # (rId, source path)


def png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def image(path: str, caption: str) -> str:
    rid = f"rIdDraft{len(MEDIA) + 1}"
    MEDIA.append((rid, path))
    w, h = png_size(path)
    cx = int(5.5 * 914400)
    cy = int(cx * h / w)
    if cy > int(6.8 * 914400):
        cy = int(6.8 * 914400)
        cx = int(cy * w / h)
    n = len(MEDIA)
    drawing = (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
               f'<wp:docPr id="{900 + n}" name="Figure {n}"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
               '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:nvPicPr><pic:cNvPr id="{900 + n}" name="figure{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
               '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
    cap = re.sub(r"^Figure \d+\.\s*", "", caption)
    return para("Figure", inner=drawing) + para("FigureCaption", cap)


# ---------- markdown blocks ----------
def convert(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln:
            i += 1
            continue
        if ln.startswith("### "):
            out.append(para("Heading3", ln[4:]))
        elif ln.startswith("## "):
            out.append(para("Heading2", ln[3:]))
        elif ln.startswith("Table: "):
            out.append(para("tablecaption", re.sub(r"^Table \d+\.\s*", "", ln[7:])))
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.+)\)", ln)
            out.append(image(os.path.join(HERE, m.group(2)), m.group(1)))
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(table(rows))
            continue
        elif ln.startswith("- "):
            out.append(para("Bullet", ln[2:]))
        elif re.match(r"\d+\. ", ln):
            out.append(para("IndentedParagraph", ln))
        else:
            out.append(para("IndentedParagraph", ln))
        i += 1
    return "".join(out)


def ref_entry(entry: str) -> str:
    return ('<w:p><w:pPr><w:pStyle w:val="ReferenceEntry"/><w:ind w:left="720" w:hanging="720"/></w:pPr>'
            + runs(entry.replace("  ", " ")) + "</w:p>")


def assign_suffixes(keys: list[str]) -> None:
    groups: dict = {}
    for k in keys:
        groups.setdefault((REFS[k]["cite"], REFS[k]["year"]), []).append(k)
    for (_, year), ks in groups.items():
        if len(ks) > 1:
            for n, k in enumerate(sorted(ks, key=lambda k: REFS[k]["entry"])):
                SUFFIX[k] = f"{year}{'abcdefgh'[n]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    REFS.update(refmod.load(a.refs))
    for v in REFS.values():
        if v["cite"].endswith(".") and not v["cite"].endswith("et al."):
            v["cite"] = v["cite"][:-1]
    texts = {n: open(os.path.join(HERE, n)).read() for n in ("front.md", "ch1.md", "ch2.md")}
    allkeys = list(dict.fromkeys(k.strip().lstrip("@") for t in texts.values()
                                 for m in re.finditer(r"\[(@[^\]]+)\]", t) for k in m.group(1).split(";")))
    assign_suffixes(allkeys)

    front = texts["front.md"]
    abstract = front.split("# Abstract", 1)[1].split("# Glossary", 1)[0].strip()
    glossary = [ln for ln in front.split("# Glossary", 1)[1].splitlines() if ln.strip()]
    ch1, ch2 = convert(texts["ch1.md"]), convert(texts["ch2.md"])

    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(a.template) as z:
        z.extractall(tmp)
    dp = os.path.join(tmp, "word", "document.xml")
    x = open(dp, encoding="utf-8").read()
    spans = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"<w:p[ >].*?</w:p>", x, flags=re.S)]
    text = lambda p: "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
    style = lambda p: (re.search(r'<w:pStyle w:val="([^"]+)"', p) or [None, None])[1]

    def find(pred, start=0):
        for j in range(start, len(spans)):
            if pred(spans[j][2]):
                return j
        raise ValueError("paragraph not found")

    i_abs = find(lambda p: text(p).startswith("[Update the abstract"))
    i_h1a = find(lambda p: style(p) == "Heading1" and "Project Overview" in text(p))
    i_h1b = find(lambda p: style(p) == "Heading1" and "Project Architecture" in text(p))
    i_glo = find(lambda p: style(p) == "GlossaryHeading")
    i_ref = find(lambda p: style(p) == "RefHeading")
    i_app = find(lambda p: style(p) == "AppendixHeading")

    refs_xml = "".join(ref_entry(REFS[k]["entry"].replace(f"({REFS[k]['year']}", f"({SUFFIX.get(k, REFS[k]['year'])}", 1))
                       for k in sorted(CITED, key=lambda k: REFS[k]["entry"].lower()))
    gl_xml = "".join(para("IndentedParagraph", ln) for ln in glossary)
    abs_xml = para("IndentedParagraph", abstract)

    # splice from the end so earlier offsets stay valid
    edits = [  # (first index replaced, last index replaced, new xml); keep the headings themselves
        (i_ref + 1, i_app - 1, refs_xml),
        (i_glo + 1, i_ref - 1, gl_xml),
        (i_h1b + 1, i_glo - 1, ch2),
        (i_h1a + 1, i_h1b - 1, ch1),
        (i_abs, i_abs, abs_xml),
    ]
    for first, last, new in edits:
        s, e = (spans[first][0], spans[last][1]) if first <= last else (spans[first][0], spans[first][0])
        x = x[:s] + new + x[e:]
    if "xmlns:wp=" not in x[:3000]:
        x = x.replace("<w:document ", '<w:document xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" ', 1)
    if "xmlns:r=" not in x[:3000]:
        x = x.replace("<w:document ", '<w:document xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ', 1)
    open(dp, "w", encoding="utf-8").write(x)

    os.makedirs(os.path.join(tmp, "word", "media"), exist_ok=True)
    rp = os.path.join(tmp, "word", "_rels", "document.xml.rels")
    rels = open(rp, encoding="utf-8").read()
    for rid, src in MEDIA:
        name = f"draft_{rid}.png"
        shutil.copy(src, os.path.join(tmp, "word", "media", name))
        rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                            f'officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
    open(rp, "w", encoding="utf-8").write(rels)
    ct = os.path.join(tmp, "[Content_Types].xml")
    c = open(ct, encoding="utf-8").read()
    if 'Extension="png"' not in c:
        c = c.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"', 1)
    open(ct, "w", encoding="utf-8").write(c)

    if os.path.exists(a.out):
        os.remove(a.out)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp)
    words = sum(len(re.sub(r"\[@[^\]]+\]", "", texts[n]).split()) for n in ("ch1.md", "ch2.md"))
    print(f"wrote {a.out}: {len(CITED)} references, {len(MEDIA)} figures, about {words} words in Ch. 1-2")


if __name__ == "__main__":
    main()
