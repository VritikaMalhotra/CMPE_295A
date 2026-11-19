"""Parse workbook_md/references.md (APA entries keyed by `key`) into in-text citations and reference entries."""
import re

def load(path):
    txt = open(path).read()
    out = {}
    for m in re.finditer(r"^\d+\.\s+(.+?)\n\s+`([^`]+)`:", txt, flags=re.M | re.S):
        entry, key = m.group(1).strip(), m.group(2)
        entry = re.sub(r"\s+", " ", entry).strip()
        am = re.match(r"(.+?)\s\((\d{4}[a-z]?|n\.d\.)(?:,[^)]*)?\)\.", entry)
        if not am:
            raise ValueError(f"cannot parse {key}: {entry[:80]}")
        authors, year = am.group(1), am.group(2)
        out[key] = {"entry": entry, "year": year, "cite": intext(authors)}
    return out

def intext(authors):
    a = authors.replace(", & ", ", ").replace(" & ", ", ")
    if a.rstrip(".").endswith("et al"):
        return a.split(",")[0] + " et al."
    # split "Last, F., Last2, F. M." into surnames; organisations have no initials
    parts = [p.strip() for p in a.split(",")]
    names = [p for p in parts if p and not re.fullmatch(r"(?:[A-Z][a-z]?\.(?:-[A-Z]\.)?\s?)+|Jr\.", p)]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return f"{names[0]} et al."

if __name__ == "__main__":
    import sys
    r = load(sys.argv[1])
    for k, v in r.items():
        print(f"{k:32s} ({v['cite']}, {v['year']})")
