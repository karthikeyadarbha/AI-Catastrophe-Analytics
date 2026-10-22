#!/usr/bin/env python3
"""Propose bronze_text_sources/raw/index.csv by matching file names to text IDs.

Run from the repository root, after copying the files into bronze_text_sources/raw/:
    python3 bronze_text_sources/pipeline/suggest_raw_index.py

This only PROPOSES. It reads the real names on disk (so exact spelling, spaces and
Devanagari characters are taken from the files, not retyped), matches each name against
the RULES table below, and writes the result to index.csv. Review index.csv, then run
register_raw_batch.py. Registration reads index.csv only; it never matches on names.

- Rows already in index.csv are kept exactly as they are, including your corrections.
- Only files not yet listed are added. A blank text_id means "no confident match":
  fill it in by hand (the batch run refuses to start while any text_id is blank).
- Matching is case-insensitive and ignores Unicode composition differences
  (macOS stores some Devanagari names differently from Linux).
"""
import csv
import sys
import unicodedata
from pathlib import Path

REPO = Path.cwd()
RAW = REPO / "bronze_text_sources" / "raw"
INDEX = RAW / "index.csv"

# (keyword, text_id, note). First match wins, so specific keywords come before general ones.
# text_id None means: deliberately not matched; the note says why and what to do.
# text_id "SKIP" means: recognised, and deliberately not registered.
RULES = [
    ("drgganita",          "TXT-04", ""),
    ("adbhuta",            "TXT-12", "Adbhuta Sagara. The 399/401-page extracts and the zip are pieces of the full PDF."),
    ("aryabhatiya",        "TXT-01", ""),
    ("brihat",             "TXT-11", "Brhat Samhita (an English edition and a Hindi edition)."),
    ("बृहत्संहिता",         "TXT-11", ""),
    ("yukti",              "TXT-03", ""),
    ("hindu zodiac",       "TXT-08", ""),
    ("source book",        "TXT-07", ""),
    ("dulakara",           "TXT-09", ""),
    ("parashara",          "TXT-13", ""),
    ("tantrasangraha",     "TXT-02", ""),
    ("panday",             "TXT-06", "Surya Siddhanta, R.C. Pandey edition with Ranganatha's commentary."),
    ("pandey",             "TXT-06", ""),
    ("surya",              None,     "Surya Siddhanta with no editor in the name: TXT-05 (Vijnana-bhasya) or TXT-06? Set by hand."),
    ("panca siddhantika",  None,     "Not one of the 13 registered texts (Varahamihira). Needs a new register entry."),
    ("muddle",             None,     "Not one of the 13 registered texts. Needs a register entry, or leave it out of raw/."),
    ("ancientgeography",   None,     "Cunningham's Ancient Geography of India, the unresolved source. Needs a register entry."),
    ("rulefirst",          "SKIP",   "Study report, not a classical source. Marked SKIP; better kept out of raw/."),
]


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def suggest(name: str):
    key = norm(name)
    for keyword, text_id, note in RULES:
        if norm(keyword) in key:
            return text_id, note
    return None, "No rule matched. Set by hand."


def read_index() -> dict:
    if not INDEX.is_file():
        return {}
    with INDEX.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["path"]: row["text_id"] for row in reader if row.get("path")}


def main() -> int:
    if not RAW.is_dir():
        print(f"Missing folder: {RAW.relative_to(REPO)}")
        return 1
    existing = read_index()
    known = {unicodedata.normalize("NFC", p) for p in existing}

    files = sorted(
        p.relative_to(RAW).as_posix()
        for p in RAW.rglob("*")
        if p.is_file() and p != INDEX and p.name != ".gitkeep"
    )

    rows = dict(existing)
    report = {"matched": [], "blank": []}
    for rel in files:
        if unicodedata.normalize("NFC", rel) in known:
            continue
        text_id, note = suggest(Path(rel).name)
        rows[rel] = text_id or ""
        (report["matched"] if text_id else report["blank"]).append((rel, text_id, note))

    with INDEX.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["path", "text_id"])
        for rel in sorted(rows):
            writer.writerow([rel, rows[rel]])

    print(f"Wrote {INDEX.relative_to(REPO)}: {len(rows)} rows ({len(existing)} kept, "
          f"{len(report['matched'])} matched, {len(report['blank'])} left blank)\n")
    for rel, text_id, note in report["matched"]:
        print(f"  {text_id}  {rel}")
    if report["blank"]:
        print("\nNeed a decision (text_id left blank):")
        for rel, _, note in report["blank"]:
            print(f"  {rel}\n      -> {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
