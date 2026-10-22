#!/usr/bin/env python3
"""Register every file under bronze_text_sources/raw/ in one run.

Run from the repository root:
    python3 bronze_text_sources/pipeline/register_raw_batch.py

Each file's text ID comes from bronze_text_sources/raw/index.csv, not from its name:

    path,text_id
    scan_0001.pdf,TXT-03
    batch2/page_set_a.pdf,TXT-05

Paths are relative to raw/. A text_id of SKIP means "seen, deliberately not registered
yet" (for example a study report, or a text that has no register entry). Change it to a
real ID later and run again; files already registered are not touched.

The run stops before registering anything if:
  - a file under raw/ has no row in index.csv (it would be silently skipped otherwise),
  - an index row points to a file that does not exist,
  - an index row is duplicated, has a blank text ID, or has an unknown text ID.
Paths are compared ignoring Unicode composition (macOS and Linux store some
Devanagari names differently).
Files that were already registered with unchanged bytes are skipped. Exit code is 0
only if every file is registered or already up to date.
"""
import csv
import sys
import unicodedata
from pathlib import Path

REPO = Path.cwd()
BRONZE = REPO / "bronze_text_sources"
sys.path.insert(0, str(BRONZE / "pipeline"))
from audit import AuditLog  # noqa: E402
from register_raw_file import AUDIT, RAW, known_texts, register  # noqa: E402

INDEX = RAW / "index.csv"
SKIP = "SKIP"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_index() -> dict:
    """Return {nfc_path: (path_as_written, text_id)}."""
    if not INDEX.is_file():
        raise SystemExit(f"Missing {INDEX.relative_to(REPO)}. Create it with columns: path,text_id "
                         "(or run suggest_raw_index.py)")
    with INDEX.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or [c.strip() for c in reader.fieldnames] != ["path", "text_id"]:
            raise SystemExit("index.csv must have exactly the header: path,text_id")
        mapping = {}
        for row in reader:
            rel = row["path"].strip()
            if not rel:
                continue
            key = nfc(rel)
            if key in mapping:
                raise SystemExit(f"Duplicate index row for {rel}")
            mapping[key] = (rel, (row["text_id"] or "").strip())
    return mapping


def files_on_disk() -> dict:
    """Return {nfc_path: path_as_on_disk}."""
    found = {}
    for p in RAW.rglob("*"):
        if p.is_file() and p != INDEX and p.name != ".gitkeep":
            rel = p.relative_to(RAW).as_posix()
            found[nfc(rel)] = rel
    return found


def main() -> int:
    mapping = load_index()
    texts = known_texts()

    blank = [rel for rel, tid in mapping.values() if not tid]
    bad_ids = [(rel, tid) for rel, tid in mapping.values() if tid and tid != SKIP and tid not in texts]
    if blank or bad_ids:
        for rel in blank:
            print(f"No text ID yet (fill it in index.csv): {rel}")
        for rel, tid in bad_ids:
            print(f"Unknown text ID {tid!r} for {rel}")
        print("Nothing was registered.")
        return 1

    on_disk = files_on_disk()
    unlisted = sorted(on_disk[k] for k in set(on_disk) - set(mapping))
    missing = sorted(mapping[k][0] for k in set(mapping) - set(on_disk))
    if unlisted or missing:
        for rel in unlisted:
            print(f"Not in index.csv (add a row): {rel}")
        for rel in missing:
            print(f"Listed in index.csv but not found: {rel}")
        print("Nothing was registered.")
        return 1

    audit = AuditLog(str(AUDIT))
    failures = 0
    skipped = 0
    for key in sorted(mapping):
        _, text_id = mapping[key]
        disk_rel = on_disk[key]
        if text_id == SKIP:
            print(f"SKIP {disk_rel}")
            skipped += 1
            continue
        ok, message = register(text_id, f"{RAW.relative_to(REPO).as_posix()}/{disk_rel}", texts, audit)
        print(("OK   " if ok else "FAIL ") + message)
        failures += 0 if ok else 1

    done = len(mapping) - skipped
    print(f"\n{done - failures} of {done} files registered or up to date"
          + (f"; {skipped} skipped." if skipped else "."))
    print("audit chain:", "OK" if AuditLog(str(AUDIT)).verify() else "FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
