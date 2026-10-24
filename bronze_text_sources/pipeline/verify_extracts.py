#!/usr/bin/env python3
"""Check the extracted text against the audit chain. Read-only: changes nothing.

Run from the repository root:
    python3 bronze_text_sources/pipeline/verify_extracts.py [ARTIFACT_ID ...]

For every extract recorded by extract_text.py it checks that
  - the audit chain itself is intact,
  - extracted/<artifact_id>/text.txt exists and its SHA-256 matches the audit entry,
  - text.txt is exactly the page files joined in order (so an edited or missing
    page file is caught, even though page files are not hashed individually),
  - there is one page file per page, and pages.jsonl has a record for each page,
  - the scan the text was made from is registered in the audit log.
Then it prints a one-line quality summary per file. Exit code 0 only if all checks pass.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
BRONZE = REPO / "bronze_text_sources"
sys.path.insert(0, str(BRONZE / "pipeline"))
from audit import AuditLog  # noqa: E402

AUDIT = BRONZE / "audit" / "ingestion_audit.jsonl"
OUT = BRONZE / "extracted"


def main() -> int:
    only = set(sys.argv[1:])        # optional: artifact IDs to check; default is all
    audit = AuditLog(str(AUDIT))
    problems = []
    if not audit.verify():
        problems.append("audit chain failed verification")

    registered = {}
    extracts = {}
    for e in audit.list_entries():
        meta = e.get("metadata") or {}
        if meta.get("artifact"):
            registered[meta["artifact"]["artifact_id"]] = meta["artifact"]
        if meta.get("extract"):
            extracts[e["source_file"]] = (e, meta["extract"])   # latest entry wins

    unknown = only - {ex["derived_from_artifact"] for _, ex in extracts.values()}
    if unknown:
        sys.exit("No recorded extract for: " + ", ".join(sorted(unknown)))

    rows = []
    for rel, (entry, ex) in sorted(extracts.items(), key=lambda kv: (kv[1][1]["text_id"], kv[0])):
        aid = ex["derived_from_artifact"]
        if only and aid not in only:
            continue
        base = OUT / aid
        text_file = REPO / rel
        tag = f"{aid} {ex['text_id']}"
        n = ex["page_count"]

        if aid not in registered:
            problems.append(f"{tag}: source scan is not registered in the audit log")
        elif registered[aid]["sha256"] != ex["derived_from_sha256"]:
            problems.append(f"{tag}: recorded scan hash differs from the registered scan")

        if not text_file.is_file():
            problems.append(f"{tag}: {rel} is missing")
            continue
        if audit.sha256_file(str(text_file)) != entry["file_hash"]:
            problems.append(f"{tag}: text.txt does not match its audit hash (edited?)")

        # Stream the comparison: one page in memory at a time, however many pages a text has.
        missing, mismatch = [], False
        with text_file.open("r", encoding="utf-8", newline="") as joined:
            for p in range(1, n + 1):
                f = base / "pages" / f"page_{p:04d}.txt"
                if not f.is_file():
                    missing.append(p)
                    continue
                if mismatch or missing:
                    continue
                page = f.read_text(encoding="utf-8")
                if joined.read(len(page)) != page or (p < n and joined.read(1) != "\f"):
                    mismatch = True
            if not missing and not mismatch and joined.read(1) != "":
                mismatch = True
        if missing:
            problems.append(f"{tag}: {len(missing)} page file(s) missing, first {missing[:5]}")
        elif mismatch:
            problems.append(f"{tag}: page files do not add up to text.txt (a page file was changed)")

        log = base / "pages.jsonl"
        logged = set()
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("method") in ("text_layer", "ocr"):
                    logged.add(r["page"])
        if logged != set(range(1, n + 1)):
            problems.append(f"{tag}: pages.jsonl does not cover pages 1-{n} exactly")

        rows.append((ex["text_id"], aid, n, ex["pages_text_layer"], ex["pages_ocr"],
                     ex["pages_low_confidence"], ex["mean_ocr_confidence"]))

    print(f"{'text':7} {'artifact':17} {'pages':>5} {'layer':>5} {'ocr':>5} {'low<60':>6} {'mean conf':>9}")
    for t, a, n, lay, ocr, low, mean in rows:
        print(f"{t:7} {a:17} {n:>5} {lay:>5} {ocr:>5} {low:>6} {('-' if mean is None else mean):>9}")
    print(f"\n{len(rows)} extract(s) checked; "
          f"{sum(r[2] for r in rows)} pages, {sum(r[4] for r in rows)} OCR, "
          f"{sum(r[5] for r in rows)} flagged low-confidence.")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
