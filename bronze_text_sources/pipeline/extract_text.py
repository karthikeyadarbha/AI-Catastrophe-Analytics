#!/usr/bin/env python3
"""Extract page text from the registered scans (direct text layer, otherwise OCR).

Run from the repository root, after register_raw_batch.py:

    python3 bronze_text_sources/pipeline/extract_text.py --plan        # list work, change nothing
    python3 bronze_text_sources/pipeline/extract_text.py               # extract everything
    python3 bronze_text_sources/pipeline/extract_text.py --only RAW-0123abcd4567 --jobs 2
    python3 bronze_text_sources/pipeline/extract_text.py --exclude RAW-89abcdef0123

Needs on the machine: pdftotext, pdftoppm, pdfinfo (poppler) and tesseract with the
san, hin and eng language packs (brew install poppler tesseract tesseract-lang).

How it works
  * The list of files comes from the audit log (files registered by register_raw_batch.py),
    not from file names. Only PDFs are processed; the zip is skipped.
  * For each page: if the PDF has a usable text layer, that text is used (method
    "text_layer"). Otherwise the page is rendered at 300 dpi and read by Tesseract
    (method "ocr"), keeping its average word confidence.
  * Per-page results are written to extracted/<artifact_id>/pages/page_NNNN.txt and
    pages.jsonl as they finish, so an interrupted run resumes where it stopped.
  * When every page of a file is done, the pages are joined into text.txt (pages
    separated by a form-feed character) and one entry is appended to the audit chain
    with the tool versions, languages and settings. The hash of text.txt is recorded.
  * Nothing here judges quality or converts to IAST. Low-confidence pages are flagged
    in pages.jsonl for the Silver layer to review.
Original scans are never modified. Extracted text is small and may be committed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path.cwd()
BRONZE = REPO / "bronze_text_sources"
sys.path.insert(0, str(BRONZE / "pipeline"))
from audit import AuditLog  # noqa: E402

AUDIT = BRONZE / "audit" / "ingestion_audit.jsonl"
OUT = BRONZE / "extracted"
LANGS = "san+hin+eng"
DPI = 300
PSM = "3"
LOW_CONFIDENCE = 60.0   # percent; pages below this are flagged, not dropped


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def tool_versions():
    tess = run(["tesseract", "--version"])
    first = (tess.stdout or tess.stderr).splitlines()[0] if (tess.stdout or tess.stderr) else "unknown"
    poppler = run(["pdftotext", "-v"])
    pv = (poppler.stderr or poppler.stdout).splitlines()[0] if (poppler.stderr or poppler.stdout) else "unknown"
    return {"tesseract": first.strip(), "poppler": pv.strip()}


def check_tools():
    missing = [t for t in ("pdftotext", "pdftoppm", "pdfinfo", "tesseract") if not shutil.which(t)]
    if missing:
        sys.exit("Missing tools: " + ", ".join(missing) + "\nInstall with: brew install poppler tesseract tesseract-lang")
    langs = set(run(["tesseract", "--list-langs"]).stdout.split())
    need = set(LANGS.split("+"))
    if not need <= langs:
        sys.exit(f"Tesseract is missing language pack(s): {', '.join(sorted(need - langs))}\n"
                 "Install with: brew install tesseract-lang")


def page_count(pdf: str) -> int:
    m = re.search(r"Pages:\s+(\d+)", run(["pdfinfo", pdf]).stdout)
    if not m:
        raise RuntimeError(f"cannot read page count of {pdf}")
    return int(m.group(1))


def good_text_layer(text: str) -> bool:
    """A page's own text is used only if it looks like real text, not scanner debris."""
    stripped = "".join(text.split())
    if len(stripped) < 200:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    return letters / len(stripped) >= 0.6


def work_page(pdf: str, page: int, pages_dir: str):
    """Process one page. Runs in a worker process. Returns the per-page record."""
    out_file = Path(pages_dir) / f"page_{page:04d}.txt"
    layer = run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf, "-"]).stdout
    layer = unicodedata.normalize("NFC", layer)
    if good_text_layer(layer):
        text, method, conf = layer, "text_layer", None
    else:
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "p")
            r = run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(DPI), "-gray", "-png", "-singlefile", pdf, stem])
            if r.returncode != 0:
                return {"page": page, "method": "failed", "error": r.stderr.strip()[:200]}
            png = stem + ".png"
            # One Tesseract pass writes both the plain text and the word-level table.
            base = os.path.join(tmp, "o")
            txt = run(["tesseract", png, base, "-l", LANGS, "--psm", PSM, "txt", "tsv"], env=env)
            if txt.returncode != 0:
                return {"page": page, "method": "failed", "error": txt.stderr.strip()[:200]}
            page_text = Path(base + ".txt").read_text(encoding="utf-8")
            tsv_text = Path(base + ".tsv").read_text(encoding="utf-8")
        text = unicodedata.normalize("NFC", page_text)
        confs = []
        for line in tsv_text.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) == 12 and parts[11].strip():
                try:
                    c = float(parts[10])
                except ValueError:
                    continue
                if c >= 0:
                    confs.append(c)
        conf = round(sum(confs) / len(confs), 1) if confs else 0.0
        method = "ocr"
    tmp_path = out_file.with_suffix(".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(out_file)       # a page file exists only when it is complete
    rec = {"page": page, "method": method, "chars": len(text), "confidence": conf}
    if method == "ocr" and (conf is None or conf < LOW_CONFIDENCE):
        rec["flag"] = "low_confidence"
    return rec


def registered_pdfs(audit: AuditLog):
    """{artifact_id: (rel_path, artifact_record)} for registered PDFs, from the audit log."""
    found = {}
    for e in audit.list_entries():
        art = (e.get("metadata") or {}).get("artifact")
        if art and art.get("media_type") == "application/pdf":
            found[art["artifact_id"]] = (e["source_file"], art)
    return found


def done_pages(pages_dir: Path):
    recs = {}
    log = pages_dir.parent / "pages.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("method") in ("text_layer", "ocr") and (pages_dir / f"page_{r['page']:04d}.txt").is_file():
                recs[r["page"]] = r
    return recs


def process(aid, rel, art, audit, jobs, versions, plan_only):
    pdf = str(REPO / rel)
    n = page_count(pdf)
    base = OUT / aid
    pages_dir = base / "pages"
    already = done_pages(pages_dir)
    todo = [p for p in range(1, n + 1) if p not in already]
    entry_rel = f"bronze_text_sources/extracted/{aid}/text.txt"
    finished = audit.latest_for(entry_rel) is not None
    print(f"{aid} {art['text_id']} {n:>4} pages, {len(already)} done, {len(todo)} to do"
          f"{'  [already recorded]' if finished else ''}  {art['original_filename']}")
    if plan_only or finished:
        return True
    if audit.latest_for(rel) and audit.latest_for(rel)["file_hash"] != audit.sha256_file(pdf):
        print(f"  REFUSED: {rel} no longer matches its registered bytes")
        return False
    pages_dir.mkdir(parents=True, exist_ok=True)
    log = (base / "pages.jsonl").open("a", encoding="utf-8")
    failed = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(work_page, pdf, p, str(pages_dir)) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if rec["method"] == "failed":
                failed += 1
                print(f"  page {rec['page']} FAILED: {rec.get('error')}")
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} pages", flush=True)
    log.close()
    if failed:
        print(f"  {failed} page(s) failed; run again to retry them. Nothing recorded for {aid}.")
        return False

    recs = done_pages(pages_dir)
    if len(recs) != n:
        print(f"  expected {n} pages, found {len(recs)}; run again. Nothing recorded for {aid}.")
        return False
    text = "\f".join((pages_dir / f"page_{p:04d}.txt").read_text(encoding="utf-8") for p in range(1, n + 1))
    (base / "text.txt").write_text(text, encoding="utf-8")
    ocr = [r for r in recs.values() if r["method"] == "ocr"]
    confs = [r["confidence"] for r in ocr if r["confidence"] is not None]
    summary = {
        "extract": {
            "text_id": art["text_id"],
            "derived_from_artifact": aid,
            "derived_from_sha256": art["sha256"],
            "page_count": n,
            "pages_text_layer": n - len(ocr),
            "pages_ocr": len(ocr),
            "pages_low_confidence": sum(1 for r in ocr if r.get("flag")),
            "mean_ocr_confidence": round(sum(confs) / len(confs), 1) if confs else None,
            "languages": LANGS, "dpi": DPI, "psm": PSM,
            "tools": versions,
            "processing_status": "ocr_done" if ocr else "transcribed",
        }
    }
    method = "ocr_assisted" if ocr else "transcribed_extract"
    audit.record(entry_rel, audit.sha256_file(str(base / "text.txt")), method, summary)
    print(f"  recorded {entry_rel}: {summary['extract']['pages_ocr']} OCR pages, "
          f"{summary['extract']['pages_low_confidence']} low-confidence")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show what would be done; change nothing")
    ap.add_argument("--only", nargs="+", metavar="ARTIFACT_ID", help="process just these artifacts")
    ap.add_argument("--exclude", nargs="+", metavar="ARTIFACT_ID", default=[], help="skip these artifacts")
    ap.add_argument("--jobs", type=int, default=2, help="parallel pages (default 2; each uses ~1 CPU)")
    args = ap.parse_args()

    check_tools()
    audit = AuditLog(str(AUDIT))
    if not audit.verify():
        sys.exit("Audit chain failed verification; stopping.")
    pdfs = registered_pdfs(audit)
    ids = [a for a in sorted(pdfs, key=lambda a: (pdfs[a][1]["text_id"], a))
           if (not args.only or a in args.only) and a not in args.exclude]
    unknown = [a for a in (args.only or []) + args.exclude if a not in pdfs]
    if unknown:
        sys.exit("Not a registered PDF: " + ", ".join(unknown))
    versions = tool_versions()
    ok = True
    for aid in ids:
        rel, art = pdfs[aid]
        ok = process(aid, rel, art, audit, args.jobs, versions, args.plan) and ok
    print("audit chain:", "OK" if AuditLog(str(AUDIT)).verify() else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
