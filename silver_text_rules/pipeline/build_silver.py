#!/usr/bin/env python3
"""Build the Silver DuckDB database from Bronze. Safe to re-run: it is rebuilt from scratch.

Run from the repository root:
    python3 silver_text_rules/pipeline/build_silver.py
    python3 silver_text_rules/pipeline/build_silver.py --db some/other.duckdb

Needs: pip install duckdb

Nothing is loaded unless Bronze checks out. The audit chain must verify, and for every
extract the text.txt hash and the page files must match what the audit log recorded
(the same checks as bronze_text_sources/pipeline/verify_extracts.py). Silver never
changes Bronze and never writes into the frozen engine.

Tables
  texts      one row per classical text (from the Bronze source register)
  artifacts  one row per registered scan (from the audit log)
  extracts   one row per text extraction run: tools, languages, quality summary
  pages      one row per scanned page: text, method, OCR confidence, flag, page hash
  rules      engine rule records (kind 'computational') and the 23 pre-registered Experiment 1
             rules (kind 'testable_claim', from silver_text_rules/rules/experiment1_rules.json,
             checked against schema/silver_rule.schema.json before loading)
  rule_sources  rule -> page links, where the engine record names a PDF page
  reported_results  what the Rule-First report published for each of the 23 rules; the
             target Gold must reproduce

The database file is derived data and is git-ignored. Delete it and run this again.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path.cwd()
BRONZE = REPO / "bronze_text_sources"
SILVER = REPO / "silver_text_rules"
sys.path.insert(0, str(BRONZE / "pipeline"))
from audit import AuditLog  # noqa: E402

try:
    import duckdb
    import jsonschema
except ImportError:
    sys.exit("The 'duckdb' and 'jsonschema' packages are required. Install them with:  "
             "python3 -m pip install duckdb jsonschema")

AUDIT = BRONZE / "audit" / "ingestion_audit.jsonl"
REGISTER = BRONZE / "manifest" / "source_register.json"
RULES_DIR = REPO / "indic-ganita-engine" / "indic_engine" / "rules" / "data"
BATCH = 1000

DDL = """
CREATE TABLE texts (text_id VARCHAR PRIMARY KEY, title VARCHAR, author VARCHAR, kind VARCHAR,
                    language VARCHAR, role VARCHAR);
CREATE TABLE artifacts (artifact_id VARCHAR PRIMARY KEY, text_id VARCHAR, original_filename VARCHAR,
                        sha256 VARCHAR, bytes BIGINT, media_type VARCHAR, registered_at VARCHAR);
CREATE TABLE extracts (artifact_id VARCHAR PRIMARY KEY, text_id VARCHAR, text_sha256 VARCHAR,
                       page_count INTEGER, pages_text_layer INTEGER, pages_ocr INTEGER,
                       pages_low_confidence INTEGER, mean_ocr_confidence DOUBLE,
                       languages VARCHAR, dpi INTEGER, tools VARCHAR, extraction_method VARCHAR,
                       recorded_at VARCHAR);
CREATE TABLE pages (artifact_id VARCHAR, text_id VARCHAR, page INTEGER, method VARCHAR,
                    confidence DOUBLE, flag VARCHAR, chars INTEGER, page_sha256 VARCHAR, text VARCHAR,
                    PRIMARY KEY (artifact_id, page));
CREATE TABLE rules (rule_id VARCHAR PRIMARY KEY, rule_kind VARCHAR, text_label VARCHAR, text_id VARCHAR,
                    reference VARCHAR, title VARCHAR, formula VARCHAR, implemented_by VARCHAR,
                    engine_status VARCHAR, sanskrit VARCHAR, sanskrit_verified BOOLEAN, gloss VARCHAR,
                    notes VARCHAR, provenance_status VARCHAR, review_status VARCHAR, source_file VARCHAR,
                    statement VARCHAR, condition VARCHAR, preregistration VARCHAR);
CREATE TABLE reported_results (rule_id VARCHAR PRIMARY KEY, discovery_p DOUBLE, holdout_p DOUBLE,
                    holdout_observed VARCHAR, holdout_expected VARCHAR);
CREATE TABLE rule_sources (rule_id VARCHAR, pdf_page_ref VARCHAR);   -- as written, e.g. '102' or '41-54'
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_bronze(audit: AuditLog):
    """Return (registered artifacts, latest extract entries) or exit with the problems found."""
    problems = []
    if not audit.verify():
        problems.append("audit chain failed verification")
    registered, extracts = {}, {}
    for e in audit.list_entries():
        meta = e.get("metadata") or {}
        if meta.get("artifact"):
            registered[meta["artifact"]["artifact_id"]] = meta["artifact"]
        if meta.get("extract"):
            extracts[e["source_file"]] = (e, meta["extract"])
    for rel, (entry, ex) in extracts.items():
        f = REPO / rel
        aid = ex["derived_from_artifact"]
        if aid not in registered:
            problems.append(f"{aid}: scan not registered")
        if not f.is_file():
            problems.append(f"{rel}: missing")
        elif audit.sha256_file(str(f)) != entry["file_hash"]:
            problems.append(f"{rel}: hash differs from the audit log")
    if problems:
        sys.exit("Bronze check failed; nothing was built:\n  - " + "\n  - ".join(problems)
                 + "\nRun bronze_text_sources/pipeline/verify_extracts.py for detail.")
    return registered, extracts


def load_pages(con, extracts):
    total = 0
    for rel, (entry, ex) in sorted(extracts.items()):
        aid, base = ex["derived_from_artifact"], REPO / "bronze_text_sources" / "extracted" / ex["derived_from_artifact"]
        recs = {}
        for line in (base / "pages.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("method") in ("text_layer", "ocr"):
                recs[r["page"]] = r
        # The pages are read once; their joined text is hashed on the way and must equal the
        # hash the audit log holds for text.txt, so an edited page file cannot slip in.
        joined = hashlib.sha256()
        batch = []
        for p in range(1, ex["page_count"] + 1):
            text = (base / "pages" / f"page_{p:04d}.txt").read_text(encoding="utf-8")
            r = recs[p]
            joined.update((("\f" if p > 1 else "") + text).encode("utf-8"))
            batch.append((aid, ex["text_id"], p, r["method"], r.get("confidence"), r.get("flag"),
                          len(text), sha256_text(text), text))
            if len(batch) >= BATCH:
                con.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)", batch)
                total += len(batch)
                batch = []
        if batch:
            con.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)", batch)
            total += len(batch)
        if joined.hexdigest() != entry["file_hash"]:
            sys.exit(f"Bronze check failed; {aid}: page files do not add up to the recorded text hash. "
                     "Nothing was built. Run bronze_text_sources/pipeline/verify_extracts.py.")
        con.execute("INSERT INTO extracts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            aid, ex["text_id"], entry["file_hash"], ex["page_count"], ex["pages_text_layer"],
            ex["pages_ocr"], ex["pages_low_confidence"], ex["mean_ocr_confidence"], ex["languages"],
            ex["dpi"], json.dumps(ex["tools"]), entry["extraction_method"], entry["timestamp"]])
    return total


# Engine text labels -> register text IDs. Matching is on the label's leading words only so
# the long edition notes in the engine records do not matter. Unmatched labels get NULL.
_TEXT_KEYS = [("aryabhatiya", "TXT-01"), ("tantrasangraha", "TXT-02"), ("ganita-yukti", "TXT-03"),
              ("drgganita", "TXT-04"), ("suryasiddhanta", "TXT-05"), ("indian astronomy", "TXT-07"),
              ("hindu zodiac", "TXT-08"), ("dulakara", "TXT-09"), ("karanaratna", "TXT-10")]


def text_id_for(label: str):
    low = label.lower()
    for key, tid in _TEXT_KEYS:
        if low.startswith(key) or (key in low[:40]):
            return tid
    return None


def load_rules(con):
    """Load the engine's rule records read-only. They keep their own IDs; nothing is renamed."""
    n = 0
    for f in sorted(RULES_DIR.glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            has_page = bool(r.get("pdf_pages"))
            prov = "page_cited" if has_page else "reference_only"
            con.execute("INSERT INTO rules (rule_id, rule_kind, text_label, text_id, reference, title, formula, "
                        "implemented_by, engine_status, sanskrit, sanskrit_verified, gloss, notes, "
                        "provenance_status, review_status, source_file) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                r["id"], "computational", r["text"], text_id_for(r["text"]), r["reference"], r["title"],
                r["formula"], r["implemented_by"], r["status"], r.get("sanskrit"),
                bool(r.get("sanskrit_verified")), r.get("gloss") or None, r.get("notes") or None,
                prov, "unreviewed", f.relative_to(REPO).as_posix()])
            pages = r.get("pdf_pages") or []
            for pg in (pages if isinstance(pages, list) else [pages]):
                con.execute("INSERT INTO rule_sources VALUES (?,?)", [r["id"], str(pg)])
            n += 1
    return n


def load_experiment1(con):
    """Load the pre-registered Experiment 1 rules after checking each against the Silver schema."""
    import csv
    schema = json.loads((SILVER / "schema" / "silver_rule.schema.json").read_text(encoding="utf-8"))
    src = SILVER / "rules" / "experiment1_rules.json"
    rules = json.loads(src.read_text(encoding="utf-8"))
    for r in rules:
        try:
            jsonschema.validate(r, schema)
        except jsonschema.ValidationError as exc:
            sys.exit(f"{r.get('rule_id')}: not valid against silver_rule.schema.json: {exc.message}")
    texts = {t["id"] for t in json.loads(REGISTER.read_text(encoding="utf-8"))["texts"]}
    for r in rules:
        if r["text_id"] not in texts:
            sys.exit(f"{r['rule_id']}: unknown text {r['text_id']}")
        con.execute("INSERT INTO rules (rule_id, rule_kind, text_id, reference, title, provenance_status, "
                    "review_status, source_file, statement, condition, preregistration) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            r["rule_id"], r["rule_kind"], r["text_id"], r["source"]["reference"], r["title"],
            r["provenance_status"], r["review"]["status"], src.relative_to(REPO).as_posix(),
            r["statement"], json.dumps(r["condition"], ensure_ascii=False),
            json.dumps(r["preregistration"])])
    with (SILVER / "rules" / "experiment1_reported_results.csv").open(newline="", encoding="utf-8") as fh:
        rows = [(x["rule_id"], float(x["discovery_p"]), float(x["holdout_p"]), x["holdout_observed"],
                 x["holdout_expected"]) for x in csv.DictReader(fh)]
    ids = {r["rule_id"] for r in rules}
    if {x[0] for x in rows} != ids:
        sys.exit("reported results do not match the Experiment 1 rule IDs")
    con.executemany("INSERT INTO reported_results VALUES (?,?,?,?,?)", rows)
    return len(rules)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(SILVER / "silver.duckdb"))
    args = ap.parse_args()

    audit = AuditLog(str(AUDIT))
    registered, extracts = check_bronze(audit)

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_suffix(".building")
    for stale in (tmp, Path(str(tmp) + ".wal")):
        stale.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute(DDL)

    texts = json.loads(REGISTER.read_text(encoding="utf-8"))["texts"]
    con.executemany("INSERT INTO texts VALUES (?,?,?,?,?,?)",
                    [(t["id"], t["title"], t.get("author"), t.get("kind"), t.get("language"), t.get("role")) for t in texts])
    con.executemany("INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
                    [(a["artifact_id"], a["text_id"], a["original_filename"], a["sha256"], a["bytes"],
                      a["media_type"], a["registered_at"]) for a in registered.values()])
    n_pages = load_pages(con, extracts)
    n_rules = load_rules(con) + load_experiment1(con)
    con.close()
    for suffix in ("", ".wal"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    tmp.replace(db)          # the old database is replaced only after a complete build

    print(f"built {db}: {len(texts)} texts, {len(registered)} artifacts, {len(extracts)} extracts, "
          f"{n_pages} pages, {n_rules} rules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
