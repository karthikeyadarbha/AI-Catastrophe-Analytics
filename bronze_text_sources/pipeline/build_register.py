"""
Build the bronze source register and record its artifacts in the audit chain.

What it does
  1. Lists every classical text, data file and engine extract the project depends on
     (the TEXTS, DATA_FILES and ENGINE_FILES tables below).
  2. For files present in the repo: computes SHA-256, row counts, and per-file provenance
     statistics from the engine rule files (page references, Sanskrit verification,
     bronze-schema conformance).
  3. Writes manifest/source_register.json.
  4. Appends to audit/ingestion_audit.jsonl one entry per artifact whose hash is new or
     has changed since its last recorded entry. Re-running with no changes adds nothing.

Scans are never stored in git. Each text lists the raw files registered for it in the audit
log (see register_raw_batch.py): artifact id, original filename, SHA-256, size, type and
processing status. This is read from the audit log, not from disk, so the register is the
same on every machine. ``scan_in_repo`` stays false because the bytes are held outside the
repo. Use AuditLog.verify_source(path) on a machine that holds the scans to re-check them.

Run from anywhere:  python bronze_text_sources/pipeline/build_register.py
"""

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRONZE = HERE.parent
REPO = BRONZE.parent
sys.path.insert(0, str(HERE))
from audit import AuditLog  # noqa: E402

REGISTER_PATH = BRONZE / "manifest" / "source_register.json"
AUDIT_PATH = BRONZE / "audit" / "ingestion_audit.jsonl"
SCHEMA_PATH = BRONZE / "schema" / "rule.schema.json"

# --- Classical texts and editions named anywhere in the project -----------------------
# role/consumers are taken from FREEZE.md, the engine README, analysis/README.md and the
# preregistrations. Fields left as "not recorded" were not stated in the repo.
TEXTS = [
    dict(id="TXT-01", title="Āryabhaṭīya", author="Āryabhaṭa", kind="classical_treatise",
         edition="not recorded", language="Sanskrit",
         role="Alternate parameter set AB; Rsine differences; epicycle sizes",
         engine_file="indic-ganita-engine/indic_engine/rules/data/aryabhatiya.json"),
    dict(id="TXT-02", title="Tantrasaṅgraha", author="Nīlakaṇṭha Somayājī", kind="classical_treatise",
         edition="not recorded", language="Sanskrit",
         role="Primary parameter set TS; dhruvas; planetary algorithm; Mādhava tables",
         engine_file="indic-ganita-engine/indic_engine/rules/data/tantrasangraha.json"),
    dict(id="TXT-03", title="Gaṇitayuktibhāṣā", author="Jyeṣṭhadeva", kind="classical_treatise",
         edition="not recorded (book page 47 and PDF page 102 cited)", language="Sanskrit",
         role="Reasoning and series behind the Mādhava tables",
         engine_file="indic-ganita-engine/indic_engine/rules/data/yuktibhasa.json"),
    dict(id="TXT-04", title="Dṛggaṇita", author="Parameśvara", kind="classical_treatise",
         edition="ed. K. V. Sarma, Hoshiarpur 1963", language="Sanskrit",
         role="Kerala Dṛk system (v1.3.0); cross-check model",
         engine_file="indic-ganita-engine/indic_engine/rules/data/drgganita.json"),
    dict(id="TXT-05", title="Sūryasiddhānta (Vijñāna-bhāṣya)", author="Sūryasiddhānta; ed. Pt. Mahāvīr Prasād Śrīvāstava",
         kind="classical_treatise", edition="Hindi Vijñāna-bhāṣya edition, 838 pp. PDF", language="Sanskrit / Hindi",
         role="Siddhāntic mean and true planets, apsides, Rsine table (v1.4.0)",
         engine_file="indic-ganita-engine/indic_engine/rules/data/suryasiddhanta.json"),
    dict(id="TXT-06", title="Sūryasiddhānta with Raṅganātha's commentary", author="ed. R. C. Pandey (Chaukhamba)",
         kind="classical_treatise", edition="Sanskrit edition with Hindi translation, 387 pp.", language="Sanskrit",
         role="Numeral-by-numeral concordance with TXT-05 (rule SS.concordance, v1.4.1)",
         engine_file="indic-ganita-engine/indic_engine/rules/data/concordance.json"),
    dict(id="TXT-07", title="Indian Astronomy: A Source Book", author="Subbarayappa & Sarma",
         kind="secondary_source_book", edition="not recorded", language="English",
         role="Ayanāṃśa models, calendar, latitudes, conjunctions, visibility, occultations, inscriptions",
         engine_file="indic-ganita-engine/indic_engine/rules/data/sourcebook.json",
         notes="Read by OCR in v1.1.0; some items checked against page images (FREEZE.md)"),
    dict(id="TXT-08", title="Hindu Zodiac and Ancient Astronomy", author="K. Chandra Hari",
         kind="secondary_monograph", edition="not recorded", language="English",
         role="Primary definition of the Chandrahari ayanāṃśa (v1.2.0); Siddhāntic zero point",
         engine_file=None, cited_via=["TXT-07 records AY.ChandraHari, AY.ChandraHari.siddhantic"]),
    dict(id="TXT-09", title="Dulakara Ayanamsha", author="B.S.H. Indrasena", kind="secondary_paper",
         edition="2011 paper (PDF text layer)", language="English",
         role="Dulakara/Indrasena ayanāṃśa", engine_file=None,
         cited_via=["TXT-07 records AY.Dulakara"]),
    dict(id="TXT-10", title="Karaṇaratna (Deva) and Khaṇḍakhādyaka (Amaraja)", author="Deva; Amaraja",
         kind="classical_treatise", edition="not recorded (cited through TXT-07 section 14.5)", language="Sanskrit",
         role="Deva/Amaraja ayanāṃśa sign convention (interpretation choice 8)", engine_file=None,
         cited_via=["TXT-07 records SB.14.AY.deva"]),
    dict(id="TXT-11", title="Bṛhat Saṃhitā", author="Varāhamihira", kind="classical_treatise",
         edition="English translation (uploaded PDF) and Sanskrit scan; editions not recorded",
         language="Sanskrit / English",
         role="Ch. 32 earthquake mandalas; Rule-First and Data-First tests",
         engine_file=None, consumers=["analysis/experiment1_rule_first/03_brihat_samhita/",
                                      "analysis/experiment2_data_first/stage5_treatise_bin_scan.py"]),
    dict(id="TXT-12", title="Adbhuta Sāgara", author="Vallāla Sena", kind="classical_treatise",
         edition="ed. Murali Dhara Jha, Varanasi 1905 (chapter pp. 383-408)", language="Sanskrit",
         role="Earthquake quotations; authority lists tested in Rule-First study",
         engine_file=None, consumers=["analysis/experiment1_rule_first/04_adbhuta_sagara/"]),
    dict(id="TXT-13", title="Parāśara Tantra (as quoted)", author="Parāśara (known through quotations)",
         kind="classical_treatise_quoted", edition="not recorded; quoted within TXT-12", language="Sanskrit",
         role="Deadly 15 mandala lists (Rule-First study)", engine_file=None,
         consumers=["analysis/experiment1_rule_first/01_deadly15/",
                    "analysis/shared/inputs/earthquakes_parashara_tantra_mandalas.csv"]),
]
UNRESOLVED = [
    dict(id="UNR-01", title="Cunningham (regions)", where="analysis/README.md (listed as a rule source); reg.py",
         status="No edition or source file could be located in the repo"),
]

# --- Data files that carry data from catalogues or are derived from them ---------------
# Each entry: path (repo-relative), role, catalogue (who produced the events, where stated).
DATA_FILES = [
    dict(id="DAT-01", path="1850-1950-EQData-MAG5.csv", role="1850-1950 M5+ file for regional tests",
         catalogue="ISC-GEM (ids 'iscgem…', per preregistration_historical_test.md)"),
    dict(id="DAT-02", path="analysis/shared/inputs/big.csv", role="Modern worldwide set (role per file name; confirm)",
         catalogue="USGS ComCat (column schema)"),
    dict(id="DAT-03", path="analysis/shared/inputs/old_m65.csv", role="Older M6.5 set (confirm)",
         catalogue="ISC-GEM via USGS ComCat (ids 'iscgem…')"),
    dict(id="DAT-04", path="analysis/shared/inputs/histA_declustered.csv",
         role="Era A 1964-1989 M>=5.5, declustered (6,111 events per preregistration)",
         catalogue="ISC-GEM (ids 'iscgem…'), declustered"),
    dict(id="DAT-05", path="analysis/shared/inputs/histB_declustered.csv",
         role="Era B 1900-1963 M>=7.0, 670 events (per preregistration)", catalogue="ISC-GEM-based ('cent…' ids)"),
    dict(id="DAT-06", path="analysis/shared/inputs/quakes_m55_input.csv",
         role="M5.5+ 1990-2026, declustered, 9,368 events (Experiment 2 override set)",
         catalogue="USGS ComCat (ids 'usp…')"),
    dict(id="DAT-07", path="analysis/shared/inputs/earthquakes_parashara_tantra_mandalas.csv",
         role="Deadly 15 event table with mandala labels (curated)", catalogue="curated; source of each event not recorded"),
    dict(id="DAT-08", path="analysis/tgsai_corrected/deadly15_events.csv", role="Deadly 15 event list",
         catalogue="curated (same 15 events as DAT-07)"),
    dict(id="DAT-09", path="analysis/tgsai_corrected/Inputs_Mega_Quake_Topocentric_Analysis_corrected.csv",
         role="TG-SAI training inputs (corrected), derived with astral ephemeris", catalogue="derived"),
    dict(id="DAT-10", path="notebooks/megaquake/Inputs_Mega_Quake_Topocentric_Analysis.csv",
         role="Mega-quake topocentric inputs, derived with astral ephemeris", catalogue="derived"),
    dict(id="DAT-11", path="notebooks/megaquake/subduction_zones.csv", role="Subduction-zone geometry (reference)",
         catalogue="not recorded"),
]

# --- Engine rule files (transcribed extracts of the texts above) -----------------------
ENGINE_FILES = [t["engine_file"] for t in TEXTS if t.get("engine_file")]
RULE_SCHEMA_KEYS = ["rule_id", "text", "source_reference", "provenance"]  # required in rule.schema.json


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def needs_record(audit: AuditLog, rel: str, digest: str) -> bool:
    """True if the artifact has no audit entry yet, or its hash differs from the latest one."""
    last = audit.latest_for(rel)
    return last is None or last["file_hash"] != digest


def csv_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def engine_stats(path: Path) -> dict:
    rules = json.loads(path.read_text(encoding="utf-8"))
    return {
        "records": len(rules),
        "status": dict(Counter(r.get("status", "unknown") for r in rules)),
        "with_pdf_pages": sum(1 for r in rules if r.get("pdf_pages")),
        "sanskrit_verified": sum(1 for r in rules if r.get("sanskrit_verified") is True),
        "schema_conformant": sum(1 for r in rules if all(k in r for k in RULE_SCHEMA_KEYS)),
        "rule_ids": [r["id"] for r in rules],
    }


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build() -> dict:
    audit = AuditLog(AUDIT_PATH)
    recorded = []

    # Latest registered raw artifact per file, grouped by text ID (from the audit log).
    latest_raw = {}
    for e in audit.list_entries():
        art = e.get("metadata", {}).get("artifact")
        if art:
            latest_raw[e["source_file"]] = art
    raw_by_text = {}
    for art in latest_raw.values():
        raw_by_text.setdefault(art["text_id"], []).append(art)

    texts = []
    for t in TEXTS:
        entry = dict(t)
        arts = sorted(raw_by_text.get(t["id"], []), key=lambda a: a["original_filename"])
        entry["scan_in_repo"] = False
        entry["raw_artifacts"] = [
            {k: a[k] for k in ("artifact_id", "original_filename", "sha256", "bytes",
                               "media_type", "processing_status")}
            for a in arts
        ]
        entry["hash_status"] = (
            f"{len(arts)} scan(s) registered in the audit log; bytes are held outside the repo"
            if arts else "not verifiable: no scan registered"
        )
        texts.append(entry)

    engines = []
    for rel in ENGINE_FILES:
        path = REPO / rel
        stats = engine_stats(path)
        digest = sha256(path)
        ident = next(t["id"] for t in TEXTS if t.get("engine_file") == rel)
        artifact_id = f"ENG-{ident[-2:]}"
        engines.append({"artifact_id": artifact_id, "path": rel, "feeds": ident, "sha256": digest, **stats})
        if needs_record(audit, rel, digest):
            audit.record(rel, digest, "transcribed_extract",
                         {"artifact_id": artifact_id, "feeds": ident, "records": stats["records"]})
            recorded.append(rel)

    data = []
    for d in DATA_FILES:
        path = REPO / d["path"]
        if not path.exists():
            data.append({**d, "present": False})
            continue
        digest = sha256(path)
        item = {**d, "present": True, "sha256": digest, "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            item["rows"] = csv_rows(path)
        data.append(item)
        if needs_record(audit, d["path"], digest):
            audit.record(d["path"], digest, "registered_by_reference",
                         {"artifact_id": d["id"], "role": d["role"], "rows": item.get("rows")})
            recorded.append(d["path"])

    chain_ok = audit.verify()
    all_rules = sum(e["records"] for e in engines)
    summary = {
        "classical_texts_identified": len(TEXTS),
        "classical_texts_unresolved": len(UNRESOLVED),
        "texts_with_scan_in_repo": sum(1 for t in texts if t["scan_in_repo"]),
        "texts_with_registered_scan": sum(1 for t in texts if t["raw_artifacts"]),
        "raw_artifacts_registered": sum(len(t["raw_artifacts"]) for t in texts),
        "texts_with_engine_extract": len(engines),
        "data_files_registered": sum(1 for d in data if d.get("present")),
        "data_files_missing": sum(1 for d in data if not d.get("present")),
        "catalogues_named": ["ISC-GEM", "USGS ComCat"],
        "engine_rules_total": all_rules,
        "engine_rules_with_pdf_pages": sum(e["with_pdf_pages"] for e in engines),
        "engine_rules_sanskrit_verified": sum(e["sanskrit_verified"] for e in engines),
        "engine_rules_schema_conformant": sum(e["schema_conformant"] for e in engines),
        "audit_entries_added_this_run": len(recorded),
        "audit_chain_valid": chain_ok,
    }

    register = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "bronze_text_sources/pipeline/build_register.py",
        "repo_commit": git_head(),
        "schema_checked": str(SCHEMA_PATH.relative_to(REPO)),
        "summary": summary,
        "texts": texts,
        "unresolved_sources": UNRESOLVED,
        "engine_extracts": engines,
        "data_files": data,
    }
    REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTER_PATH.write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return register


if __name__ == "__main__":
    reg = build()
    print(json.dumps(reg["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {REGISTER_PATH.relative_to(REPO)}")
    print(f"audit: {AUDIT_PATH.relative_to(REPO)} (valid={reg['summary']['audit_chain_valid']})")
