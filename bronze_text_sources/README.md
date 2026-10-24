# Bronze Text Sources Layer

## Overview

The **Bronze Layer** is the immutable foundation of the Lakehouse architecture for the Ancient Mathematical Astronomy Seismic Pattern Analysis project. It contains raw artifacts from text ingestion pipelines without any modification or transformation.

This layer serves as the single source of truth for all extracted astronomical rules from classical mathematical astronomy treatises (Sūryasiddhānta, Dṛggaṇita, and others), with complete provenance tracking and quality metrics.

## Architecture

The bronze layer follows the medallion (lakehouse) pattern:

```
Raw PDFs (Classical Mathematical Astronomy Treatises)
   ↓
[Bronze Layer] ← Immutable raw text + metadata
   ↓
[Silver Layer] ← Parsed, validated, normalized rules
   ↓
[Gold Layers] ← Study-specific datasets
   ├─ Study 1 (Rule-First Analysis)
   └─ Study 2 (Data-First Analysis)
```

## Components

### 1. `schema/rule.schema.json`

**Purpose:** Defines the JSON schema for all astronomical rules extracted from classical treatises in the bronze layer.

**Key Fields:**
- `rule_id` - Unique identifier for the astronomical rule (e.g., SUR-0042, DRK-0127)
- `text` - Full rule text in transliterated Sanskrit (IAST)
- `source_reference` - Treatise, chapter, and verse information
- `parameters` - Astronomical parameters with their mathematical decodings
- `interpretation_judgments` - Ambiguities in interpretation, mathematical modernization notes, confidence levels
- `provenance` - Complete link to source PDF, page number, extraction method, OCR confidence
- `review_status` - Current approval status and reviewer notes

**Why This Matters:**
The schema captures not just the rule content, but also the complete audit trail back to the original PDF. This enables reproducible research and detection of OCR errors.

### 2. `pipeline/audit.py`

**Purpose:** Append-only hash-chained audit log for tracking file provenance during ingestion of classical texts.

**Key Functions:**

- `sha256_file(file_path)` - Calculate SHA256 hash of source documents
- `record(source_file, file_hash, extraction_method, metadata)` - Log a new ingestion event
- `verify()` - Verify integrity of the entire audit chain
- `get_entry(index)` - Retrieve a specific entry
- `list_entries()` - Get all logged entries

**How It Works:**
```
Entry 1: source_file="treatise_v1.pdf", file_hash=SHA256(...), previous_hash=""
Entry 2: source_file="treatise_v2.pdf", file_hash=SHA256(...), previous_hash=SHA256(Entry1)
Entry 3: ...
```

Each entry is hash-chained to the previous one, so `verify()` detects:
- Edited entries
- Reordered entries
- Deleted entries (except the last one, which nothing refers to; anchor the head hash externally to cover that)

`verify_source(path)` also checks that a file on disk still matches the hash recorded for it.

**Usage:**
```python
from bronze_text_sources.pipeline.audit import AuditLog

audit = AuditLog("ingestion_audit.jsonl")

# Record a new ingestion
entry = audit.record(
    source_file="Sūryasiddhānta_edition_v2.pdf",
    file_hash="abc123...",
    extraction_method="ocr_assisted",
    metadata={"pages": "1-50", "language": "Sanskrit"}
)

# Verify integrity
if audit.verify():
    print("Audit chain is valid")
```

### 3. `pipeline/ingest.py`

**Purpose:** Text ingestion pipeline for classical mathematical astronomy treatises, handling OCR quality assessment, transliteration, and validation.

**Key Functions:**

- `ocr_quality_check(text, min_confidence)` - Detect OCR errors and estimate quality
- `transliterate_iast(text)` - Convert to IAST (International Alphabet of Sanskrit Transliteration)
- `normalize_unicode(text)` - NFC Unicode normalization for consistent text representation
- `extract_word_confidence(text)` - Per-word confidence scoring for OCR output
- `ingest(text, source_reference)` - Run the full ingestion pipeline

**Quality Metrics (v2):**
- Token-level gate: flags tokens that mix digits and letters (e.g. `5ā`, `l0ka`) and tokens with characters outside the IAST/Devanagari set
- Citation tokens (`vv.15-18`, `p.47`, `Ch.1`) and pure numerals are not flagged
- `estimated_confidence` = share of scored tokens that are clean (a screening heuristic, not calibrated OCR confidence)
- `transliterate_iast` is a placeholder: `ingest()` returns `iast_text: None` and `transliteration_status: "not_implemented"` until Aksharamukha is wired in
- Returns `pipeline_status`: either "ready_for_bronze" or "needs_review"

**Usage:**
```python
from bronze_text_sources.pipeline.ingest import TextIngestionPipeline

pipeline = TextIngestionPipeline()

result = pipeline.ingest(
    text="raw_ocr_output_here",
    source_reference={
        "treatise": "Sūryasiddhānta",
        "chapter": 2,
        "verse": "42"
    }
)

# Check if ingestion passed quality gate
if result['is_valid_ocr']:
    print(f"Confidence: {result['ocr_metrics']['estimated_confidence']}")
    # Proceed to bronze storage
else:
    print(f"Issues detected: {result['ocr_metrics']['detected_issues']}")
    # Flag for manual review
```

### 4. Source register (`manifest/source_register.json`)

Generated by `pipeline/build_register.py`. It lists:
- **Texts**: 13 classical texts and editions named in the project, with role, consumers and edition as recorded. Each lists the raw scans registered for it (`raw_artifacts`: id, original filename, SHA-256, size, type, processing status), read from the audit log. `scan_in_repo` is `false` because scans are never stored in git.
- **Unresolved**: Cunningham (named as a rule source in `analysis/README.md`; no source located).
- **Engine extracts**: the 7 rule files in `indic-ganita-engine/indic_engine/rules/data/`, hashed, with per-file provenance counts (page references, Sanskrit verification, bronze-schema conformance).
- **Data files**: 11 catalogue-derived or curated CSV/notebook inputs, hashed, with row counts and the catalogue named where the repo states it.

Each hashed artifact is recorded in `audit/ingestion_audit.jsonl` when its hash is new or has changed. Re-running with no changes adds nothing.

```
python3 bronze_text_sources/pipeline/build_register.py
python3 -m unittest discover -s bronze_text_sources/tests
```

### 5. Raw file registration (`pipeline/register_raw_batch.py`)

Scans and PDFs are large and stay out of git (`bronze_text_sources/raw/` is ignored except `index.csv`). They are registered by fingerprint instead. File names carry no meaning: the text each file belongs to is written in `raw/index.csv`.

```
python3 bronze_text_sources/pipeline/suggest_raw_index.py    # proposes raw/index.csv from the file names
# review index.csv: fill any blank text_id, or write SKIP for a file not to register yet
python3 bronze_text_sources/pipeline/register_raw_batch.py   # registers every file, appends to the audit chain
python3 bronze_text_sources/pipeline/build_register.py       # refreshes the source register
```

- Each record is validated against `schema/source_artifact.schema.json`. The media type is detected from the file's first bytes, not its extension.
- The batch refuses to start if a file is missing from `index.csv`, a listed file is absent, or a text ID is blank or unknown.
- A file whose bytes changed after registration is refused, so bronze files stay immutable.
- `register_raw_file.py TXT-nn path` registers a single file. Requires the `jsonschema` package.

### 6. Text extraction (after registration)

Needs `brew install poppler tesseract tesseract-lang` (Sanskrit, Hindi and English packs).

```
python3 bronze_text_sources/pipeline/extract_text.py --plan   # what would run, changes nothing
python3 bronze_text_sources/pipeline/extract_text.py --jobs 2 # extract; safe to stop and re-run
```

Pages with a real text layer are read directly; the rest are OCR'd. Output goes to
`extracted/<artifact_id>/` (per-page text, `pages.jsonl` with method and confidence, and
`text.txt`), and one audit entry per file records the tools, languages and settings.
Quality judgment and IAST conversion belong to the Silver layer.

## Workflow

### Ingesting a New Treatise

1. **Extract Text from PDF**
   - Use OCR (Tesseract) to extract raw text from classical mathematical astronomy texts
   - Run through quality checks

2. **Run Through Pipeline**
   ```python
   pipeline = TextIngestionPipeline()
   result = pipeline.ingest(raw_text, source_metadata)
   ```

3. **Log in Audit Trail**
   ```python
   audit = AuditLog("audit.jsonl")
   pdf_hash = audit.sha256_file("treatise.pdf")
   audit.record("treatise.pdf", pdf_hash, "ocr_assisted")
   ```

4. **Validate Schema**
   - Ensure extracted rules match `rule.schema.json`
   - All required astronomical and mathematical fields populated

5. **Store in Bronze**
   - Immutable storage (no updates after ingestion)
   - Full provenance metadata preserved for reproducibility

### Quality Assurance

- OCR confidence must exceed threshold (default 0.85)
- Per-word confidence scores guide manual review
- Hash chains prevent tampering with source records
- Audit trail enables complete reproducibility of ingestion process

## Integration with Silver and Gold Layers

The Bronze Layer outputs are **immutable inputs** to the Silver Layer, which:
- Parses rule syntax and mathematical notation
- Validates parameter decodings and astronomical calculations
- Normalizes rules across different treatises
- Flags ambiguities in mathematical interpretation for expert review

The Silver Layer feeds into multiple Gold Layers, each representing a separate study or analysis (Study 1, Study 2, etc.), all sharing the same canonical Silver Layer of normalized astronomical rules.

## Files and Formats

```
bronze_text_sources/
├── README.md (this file)
├── REVIEW.md                     # Code review findings for this layer
├── schema/
│   ├── rule.schema.json          # JSON Schema for astronomical rules
│   └── source_artifact.schema.json  # JSON Schema for a registered raw file
├── pipeline/
│   ├── __init__.py
│   ├── audit.py                  # Hash-chained audit log
│   ├── ingest.py                 # OCR gate, normalization, per-token scores
│   ├── build_register.py         # Builds the source register and audit entries
│   ├── suggest_raw_index.py      # Proposes raw/index.csv from file names
│   ├── register_raw_batch.py     # Registers every file listed in raw/index.csv
│   └── register_raw_file.py      # Registers one raw file (used by the batch)
├── raw/                          # Scans and PDFs (git-ignored); only index.csv is tracked
│   └── index.csv                 # path,text_id for each raw file
├── manifest/
│   └── source_register.json      # Generated register of texts, engine extracts and data files
├── audit/
│   └── ingestion_audit.jsonl     # Append-only audit chain
└── tests/
    ├── test_audit.py
    └── test_ingest.py
```

## Dependencies

- `hashlib` - SHA256 hashing (stdlib)
- `json` - JSON handling (stdlib)
- `unicodedata` - Unicode normalization (stdlib)
- `re` - Regular expressions (stdlib)
- `Aksharamukha` - Sanskrit transliteration (production)
- `Tesseract` - OCR for classical text extraction (upstream in pipeline)

## Key Principles

1. **Immutability** - Once ingested, bronze records never change
2. **Provenance** - Complete audit trail from source PDF to processed artifact
3. **Quality First** - OCR and confidence metrics captured at ingestion time
4. **Reproducibility** - Hash chains and audit logs enable verification of all processing steps
5. **Foundation** - Solid, transparent base for all downstream analysis layers

## Future Extensions

- Integration with external Sanskrit lexicon and mathematical notation resources
- Automated OCR confidence estimation using machine learning models
- Support for manuscripts in other classical writing systems
- Batch ingestion API for bulk treatise processing
- Integration with caching mechanisms for efficient rule lookups
- Support for cross-referencing rules across multiple classical mathematical astronomy treatises

---

**Related Documentation:**
- Parent project: [AI-Catastrophe-Analytics](../README.md)
- Engine rule files (transcribed extracts): `indic-ganita-engine/indic_engine/rules/data/`
- Silver and Gold layers: not yet in this repository (no `docs/silver_layer/` or `docs/gold_layers/` exists)
- Design document: see `REVIEW.md` for the findings this layer was checked against
