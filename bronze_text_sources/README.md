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

Each entry is hash-chained to the previous one, making it impossible to:
- Edit an entry without breaking the chain
- Delete entries without detection
- Reorder entries

This ensures complete reproducibility and data integrity throughout the ingestion pipeline.

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

**Quality Metrics:**
- Detects common OCR confusions (0/O, l/1, 5/S, 8/B)
- Assigns confidence scores to individual words
- Flags unusual word lengths or special characters
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
├── schema/
│   └── rule.schema.json          # JSON Schema for astronomical rules
└── pipeline/
    ├── audit.py                  # Audit log implementation
    ├── ingest.py                 # Ingestion pipeline
    └── __init__.py               # Package marker
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
- Silver Layer: `docs/silver_layer/`
- Gold Layers: `docs/gold_layers/`
- Design Document: `docs/design/`
