#!/usr/bin/env python3
"""Register raw files in the bronze layer.

Single file (from the repository root):
    python3 bronze_text_sources/pipeline/register_raw_file.py TXT-01 path/to/file

All files under bronze_text_sources/raw/ (see register_raw_batch.py):
    python3 bronze_text_sources/pipeline/register_raw_batch.py

The text ID is always given explicitly, never inferred from the filename.
The file is fingerprinted and its media type is detected from its leading bytes.
A record is validated against schema/source_artifact.schema.json and appended to
the audit chain. The file is not copied or changed. Re-running on unchanged bytes
adds nothing. Changed bytes are refused.
"""
import codecs
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path.cwd()
BRONZE = REPO / "bronze_text_sources"
sys.path.insert(0, str(BRONZE / "pipeline"))
from audit import AuditLog  # noqa: E402

try:
    import jsonschema
except ImportError:
    sys.exit("The 'jsonschema' package is required. Install it with:  python3 -m pip install jsonschema")

REGISTER = BRONZE / "manifest" / "source_register.json"
AUDIT = BRONZE / "audit" / "ingestion_audit.jsonl"
RAW = BRONZE / "raw"
SCHEMA = BRONZE / "schema" / "source_artifact.schema.json"

_SIGNATURES = [
    (b"%PDF", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"PK\x03\x04", "application/zip"),
]


def media_type(path: Path) -> str:
    """Detect type from the first bytes only, so very large files are not loaded into memory."""
    with path.open("rb") as fh:
        head = fh.read(65536)
    for magic, mime in _SIGNATURES:
        if head.startswith(magic):
            return mime
    try:
        # Incremental decoder tolerates a multi-byte character cut off at the 64 KB boundary.
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
        return "text/plain"
    except UnicodeDecodeError:
        return "application/octet-stream"


def known_texts() -> dict:
    return {t["id"]: t for t in json.loads(REGISTER.read_text(encoding="utf-8"))["texts"]}


def register(text_id: str, file_arg: str, texts: dict = None, audit: AuditLog = None):
    """Register one file. Returns (ok, message). Does not exit."""
    texts = texts if texts is not None else known_texts()
    if text_id not in texts:
        return False, f"Unknown text ID {text_id}. Valid IDs: {', '.join(sorted(texts))}"

    path = (REPO / file_arg).resolve()
    if not path.is_file():
        return False, f"File not found: {file_arg}"
    if RAW.resolve() not in path.parents:
        return False, f"File must be inside {RAW.relative_to(REPO)}/"

    audit = audit or AuditLog(str(AUDIT))
    rel = path.relative_to(REPO).as_posix()
    digest = audit.sha256_file(str(path))
    previous = audit.latest_for(rel)

    if previous and previous["file_hash"] != digest:
        return False, (f"REFUSED {rel}: bytes changed since registration "
                       f"(recorded {previous['file_hash'][:12]}..., current {digest[:12]}...). "
                       "Restore the original bytes, or place the new file at a new path.")
    if previous:
        return True, f"unchanged {rel}"

    record = {
        "artifact_id": f"RAW-{digest[:12]}",
        "text_id": text_id,
        "original_filename": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "media_type": media_type(path),
        "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "processing_status": "registered",
        "extraction_method": "registered_by_reference",
    }
    try:
        jsonschema.validate(record, json.loads(SCHEMA.read_text(encoding="utf-8")))
    except jsonschema.ValidationError as exc:
        return False, f"REFUSED {rel}: record failed schema validation: {exc.message}"

    audit.record(rel, digest, "registered_by_reference", {"artifact": record})
    return True, f"registered {rel} as {record['artifact_id']} for {text_id} ({texts[text_id]['title']})"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ok, message = register(sys.argv[1], sys.argv[2])
    print(message)
    if ok:
        audit = AuditLog(str(AUDIT))
        print("audit chain:", "OK" if audit.verify() else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
