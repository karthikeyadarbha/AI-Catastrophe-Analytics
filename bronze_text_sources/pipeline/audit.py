"""
Append-only, hash-chained audit log for bronze-layer ingestion of classical
mathematical astronomy texts and the data files built on them.

Each entry stores the SHA-256 of a source artifact and the SHA-256 of the
previous entry, so editing, deleting or reordering entries breaks the chain
and is detected by ``verify()``. ``verify_source()`` additionally checks that
a file on disk still matches the hash recorded for it.

Known limit: the chain cannot detect removal of the *last* entry, because no
later entry refers to it. Anchor the head hash externally (e.g. in a git tag)
if that matters.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List


def _entry_hash(entry: Dict[str, Any]) -> str:
    """Canonical hash of an entry: SHA-256 over its sorted-key JSON form."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


class AuditLog:
    """Append-only hash-chained audit log for provenance in ancient text ingestion."""

    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to the JSON-lines audit log file (created on first record)
        """
        self.log_path = Path(log_path)
        self.entries: List[Dict[str, Any]] = []
        self._load_existing_log()

    def _load_existing_log(self) -> None:
        """Load existing audit log entries from disk."""
        if self.log_path.exists():
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for lineno, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            self.entries.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"{self.log_path}: line {lineno} is not valid JSON"
                            ) from exc

    def sha256_file(self, file_path: str) -> str:
        """
        Calculate SHA-256 of a source file.

        Args:
            file_path: Path to the source document

        Returns:
            Hexadecimal SHA-256 digest
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def record(
        self,
        source_file: str,
        file_hash: str,
        extraction_method: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Append a new entry to the audit log.

        Args:
            source_file: Path (repo-relative, by convention) of the source artifact
            file_hash: SHA-256 of the source artifact
            extraction_method: How the artifact entered the layer
                (e.g. ``registered_by_reference``, ``transcribed_extract``, ``ocr_assisted``)
            metadata: Additional descriptive metadata

        Returns:
            The recorded entry
        """
        previous_hash = _entry_hash(self.entries[-1]) if self.entries else ""

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "file_hash": file_hash,
            "extraction_method": extraction_method,
            "previous_hash": previous_hash,
            "metadata": metadata or {},
        }

        self.entries.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        return entry

    def verify(self) -> bool:
        """
        Verify the hash chain of the loaded log.

        Checks that the first entry has an empty ``previous_hash`` and that every
        later entry's ``previous_hash`` equals the hash of the entry before it.
        This detects edited, deleted (except the last entry) and reordered entries.

        Returns:
            True if the chain is intact, False otherwise
        """
        if not self.entries:
            return True

        if self.entries[0].get("previous_hash") != "":
            return False

        for i in range(1, len(self.entries)):
            if self.entries[i].get("previous_hash") != _entry_hash(self.entries[i - 1]):
                return False

        return True

    def latest_for(self, source_file: str) -> Optional[Dict[str, Any]]:
        """Return the most recent entry recorded for ``source_file``, or None."""
        for entry in reversed(self.entries):
            if entry.get("source_file") == source_file:
                return entry
        return None

    def verify_source(self, source_file: str) -> Optional[bool]:
        """
        Check that a file on disk still matches the hash recorded for it.

        Args:
            source_file: Path as it was passed to ``record()``

        Returns:
            True if the file matches its latest recorded hash,
            False if it differs or is missing,
            None if no entry exists for it
        """
        entry = self.latest_for(source_file)
        if entry is None:
            return None
        if not Path(source_file).exists():
            return False
        return self.sha256_file(source_file) == entry["file_hash"]

    def get_entry(self, index: int) -> Optional[Dict[str, Any]]:
        """Return the entry at ``index`` (0-based), or None if out of range."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def list_entries(self) -> List[Dict[str, Any]]:
        """Return a copy of all entries in chain order."""
        return self.entries.copy()
