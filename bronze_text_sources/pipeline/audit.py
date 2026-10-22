"""
Append-only audit log for bronze layer ingestion of classical mathematical astronomy texts.

Tracks provenance of ingested files with hash-chained verification
to detect edited, deleted, or reordered entries. Uses SHA256 file hashing
to create an immutable record linking processed artifacts back to source documents
containing ancient mathematical astronomy knowledge.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class AuditLog:
    """Append-only hash-chained audit log for tracking provenance in ancient text ingestion."""

    def __init__(self, log_path: str):
        """
        Initialize the audit log for classical mathematical astronomy text ingestion.
        
        Args:
            log_path: Path to the audit log file
        """
        self.log_path = Path(log_path)
        self.entries: List[Dict[str, Any]] = []
        self._load_existing_log()

    def _load_existing_log(self) -> None:
        """Load existing audit log entries from disk."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.entries.append(json.loads(line))

    def sha256_file(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a classical text source file.
        
        Args:
            file_path: Path to the source document
            
        Returns:
            Hexadecimal SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
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
        Record a new entry in the audit log for ingested classical texts.
        
        Args:
            source_file: Path to the source document containing ancient mathematical astronomy knowledge
            file_hash: SHA256 hash of the source file
            extraction_method: How the text was processed (manual, ocr_assisted, etc.)
            metadata: Additional metadata about the extraction
            
        Returns:
            The recorded entry as a dictionary
        """
        # Calculate hash of the previous entry (or empty string for first entry)
        previous_hash = ""
        if self.entries:
            previous_entry_str = json.dumps(self.entries[-1], sort_keys=True)
            previous_hash = hashlib.sha256(previous_entry_str.encode()).hexdigest()

        # Create new entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_file": source_file,
            "file_hash": file_hash,
            "extraction_method": extraction_method,
            "previous_hash": previous_hash,
            "metadata": metadata or {},
        }

        # Add to in-memory log
        self.entries.append(entry)

        # Append to file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        return entry

    def verify(self) -> bool:
        """
        Verify the integrity of the audit log for ancient text ingestion.
        
        Checks that:
        1. All entries have valid hash chaining
        2. No entries have been deleted or reordered
        3. All hashes match the expected values
        
        Returns:
            True if the log is valid and unaltered, False otherwise
        """
        if not self.entries:
            return True

        # Verify first entry has empty previous_hash
        if self.entries[0].get("previous_hash") != "":
            return False

        # Verify hash chaining
        for i, entry in enumerate(self.entries):
            if i > 0:
                # Calculate what the previous entry's hash should be
                previous_entry_str = json.dumps(self.entries[i - 1], sort_keys=True)
                expected_previous_hash = hashlib.sha256(
                    previous_entry_str.encode()
                ).hexdigest()

                # Check that this entry's previous_hash matches
                if entry.get("previous_hash") != expected_previous_hash:
                    return False

        return True

    def get_entry(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get an entry by index.
        
        Args:
            index: Index of the entry (0-based)
            
        Returns:
            The entry dictionary, or None if index is out of bounds
        """
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def list_entries(self) -> List[Dict[str, Any]]:
        """
        Get all audit log entries from classical text ingestion.
        
        Returns:
            List of all entries in the audit chain
        """
        return self.entries.copy()
