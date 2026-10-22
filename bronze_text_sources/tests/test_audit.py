"""Tests for the hash-chained audit log. Run: python3 -m unittest discover -s bronze_text_sources/tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from audit import AuditLog  # noqa: E402


class AuditLogTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.log_path = self.dir / "audit.jsonl"
        self.log = AuditLog(self.log_path)
        self.src = self.dir / "source.txt"
        self.src.write_bytes("sūryasiddhānta chapter one\n".encode("utf-8"))
        self.digest = self.log.sha256_file(self.src)
        for name in ("a", "b", "c"):
            self.log.record(f"{name}.csv", f"hash-{name}", "registered_by_reference", {"n": name})

    def tearDown(self):
        self._tmp.cleanup()

    def _lines(self):
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def _write_lines(self, lines):
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_fresh_chain_verifies(self):
        self.assertTrue(self.log.verify())

    def test_chain_survives_reload(self):
        self.assertTrue(AuditLog(self.log_path).verify())
        self.assertEqual(len(AuditLog(self.log_path).list_entries()), 3)

    def test_timestamps_are_timezone_aware(self):
        ts = self.log.list_entries()[0]["timestamp"]
        self.assertTrue(ts.endswith("+00:00"), ts)

    def test_detects_edited_entry(self):
        lines = self._lines()
        lines[1] = lines[1].replace("hash-b", "hash-X")
        self._write_lines(lines)
        self.assertFalse(AuditLog(self.log_path).verify())

    def test_detects_reordered_entries(self):
        lines = self._lines()
        lines[1], lines[2] = lines[2], lines[1]
        self._write_lines(lines)
        self.assertFalse(AuditLog(self.log_path).verify())

    def test_detects_deleted_middle_entry(self):
        lines = self._lines()
        del lines[1]
        self._write_lines(lines)
        self.assertFalse(AuditLog(self.log_path).verify())

    def test_detects_wrong_first_previous_hash(self):
        entry = json.loads(self._lines()[0])
        entry["previous_hash"] = "not-empty"
        lines = self._lines()
        lines[0] = json.dumps(entry)
        self._write_lines(lines)
        self.assertFalse(AuditLog(self.log_path).verify())

    def test_verify_source_matches_recorded_hash(self):
        self.log.record(str(self.src), self.digest, "transcribed_extract")
        self.assertTrue(self.log.verify_source(str(self.src)))

    def test_verify_source_detects_changed_file(self):
        self.log.record(str(self.src), self.digest, "transcribed_extract")
        self.src.write_bytes(b"edited\n")
        self.assertFalse(self.log.verify_source(str(self.src)))

    def test_verify_source_detects_missing_file(self):
        self.log.record(str(self.src), self.digest, "transcribed_extract")
        self.src.unlink()
        self.assertFalse(self.log.verify_source(str(self.src)))

    def test_verify_source_none_when_never_recorded(self):
        self.assertIsNone(self.log.verify_source(str(self.src)))

    def test_latest_entry_wins_for_a_path(self):
        self.log.record("x.csv", "first", "registered_by_reference")
        self.log.record("x.csv", "second", "registered_by_reference")
        self.assertEqual(self.log.latest_for("x.csv")["file_hash"], "second")

    def test_corrupt_line_raises_with_line_number(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("{not json\n")
        with self.assertRaisesRegex(ValueError, "line 4"):
            AuditLog(self.log_path)


if __name__ == "__main__":
    unittest.main()
