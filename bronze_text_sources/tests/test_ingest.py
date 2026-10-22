"""Tests for the bronze ingestion quality gate. Run: python3 -m unittest discover -s bronze_text_sources/tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from ingest import TextIngestionPipeline  # noqa: E402


class OcrGateTests(unittest.TestCase):

    def setUp(self):
        self.p = TextIngestionPipeline()

    def test_clean_iast_passes(self):
        ok, m = self.p.ocr_quality_check("sūryasiddhānta mahāyuga yugādi ṭīkā ṛṣi")
        self.assertTrue(ok)
        self.assertEqual(m["detected_issues"], [])

    def test_ordinary_iast_l_is_not_flagged(self):
        # The previous gate flagged every 'l' (one/L confusion) and failed mixed texts.
        ok, m = self.p.ocr_quality_check("loka līlā kalpa sūkṣma")
        self.assertTrue(ok)
        self.assertEqual(m["detected_issues"], [])

    def test_verse_numerals_and_citations_are_not_flagged(self):
        ok, m = self.p.ocr_quality_check("TS 1.9 vv.15-18 p.47 Ch.1 1,577,917,828 10")
        self.assertTrue(ok)
        self.assertEqual(m["detected_issues"], [])

    def test_digit_letter_mixed_token_is_flagged(self):
        _, m = self.p.ocr_quality_check("5ā l0ka kalpa")
        self.assertIn("mixed_digit_letter: 2", m["detected_issues"])
        self.assertIn("l0ka", m["flagged_examples"])

    def test_unexpected_character_is_flagged(self):
        _, m = self.p.ocr_quality_check("sūrya@siddhānta")
        self.assertIn("unexpected_character: 1", m["detected_issues"])

    def test_confidence_is_share_of_clean_tokens(self):
        _, m = self.p.ocr_quality_check("kalpa yuga 5ā 7x")
        self.assertEqual(m["scored_tokens"], 4)
        self.assertEqual(m["estimated_confidence"], 0.5)

    def test_empty_text_fails(self):
        ok, m = self.p.ocr_quality_check("   ")
        self.assertFalse(ok)
        self.assertEqual(m["estimated_confidence"], 0.0)

    def test_decomposed_diacritics_are_normalized_before_checks(self):
        decomposed = "āṭ"  # a + combining macron, t + combining dot below
        self.assertEqual(self.p.normalize_unicode(decomposed), "āṭ")


class IngestTests(unittest.TestCase):

    def setUp(self):
        self.p = TextIngestionPipeline()
        self.ref = {"treatise": "Tantrasaṅgraha", "chapter": 1, "verse": "9"}

    def test_clean_text_is_ready_for_bronze(self):
        out = self.p.ingest("mahāyuga yugādi kalpa", self.ref)
        self.assertEqual(out["pipeline_status"], "ready_for_bronze")
        self.assertTrue(out["is_valid_ocr"])

    def test_noisy_text_needs_review(self):
        out = self.p.ingest("5ā l0ka 7x 9y 3z", self.ref)
        self.assertEqual(out["pipeline_status"], "needs_review")

    def test_iast_is_not_claimed_while_transliteration_is_placeholder(self):
        out = self.p.ingest("kalpa", self.ref)
        self.assertIsNone(out["iast_text"])
        self.assertEqual(out["transliteration_status"], "not_implemented")

    def test_word_confidence_one_per_token(self):
        out = self.p.ingest("kalpa yuga manvantara", self.ref)
        self.assertEqual([w["position"] for w in out["word_confidences"]], [0, 1, 2])
        self.assertTrue(all(0.0 <= w["confidence"] <= 1.0 for w in out["word_confidences"]))

    def test_source_reference_is_carried_through(self):
        self.assertEqual(self.p.ingest("kalpa", self.ref)["source_reference"], self.ref)


if __name__ == "__main__":
    unittest.main()
