"""
Text ingestion pipeline for the bronze layer of ancient mathematical astronomy treatises.

Quality gate, Unicode normalization and per-word scoring for text that has
already been OCR'd or transcribed. Transliteration to IAST is NOT implemented
yet (``transliterate_iast`` is a placeholder); ``ingest()`` reports this
explicitly rather than presenting unconverted text as IAST.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Letters allowed in a clean IAST or Devanagari transcription:
# Latin-1 and Latin Extended-A/B (ā ī ū ē ō ś ñ ...), Latin Extended Additional
# (ṭ ḍ ṇ ṣ ṛ ḷ ṃ ḥ ṅ ...), Devanagari, and combining marks.
_LETTERS = r"A-Za-zÀ-ɏḀ-ỿऀ-ॿ"
_PUNCT_EDGE = ".,;:!?()\"'-|।॥"
_ALLOWED_CORE = re.compile(rf"[{_LETTERS}0-9̀-ͯ{re.escape(_PUNCT_EDGE)}]*")
_HAS_DIGIT = re.compile(r"[0-9]")
_HAS_LETTER = re.compile(rf"[{_LETTERS}]")
# Citation tokens such as "vv.15-18", "p.47", "Ch.1", "1.9": a short letter prefix
# followed by digits. OCR errors put letters after digits, which this does not match.
_REFERENCE = re.compile(r"[A-Za-z]{1,4}\.?[0-9][0-9.\-–,]*")


@dataclass
class WordConfidence:
    """Confidence score for an individual token."""
    word: str
    confidence: float
    position: int


def _token_issue(token: str):
    """Return an issue label for a whitespace token, or None if it looks clean."""
    core = token.strip(_PUNCT_EDGE)
    if not core:
        return None
    if _REFERENCE.fullmatch(core):
        return None
    if _HAS_DIGIT.search(core) and _HAS_LETTER.search(core):
        return "mixed_digit_letter"       # e.g. "5ā", "l0ka": typical OCR substitution
    if not _ALLOWED_CORE.fullmatch(core):
        return "unexpected_character"     # symbols or scripts outside the expected set
    return None


class TextIngestionPipeline:
    """Pipeline for ingesting classical mathematical astronomy texts into the bronze layer."""

    # Set to True only once a real IAST converter (e.g. Aksharamukha) is wired in.
    TRANSLITERATION_IMPLEMENTED = False

    def __init__(self):
        pass

    def ocr_quality_check(self, text: str, min_confidence: float = 0.85) -> Tuple[bool, Dict]:
        """
        Token-level quality gate for OCR or transcribed text.

        A token is flagged if it mixes digits and letters (a common OCR substitution)
        or contains a character outside the expected IAST/Devanagari set. Pure
        numerals (verse and page numbers) and ordinary IAST letters are not flagged.
        ``estimated_confidence`` is the share of scored tokens that are clean. This is a
        screening heuristic, not a calibrated OCR confidence.

        Args:
            text: Extracted text
            min_confidence: Minimum share of clean tokens to pass the gate

        Returns:
            Tuple of (is_valid, metrics)
        """
        text = unicodedata.normalize("NFC", text)
        tokens = text.split()

        issues: Counter = Counter()
        flagged_examples: List[str] = []
        scored = 0
        for tok in tokens:
            if not tok.strip(_PUNCT_EDGE):
                continue
            scored += 1
            issue = _token_issue(tok)
            if issue:
                issues[issue] += 1
                if len(flagged_examples) < 10:
                    flagged_examples.append(tok)

        clean_share = (scored - sum(issues.values())) / scored if scored else 0.0
        metrics = {
            'length': len(text),
            'word_count': len(tokens),
            'scored_tokens': scored,
            'has_special_chars': bool(re.search(r'[^\w\s]', text)),
            'detected_issues': [f"{k}: {v}" for k, v in sorted(issues.items())],
            'flagged_examples': flagged_examples,
            'estimated_confidence': round(clean_share, 4),
        }

        is_valid = scored > 0 and clean_share >= min_confidence
        return is_valid, metrics

    def transliterate_iast(self, text: str) -> str:
        """
        Transliterate Sanskrit text to IAST.

        Placeholder: returns the input unchanged. Replace with Aksharamukha and set
        ``TRANSLITERATION_IMPLEMENTED = True``.
        """
        return text

    def normalize_unicode(self, text: str) -> str:
        """NFC-normalize text so diacritics have one canonical encoding."""
        return unicodedata.normalize('NFC', text)

    def extract_word_confidence(self, text: str) -> List[WordConfidence]:
        """
        Heuristic per-token score: 0.95 baseline, minus 0.1 for very short or very long
        tokens and 0.2 for tokens containing non-letters (other than the IAST letters).
        Used for review prioritisation, not for the pass/fail gate.
        """
        confidences = []
        for position, word in enumerate(text.split()):
            confidence = 0.95
            if len(word) < 2 or len(word) > 20:
                confidence -= 0.1
            if _token_issue(word) is not None:
                confidence -= 0.2
            confidences.append(WordConfidence(
                word=word,
                confidence=max(0.0, min(1.0, confidence)),
                position=position,
            ))
        return confidences

    def ingest(self, text: str, source_reference: Dict) -> Dict:
        """
        Run the ingestion checks on raw OCR or transcribed text.

        Args:
            text: Raw extracted text
            source_reference: Source metadata (treatise, chapter, verse, ...)

        Returns:
            Dict with the normalized text, IAST status, quality metrics and
            ``pipeline_status`` ('ready_for_bronze' or 'needs_review')
        """
        is_valid_ocr, ocr_metrics = self.ocr_quality_check(text)
        normalized_text = self.normalize_unicode(text)

        if self.TRANSLITERATION_IMPLEMENTED:
            iast_text = self.transliterate_iast(normalized_text)
            transliteration_status = 'implemented'
        else:
            iast_text = None
            transliteration_status = 'not_implemented'

        word_confidences = self.extract_word_confidence(normalized_text)

        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'iast_text': iast_text,
            'transliteration_status': transliteration_status,
            'source_reference': source_reference,
            'ocr_metrics': ocr_metrics,
            'word_confidences': [
                {'word': wc.word, 'confidence': wc.confidence, 'position': wc.position}
                for wc in word_confidences
            ],
            'is_valid_ocr': is_valid_ocr,
            'pipeline_status': 'ready_for_bronze' if is_valid_ocr else 'needs_review',
        }
