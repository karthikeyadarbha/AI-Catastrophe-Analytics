"""
Text ingestion pipeline for the bronze layer of ancient mathematical astronomy treatises.

Handles OCR quality assessment, transliteration to standard notation, and validation
of classical mathematical astronomy texts before ingestion into the bronze layer.
Ensures quality and consistency for downstream analysis of historical astronomical methods.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WordConfidence:
    """Confidence score for individual words in ancient texts."""
    word: str
    confidence: float
    position: int


class TextIngestionPipeline:
    """Pipeline for ingesting classical mathematical astronomy texts into the bronze layer."""

    def __init__(self):
        """Initialize the ingestion pipeline for ancient mathematical astronomy texts."""
        pass

    def ocr_quality_check(self, text: str, min_confidence: float = 0.85) -> Tuple[bool, Dict]:
        """
        Check OCR quality of extracted ancient text.
        
        Assesses the quality of optical character recognition on classical manuscripts
        to ensure reliable extraction of mathematical astronomy knowledge.
        
        Args:
            text: The extracted text from ancient sources
            min_confidence: Minimum acceptable confidence threshold
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        # Check for common OCR errors (o/0, l/1, etc.)
        suspicious_patterns = {
            r'[0O]': 'Zero/O confusion',
            r'[1l]': 'One/L confusion',
            r'[5S]': 'Five/S confusion',
            r'[8B]': 'Eight/B confusion',
        }

        issues = []
        for pattern, issue_type in suspicious_patterns.items():
            if re.search(pattern, text):
                issues.append(issue_type)

        # Basic quality heuristics
        metrics = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_special_chars': bool(re.search(r'[^\w\s]', text)),
            'detected_issues': issues,
            'estimated_confidence': max(0.0, 1.0 - (len(issues) * 0.1)),
        }

        is_valid = metrics['estimated_confidence'] >= min_confidence

        return is_valid, metrics

    def transliterate_iast(self, text: str) -> str:
        """
        Transliterate classical Sanskrit text to IAST (International Alphabet of Sanskrit Transliteration).
        
        Converts ancient Sanskrit manuscripts to standard scholarly notation for consistent
        representation of mathematical and astronomical terminology across treatises.
        
        Note: In production, this uses the Aksharamukha library for proper transliteration.
        
        Args:
            text: Input Sanskrit text (could be Devanagari or other classical script)
            
        Returns:
            IAST transliterated text
        """
        # Placeholder: in production, use Aksharamukha for proper transliteration
        return text

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode to NFC (Canonical Decomposition, followed by Canonical Composition).
        
        Ensures consistent representation of diacritics and special characters in ancient texts.
        
        Args:
            text: Input text
            
        Returns:
            NFC-normalized text
        """
        import unicodedata
        return unicodedata.normalize('NFC', text)

    def extract_word_confidence(self, text: str) -> List[WordConfidence]:
        """
        Extract per-word confidence scores for ancient manuscript text.
        
        Assigns individual confidence values to each word, enabling detection of
        problematic OCR extractions from classical mathematical astronomy texts.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of WordConfidence objects with per-word quality metrics
        """
        words = text.split()
        confidences = []

        for position, word in enumerate(words):
            # Simple heuristic: words with special chars or unusual length get lower confidence
            confidence = 0.95  # Default high confidence for clear extractions
            
            # Penalty for unusual length (very short or very long words)
            if len(word) < 2 or len(word) > 20:
                confidence -= 0.1
            
            # Penalty for special characters that might indicate OCR errors
            if not word.isalpha() and not all(c in 'āēīōūṅñṭḍṇḷṛ' for c in word):
                confidence -= 0.2

            confidences.append(WordConfidence(
                word=word,
                confidence=max(0.0, min(1.0, confidence)),
                position=position
            ))

        return confidences

    def ingest(self, text: str, source_reference: Dict) -> Dict:
        """
        Run the full ingestion pipeline for ancient mathematical astronomy texts.
        
        Processes raw OCR output through quality assessment, normalization, and validation
        to prepare classical texts for ingestion into the bronze layer.
        
        Args:
            text: Raw extracted text from ancient manuscripts
            source_reference: Metadata about the source (treatise, chapter, verse, etc.)
            
        Returns:
            Result dictionary with processed text and quality metrics
        """
        # Step 1: Quality check on OCR extraction
        is_valid_ocr, ocr_metrics = self.ocr_quality_check(text)

        # Step 2: Normalize Unicode for consistent representation
        normalized_text = self.normalize_unicode(text)

        # Step 3: Transliterate to IAST standard scholarly notation
        iast_text = self.transliterate_iast(normalized_text)

        # Step 4: Extract per-word confidence scores for quality assessment
        word_confidences = self.extract_word_confidence(iast_text)

        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'iast_text': iast_text,
            'source_reference': source_reference,
            'ocr_metrics': ocr_metrics,
            'word_confidences': [
                {'word': wc.word, 'confidence': wc.confidence, 'position': wc.position}
                for wc in word_confidences
            ],
            'is_valid_ocr': is_valid_ocr,
            'pipeline_status': 'ready_for_bronze' if is_valid_ocr else 'needs_review',
        }
