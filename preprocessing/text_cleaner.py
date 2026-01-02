"""
DeedLens Text Cleaner
Cleans and normalizes raw OCR text from property documents.
"""

import re
import unicodedata
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CleanedText:
    """Result of text cleaning."""
    original: str
    cleaned: str
    paragraphs: List[str]
    word_count: int
    char_count: int


class TextCleaner:
    """
    Cleans and normalizes OCR-extracted text from property documents.
    Handles common OCR errors and legal document formatting.
    """
    
    # Common OCR errors and corrections
    OCR_CORRECTIONS = {
        r'\b0\b(?=\s*(?:Rs|INR|₹))': 'O',  # O misread as 0 before currency
        r'(?<=[a-z])1(?=[a-z])': 'l',  # 1 misread as l in words
        r'(?<=[A-Z])0(?=[A-Z])': 'O',  # 0 misread as O in uppercase
        r'\bl\b': 'I',  # Standalone l should be I
        r'rn': 'm',  # rn often misread for m
        r'vv': 'w',  # vv misread as w
    }
    
    # Patterns to remove
    NOISE_PATTERNS = [
        r'Page\s*\d+\s*of\s*\d+',  # Page numbers
        r'^\s*\d+\s*$',  # Standalone numbers (page numbers)
        r'^[-_=]{3,}$',  # Decorative lines
        r'^\s*(?:CONFIDENTIAL|DRAFT|COPY)\s*$',  # Document markers
        r'www\.[^\s]+',  # URLs
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Emails (keep or remove?)
    ]
    
    # Legal document section headers
    SECTION_MARKERS = [
        r'(?:SCHEDULE|ANNEXURE|EXHIBIT)\s*[A-Z0-9-]+',
        r'(?:WITNESSETH|WHEREAS|NOW THEREFORE)',
        r'(?:IN WITNESS WHEREOF)',
        r'(?:FIRST PARTY|SECOND PARTY|VENDOR|VENDEE|PURCHASER)',
    ]
    
    def __init__(
        self,
        fix_ocr_errors: bool = True,
        remove_noise: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True
    ):
        self.fix_ocr_errors = fix_ocr_errors
        self.remove_noise = remove_noise
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFKC form
        text = unicodedata.normalize('NFKC', text)
        
        # Replace common unicode variants
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u20b9': 'Rs.',  # Rupee symbol
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR recognition errors."""
        for pattern, replacement in self.OCR_CORRECTIONS.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _remove_noise(self, text: str) -> str:
        """Remove noise patterns like headers, footers, page numbers."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            skip = False
            for pattern in self.NOISE_PATTERNS:
                if re.match(pattern, line.strip(), re.IGNORECASE | re.MULTILINE):
                    skip = True
                    break
            if not skip:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace from entire text
        text = text.strip()
        
        return text
    
    def _normalize_legal_terms(self, text: str) -> str:
        """Standardize common legal terminology."""
        # Standardize currency formats
        text = re.sub(r'Rs\.?\s*', 'Rs. ', text)
        text = re.sub(r'INR\.?\s*', 'Rs. ', text)
        text = re.sub(r'₹\s*', 'Rs. ', text)
        
        # Standardize number formats (lakhs, crores)
        text = re.sub(r'(\d+)\s*[Ll]akhs?', r'\1 Lakhs', text)
        text = re.sub(r'(\d+)\s*[Cc]rores?', r'\1 Crores', text)
        
        # Standardize area units
        text = re.sub(r'[Ss]q\.?\s*[Ff]t\.?', 'Sq.Ft.', text)
        text = re.sub(r'[Ss]q\.?\s*[Mm]\.?', 'Sq.M.', text)
        text = re.sub(r'[Ss]q\.?\s*[Yy]ards?', 'Sq.Yards', text)
        
        return text
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from cleaned text."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def clean(self, text: str) -> CleanedText:
        """
        Clean and normalize OCR text.
        
        Args:
            text: Raw OCR text
        
        Returns:
            CleanedText with original, cleaned text, and paragraphs
        """
        original = text
        
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        if self.fix_ocr_errors:
            text = self._fix_ocr_errors(text)
        
        if self.remove_noise:
            text = self._remove_noise(text)
        
        text = self._normalize_legal_terms(text)
        
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        paragraphs = self._extract_paragraphs(text)
        
        return CleanedText(
            original=original,
            cleaned=text,
            paragraphs=paragraphs,
            word_count=len(text.split()),
            char_count=len(text)
        )
    
    def clean_batch(self, texts: List[str]) -> List[CleanedText]:
        """Clean multiple texts."""
        return [self.clean(text) for text in texts]


def clean_ocr_text(text: str) -> Dict:
    """
    Convenience function to clean OCR text.
    
    Args:
        text: Raw OCR text
    
    Returns:
        Dictionary with cleaned text and metadata
    """
    cleaner = TextCleaner()
    result = cleaner.clean(text)
    
    return {
        "cleaned_text": result.cleaned,
        "paragraphs": result.paragraphs,
        "word_count": result.word_count,
        "char_count": result.char_count
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Page 1 of 5
    
    SALE DEED
    
    This Sale Deed is executed on this 15th day of January, 2024
    at Bangalore, Karnataka.
    
    BETWEEN
    
    Mr. Ramesh Kumar S/o Late Shri Krishna Kumar,
    aged about 55 years, residing at No. 123, MG Road,
    Indiranagar, Bangalore - 560038
    (hereinafter called the "VENDOR")
    
    AND
    
    Mrs. Priya Sharma W/o Shri Anil Sharma,
    aged about 35 years, residing at No. 456, 100 Feet Road,
    Koramangala, Bangalore - 560034
    (hereinafter called the "PURCHASER")
    
    Property Details:
    - Survey No: 123/4
    - Total Area: 2400 sq.ft.
    - Sale Consideration: Rs 1,50,00,000/- (Rupees One Crore Fifty Lakhs only)
    
    Page 2 of 5
    """
    
    cleaner = TextCleaner()
    result = cleaner.clean(sample_text)
    
    print("=== Cleaned Text ===")
    print(result.cleaned)
    print(f"\nWord count: {result.word_count}")
    print(f"Paragraphs: {len(result.paragraphs)}")
