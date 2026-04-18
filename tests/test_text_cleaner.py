"""
Tests for preprocessing.text_cleaner module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.text_cleaner import TextCleaner, clean_ocr_text


class TestTextCleaner:
    """Tests for TextCleaner class."""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        cleaner = TextCleaner()
        text = "  Hello   World  \n\n  This is a test  "
        result = cleaner.clean(text)
        
        assert "Hello" in result.cleaned
        assert "World" in result.cleaned
        assert result.word_count > 0
    
    def test_unicode_normalization(self):
        """Test unicode character normalization."""
        cleaner = TextCleaner()

        # Test smart quotes
        text = "\"Hello World\""
        result = cleaner.clean(text)
        assert '"Hello World"' in result.cleaned
        
        # Test rupee symbol
        text = "₹50,000"
        result = cleaner.clean(text)
        assert "Rs." in result.cleaned
    
    def test_noise_removal(self):
        """Test removal of page numbers and noise."""
        cleaner = TextCleaner()
        text = """
        Page 1 of 5
        
        SALE DEED
        
        This is the content.
        
        Page 2 of 5
        """
        result = cleaner.clean(text)
        
        assert "SALE DEED" in result.cleaned
        assert "This is the content" in result.cleaned
        assert "Page 1 of 5" not in result.cleaned
    
    def test_legal_term_normalization(self):
        """Test normalization of legal terms."""
        cleaner = TextCleaner()
        
        # Test currency normalization
        text = "Rs.50000 or INR 50000 or ₹50000"
        result = cleaner.clean(text)
        assert result.cleaned.count("Rs.") >= 1
        
        # Test area normalization
        text = "Area is 2400 sq.ft. or 2400 Sq Ft"
        result = cleaner.clean(text)
        assert "Sq.Ft." in result.cleaned
    
    def test_paragraph_extraction(self):
        """Test paragraph extraction."""
        cleaner = TextCleaner()
        text = """
        First paragraph here.
        
        Second paragraph here.
        
        Third paragraph here.
        """
        result = cleaner.clean(text)
        
        assert len(result.paragraphs) == 3
    
    def test_convenience_function(self):
        """Test the clean_ocr_text convenience function."""
        text = "Sample text for testing"
        result = clean_ocr_text(text)
        
        assert "cleaned_text" in result
        assert "paragraphs" in result
        assert "word_count" in result
        assert result["word_count"] == 4


class TestTextCleanerEdgeCases:
    """Edge case tests for TextCleaner."""
    
    def test_empty_text(self):
        """Test handling of empty text."""
        cleaner = TextCleaner()
        result = cleaner.clean("")
        
        assert result.cleaned == ""
        assert result.word_count == 0
        assert len(result.paragraphs) == 0
    
    def test_only_whitespace(self):
        """Test handling of whitespace-only text."""
        cleaner = TextCleaner()
        result = cleaner.clean("   \n\n\t\t   ")
        
        assert result.cleaned == ""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        cleaner = TextCleaner()
        text = "word " * 10000
        result = cleaner.clean(text)
        
        assert result.word_count == 10000


class TestIndianOCRCorrections:
    """Tests for Indian-specific OCR corrections."""
    
    def test_state_name_corrections(self):
        """Test correction of misspelled Indian state names."""
        cleaner = TextCleaner()
        
        # Karnataka variations
        assert "Karnataka" in cleaner.clean("Kamataka").cleaned
        assert "Karnataka" in cleaner.clean("Karn ataka").cleaned
        
        # Other states
        assert "Tamil Nadu" in cleaner.clean("Tamii Nadu").cleaned
        assert "Maharashtra" in cleaner.clean("Maharash tra").cleaned
    
    def test_city_name_corrections(self):
        """Test correction of misspelled Indian city names."""
        cleaner = TextCleaner()
        
        assert "Bangalore" in cleaner.clean("Bangal ore").cleaned
        assert "Chennai" in cleaner.clean("Chenna1").cleaned
        assert "Mumbai" in cleaner.clean("Mumba i").cleaned
        assert "Indiranagar" in cleaner.clean("lndiranagar").cleaned
        assert "Koramangala" in cleaner.clean("Koramanga1a").cleaned
    
    def test_number_letter_confusion(self):
        """Test correction of number/letter OCR confusion."""
        cleaner = TextCleaner()
        
        assert "10th" in cleaner.clean("1Oth March").cleaned
        assert "20th" in cleaner.clean("2Oth January").cleaned
    
    def test_currency_corrections(self):
        """Test correction of currency formatting."""
        cleaner = TextCleaner()
        
        # Various rs, formats
        result = cleaner.clean("rs, 50000")
        assert "Rs." in result.cleaned
        
        result = cleaner.clean("Rs, 1,00,000")
        assert "Rs." in result.cleaned
    
    def test_real_world_ocr_text(self):
        """Test cleaning of realistic OCR output."""
        cleaner = TextCleaner()
        
        ocr_text = """
        SALE DEED executed at Bangal ore, Kamataka on 1Oth March 2024.
        Property located at Koramanga1a, valued at rs, 1,50,00,000.
        """
        
        result = cleaner.clean(ocr_text)
        
        assert "Bangalore" in result.cleaned
        assert "Karnataka" in result.cleaned
        assert "Koramangala" in result.cleaned
        assert "10th" in result.cleaned
        assert "Rs." in result.cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
