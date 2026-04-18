"""
Tests for nlp.ner_model module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.ner_model import PropertyNERModel, EntityType, extract_entities


class TestPropertyNERModel:
    """Tests for PropertyNERModel class."""
    
    @pytest.fixture
    def ner_model(self):
        """Create a NER model instance."""
        return PropertyNERModel(use_transformers=False)
    
    def test_extract_survey_number(self, ner_model):
        """Test extraction of survey numbers."""
        text = "Property at Survey No. 123/4 in Bangalore"
        result = ner_model.extract(text)
        
        property_ids = [e for e in result.entities if e.entity_type == EntityType.PROPERTY_ID]
        assert len(property_ids) >= 1
        assert "123/4" in property_ids[0].text or "123/4" in (property_ids[0].normalized or "")
    
    def test_extract_area(self, ner_model):
        """Test extraction of area measurements."""
        text = "Total area is 2400 Sq.Ft. or 223 Sq.M."
        result = ner_model.extract(text)
        
        areas = [e for e in result.entities if e.entity_type == EntityType.AREA]
        assert len(areas) >= 1
    
    def test_extract_money(self, ner_model):
        """Test extraction of monetary values."""
        text = "Sale consideration is Rs. 1,50,00,000/- (One Crore Fifty Lakhs)"
        result = ner_model.extract(text)
        
        money = [e for e in result.entities if e.entity_type == EntityType.MONEY]
        assert len(money) >= 1
    
    def test_extract_dates(self, ner_model):
        """Test extraction of dates."""
        text = "Executed on 15th January, 2024 at Bangalore"
        result = ner_model.extract(text)
        
        dates = [e for e in result.entities if e.entity_type == EntityType.DATE]
        assert len(dates) >= 1
        assert "January" in dates[0].text or "15" in dates[0].text
    
    def test_extract_document_type(self, ner_model):
        """Test extraction of document type."""
        text = "This Sale Deed is executed between the parties"
        result = ner_model.extract(text)
        
        doc_types = [e for e in result.entities if e.entity_type == EntityType.DOCUMENT_TYPE]
        assert len(doc_types) >= 1
        assert "Sale" in doc_types[0].text
    
    def test_extract_multiple_property_ids(self, ner_model):
        """Test extraction of multiple property identifiers."""
        text = """
        Property Details:
        Survey No. 123/4
        Plot No. 567
        Khata No. 89
        """
        result = ner_model.extract(text)
        
        property_ids = [e for e in result.entities if e.entity_type == EntityType.PROPERTY_ID]
        assert len(property_ids) >= 3
    
    def test_entity_counts(self, ner_model):
        """Test entity counting."""
        text = """
        Survey No. 123/4
        Area: 2400 Sq.Ft.
        Value: Rs. 50,00,000/-
        Date: 15-01-2024
        """
        result = ner_model.extract(text)
        
        assert "PROPERTY_ID" in result.entity_counts
        assert "AREA" in result.entity_counts
        assert "MONEY" in result.entity_counts
    
    def test_entity_deduplication(self, ner_model):
        """Test that overlapping entities are deduplicated."""
        text = "Survey No. 123/4"
        result = ner_model.extract(text)
        
        # Should not have duplicate entities for the same span
        property_ids = [e for e in result.entities if e.entity_type == EntityType.PROPERTY_ID]
        texts = [e.text for e in property_ids]
        
        # No exact duplicates
        assert len(texts) == len(set(texts))


class TestExtractEntitiesFunction:
    """Tests for the extract_entities convenience function."""
    
    def test_convenience_function(self):
        """Test the extract_entities function."""
        text = """
        SALE DEED
        Survey No. 123/4
        Area: 2400 Sq.Ft.
        Value: Rs. 1,00,00,000/-
        Date: 15th January, 2024
        """
        result = extract_entities(text)
        
        assert "entities" in result
        assert "counts" in result
        assert isinstance(result["entities"], dict)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = extract_entities("")
        
        assert "entities" in result
        assert len(result["entities"]) == 0


class TestIndianPropertyPatterns:
    """Tests specific to Indian property document patterns."""
    
    @pytest.fixture
    def ner_model(self):
        return PropertyNERModel(use_transformers=False)
    
    def test_khasra_khatoni(self, ner_model):
        """Test extraction of Khasra/Khatoni numbers."""
        text = "Khasra No. 456/2 and Khatoni No. 789"
        result = ner_model.extract(text)
        
        property_ids = [e for e in result.entities if e.entity_type == EntityType.PROPERTY_ID]
        assert len(property_ids) >= 2
    
    def test_lakhs_crores(self, ner_model):
        """Test extraction of Indian number formats."""
        text = "Value is 50 Lakhs or 1.5 Crores"
        result = ner_model.extract(text)
        
        money = [e for e in result.entities if e.entity_type == EntityType.MONEY]
        assert len(money) >= 2
    
    def test_area_units(self, ner_model):
        """Test various Indian area units."""
        text = "Land measuring 5 Acres 2 Guntas or 100 Cents"
        result = ner_model.extract(text)
        
        areas = [e for e in result.entities if e.entity_type == EntityType.AREA]
        assert len(areas) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
