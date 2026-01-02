"""
DeedLens Named Entity Recognition (NER) Model
Extracts property-related entities using pre-trained transformers.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EntityType(Enum):
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    PROPERTY_ID = "PROPERTY_ID"
    AREA = "AREA"
    MONEY = "MONEY"
    DATE = "DATE"
    ORGANIZATION = "ORGANIZATION"
    DOCUMENT_TYPE = "DOCUMENT_TYPE"


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float
    normalized: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class NERResult:
    """Result from NER processing."""
    text: str
    entities: List[Entity]
    entity_counts: Dict[str, int] = field(default_factory=dict)


class PropertyNERModel:
    """
    Named Entity Recognition for property documents.
    Combines spaCy NER with custom regex patterns for property-specific entities.
    """
    
    # Regex patterns for property-specific entities
    PATTERNS = {
        EntityType.PROPERTY_ID: [
            r'(?:Survey\s*No\.?|S\.?\s*No\.?)\s*[:.]?\s*([\d/\-A-Za-z]+)',
            r'(?:Plot\s*No\.?)\s*[:.]?\s*([\d/\-A-Za-z]+)',
            r'(?:Khata\s*No\.?)\s*[:.]?\s*([\d/\-A-Za-z]+)',
            r'(?:Property\s*ID)\s*[:.]?\s*([\w\-/]+)',
            r'(?:PID)\s*[:.]?\s*([\w\-/]+)',
            r'(?:Khasra\s*No\.?)\s*[:.]?\s*([\d/\-A-Za-z]+)',
            r'(?:Khatoni\s*No\.?)\s*[:.]?\s*([\d/\-A-Za-z]+)',
            r'(?:Document\s*No\.?)\s*[:.]?\s*([\d/\-A-Za-z]+)',
        ],
        EntityType.AREA: [
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:Sq\.?\s*Ft\.?|Square\s*Feet|sq\.?\s*ft\.?)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:Sq\.?\s*M\.?|Square\s*Meters?|sq\.?\s*m\.?)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:Sq\.?\s*Yards?|Square\s*Yards?)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:Acres?|acres?)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:Guntas?|guntas?)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:Cents?|cents?)',
        ],
        EntityType.MONEY: [
            r'Rs\.?\s*([\d,]+(?:\.\d{2})?)\s*(?:/-)?',
            r'INR\.?\s*([\d,]+(?:\.\d{2})?)',
            r'₹\s*([\d,]+(?:\.\d{2})?)',
            r'([\d,]+)\s*(?:Lakhs?|lakhs?)',
            r'([\d,]+)\s*(?:Crores?|crores?)',
            r'(?:Rupees?)\s+([\w\s]+(?:Lakhs?|Crores?|Thousand)\s*(?:only)?)',
        ],
        EntityType.DATE: [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)[,]?\s+\d{4})',
        ],
        EntityType.DOCUMENT_TYPE: [
            r'(Sale\s*Deed)',
            r'(General\s*Power\s*of\s*Attorney)',
            r'(Special\s*Power\s*of\s*Attorney)',
            r'(Agreement\s*to\s*Sell)',
            r'(Gift\s*Deed)',
            r'(Will\s*Deed)',
        ],
    }
    
    def __init__(self, use_transformers: bool = False, spacy_model: str = "en_core_web_sm"):
        self.use_transformers = use_transformers
        self.spacy_model_name = spacy_model
        self._nlp = None
        self._transformer_ner = None
        
    def _load_spacy(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy not available. Install spacy and download en_core_web_sm")
            try:
                self._nlp = spacy.load(self.spacy_model_name)
            except OSError:
                # Model not found, try to download
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.spacy_model_name])
                self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp
    
    def _load_transformer(self):
        """Lazy load transformer-based NER."""
        if self._transformer_ner is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install transformers")
            self._transformer_ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
        return self._transformer_ner
    
    def _extract_with_regex(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Get the full match or first group
                    matched_text = match.group(0)
                    captured = match.group(1) if match.lastindex else matched_text
                    
                    entities.append(Entity(
                        text=matched_text,
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,  # Regex matches are high confidence
                        normalized=captured.strip(),
                        metadata={"source": "regex"}
                    ))
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        nlp = self._load_spacy()
        doc = nlp(text)
        
        entities = []
        
        # Map spaCy labels to our entity types
        label_map = {
            "PERSON": EntityType.PERSON,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
            "DATE": EntityType.DATE,
            "MONEY": EntityType.MONEY,
        }
        
        for ent in doc.ents:
            if ent.label_ in label_map:
                entities.append(Entity(
                    text=ent.text,
                    entity_type=label_map[ent.label_],
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,
                    metadata={"source": "spacy", "label": ent.label_}
                ))
        
        return entities
    
    def _extract_with_transformer(self, text: str) -> List[Entity]:
        """Extract entities using transformer-based NER."""
        ner = self._load_transformer()
        results = ner(text)
        
        entities = []
        
        # Map transformer labels to our entity types
        label_map = {
            "PER": EntityType.PERSON,
            "LOC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
        }
        
        for r in results:
            label = r["entity_group"]
            if label in label_map:
                entities.append(Entity(
                    text=r["word"],
                    entity_type=label_map[label],
                    start=r["start"],
                    end=r["end"],
                    confidence=r["score"],
                    metadata={"source": "transformer", "label": label}
                ))
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        deduped = []
        for entity in entities:
            # Check if this overlaps with any existing entity
            overlaps = False
            for existing in deduped:
                if (entity.start < existing.end and entity.end > existing.start):
                    overlaps = True
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduped.remove(existing)
                        deduped.append(entity)
                    break
            if not overlaps:
                deduped.append(entity)
        
        return sorted(deduped, key=lambda e: e.start)
    
    def extract(self, text: str) -> NERResult:
        """
        Extract all entities from text.
        
        Args:
            text: Document text
        
        Returns:
            NERResult with all extracted entities
        """
        all_entities = []
        
        # Extract with regex (property-specific patterns)
        all_entities.extend(self._extract_with_regex(text))
        
        # Extract with spaCy or transformer
        if self.use_transformers and TRANSFORMERS_AVAILABLE:
            all_entities.extend(self._extract_with_transformer(text))
        elif SPACY_AVAILABLE:
            all_entities.extend(self._extract_with_spacy(text))
        
        # Deduplicate
        entities = self._deduplicate_entities(all_entities)
        
        # Count entities by type
        entity_counts = {}
        for entity in entities:
            type_name = entity.entity_type.value
            entity_counts[type_name] = entity_counts.get(type_name, 0) + 1
        
        return NERResult(
            text=text,
            entities=entities,
            entity_counts=entity_counts
        )


def extract_entities(text: str, use_transformers: bool = False) -> Dict:
    """
    Convenience function to extract entities from text.
    
    Args:
        text: Document text
        use_transformers: Whether to use transformer-based NER
    
    Returns:
        Dictionary with entities grouped by type
    """
    model = PropertyNERModel(use_transformers=use_transformers)
    result = model.extract(text)
    
    # Group entities by type
    grouped = {}
    for entity in result.entities:
        type_name = entity.entity_type.value
        if type_name not in grouped:
            grouped[type_name] = []
        grouped[type_name].append({
            "text": entity.text,
            "normalized": entity.normalized,
            "confidence": entity.confidence,
            "start": entity.start,
            "end": entity.end
        })
    
    return {
        "entities": grouped,
        "counts": result.entity_counts
    }


if __name__ == "__main__":
    sample_text = """
    SALE DEED
    
    This Sale Deed is executed on 15th January, 2024 at Bangalore.
    
    Mr. Ramesh Kumar, residing at Indiranagar, Bangalore
    sells property Survey No. 123/4 to Mrs. Priya Sharma.
    
    Property Details:
    - Total Area: 2400 Sq.Ft.
    - Sale Value: Rs. 1,50,00,000/- (One Crore Fifty Lakhs only)
    - Location: Koramangala, Bangalore
    """
    
    result = extract_entities(sample_text)
    
    print("=== Extracted Entities ===")
    for entity_type, entities in result["entities"].items():
        print(f"\n{entity_type}:")
        for e in entities:
            print(f"  - {e['text']} (conf: {e['confidence']:.2f})")
    
    print(f"\n=== Counts ===")
    print(result["counts"])
