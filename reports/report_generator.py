"""
DeedLens Report Generator
Generates property reports using LLM.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from string import Template

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class PropertyReport:
    """Generated property report."""
    title: str
    summary: str
    sections: Dict[str, str]
    raw_response: str


class ReportGenerator:
    """
    Generates property reports using Gemini LLM.
    """
    
    REPORT_PROMPT = """You are a property document analyst. Generate a structured property report based on the following extracted information.

**Document Text:**
{document_text}

**Extracted Entities:**
{entities}

**Generate a report with the following sections:**

1. **Property Summary** - Brief overview of the property
2. **Parties Involved** - Who are the buyers/sellers
3. **Property Details** - Location, area, identifiers
4. **Transaction Details** - Value, date, terms
5. **Key Observations** - Any notable points or concerns

Keep the report professional, concise, and factual. Use bullet points where appropriate.
"""

    FALLBACK_TEMPLATE = Template("""
# Property Report

## Property Summary
This is a ${doc_type} document involving property transaction.

## Parties Involved
${parties}

## Property Details
- **Location:** ${location}
- **Area:** ${area}
- **Property ID:** ${property_id}

## Transaction Details
- **Transaction Value:** ${value}
- **Transaction Date:** ${date}

## Key Observations
- Document has been processed and entities extracted
- Further manual verification recommended
""")
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._model = None
    
    def _load_model(self):
        """Initialize Gemini model."""
        if self._model is None and GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel('gemini-pro')
        return self._model
    
    def _format_entities(self, entities: Dict) -> str:
        """Format entities for the prompt."""
        lines = []
        for entity_type, entity_list in entities.items():
            if entity_list:
                lines.append(f"**{entity_type}:**")
                for e in entity_list[:5]:  # Limit to 5 per type
                    if isinstance(e, dict):
                        lines.append(f"  - {e.get('text', e.get('canonical', str(e)))}")
                    else:
                        lines.append(f"  - {e}")
        return "\n".join(lines)
    
    def generate(
        self,
        document_text: str,
        entities: Dict,
        use_llm: bool = True
    ) -> PropertyReport:
        """
        Generate a property report.
        
        Args:
            document_text: Full document text
            entities: Extracted entities dictionary
            use_llm: Whether to use LLM (requires API key)
        
        Returns:
            PropertyReport object
        """
        if use_llm and GEMINI_AVAILABLE and self.api_key:
            return self._generate_with_llm(document_text, entities)
        else:
            return self._generate_fallback(document_text, entities)
    
    def _generate_with_llm(
        self,
        document_text: str,
        entities: Dict
    ) -> PropertyReport:
        """Generate report using Gemini LLM."""
        model = self._load_model()
        
        prompt = self.REPORT_PROMPT.format(
            document_text=document_text[:3000],  # Limit text length
            entities=self._format_entities(entities)
        )
        
        try:
            response = model.generate_content(prompt)
            raw_response = response.text
            
            # Parse sections from response
            sections = self._parse_sections(raw_response)
            
            return PropertyReport(
                title="Property Transaction Report",
                summary=sections.get("Property Summary", "Report generated successfully."),
                sections=sections,
                raw_response=raw_response
            )
        except Exception as e:
            # Fall back if LLM fails
            return self._generate_fallback(document_text, entities)
    
    def _generate_fallback(
        self,
        document_text: str,
        entities: Dict
    ) -> PropertyReport:
        """Generate report using template."""
        # Extract info from entities
        persons = entities.get("PERSON", [])
        locations = entities.get("LOCATION", [])
        areas = entities.get("AREA", [])
        money = entities.get("MONEY", [])
        dates = entities.get("DATE", [])
        prop_ids = entities.get("PROPERTY_ID", [])
        
        def get_text(items, default="Not specified"):
            if not items:
                return default
            first = items[0]
            if isinstance(first, dict):
                return first.get("text", first.get("canonical", default))
            return str(first)
        
        # Format parties
        parties_list = []
        for p in persons[:4]:
            if isinstance(p, dict):
                parties_list.append(f"- {p.get('text', p.get('canonical', ''))}")
            else:
                parties_list.append(f"- {p}")
        parties = "\n".join(parties_list) if parties_list else "- Not identified"
        
        report_text = self.FALLBACK_TEMPLATE.substitute(
            doc_type="property transaction",
            parties=parties,
            location=get_text(locations),
            area=get_text(areas),
            property_id=get_text(prop_ids),
            value=get_text(money),
            date=get_text(dates)
        )
        
        return PropertyReport(
            title="Property Transaction Report",
            summary="Report generated from extracted entities.",
            sections={
                "Property Summary": "Property transaction document processed.",
                "Parties Involved": parties,
                "Property Details": f"Location: {get_text(locations)}, Area: {get_text(areas)}",
                "Transaction Details": f"Value: {get_text(money)}, Date: {get_text(dates)}",
            },
            raw_response=report_text
        )
    
    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse sections from LLM response."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            # Check for section headers
            if line.strip().startswith('**') and line.strip().endswith('**'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = line.strip().strip('*').strip()
                current_content = []
            elif line.strip().startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = line.strip()[3:].strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections


def generate_report(
    document_text: str,
    entities: Dict,
    api_key: Optional[str] = None
) -> Dict:
    """
    Convenience function to generate a property report.
    
    Args:
        document_text: Full document text
        entities: Extracted entities
        api_key: Optional Gemini API key
    
    Returns:
        Dictionary with report sections
    """
    generator = ReportGenerator(api_key=api_key)
    report = generator.generate(document_text, entities)
    
    return {
        "title": report.title,
        "summary": report.summary,
        "sections": report.sections,
        "full_report": report.raw_response
    }


if __name__ == "__main__":
    # Example
    sample_text = """
    SALE DEED executed on 15th January, 2024.
    Seller: Mr. Ramesh Kumar
    Buyer: Mrs. Priya Sharma
    Property: Survey No. 123/4, Indiranagar, Bangalore
    Area: 2400 Sq.Ft.
    Sale Value: Rs. 1,50,00,000/-
    """
    
    entities = {
        "PERSON": [{"text": "Mr. Ramesh Kumar"}, {"text": "Mrs. Priya Sharma"}],
        "LOCATION": [{"text": "Indiranagar, Bangalore"}],
        "AREA": [{"text": "2400 Sq.Ft."}],
        "MONEY": [{"text": "Rs. 1,50,00,000/-"}],
        "DATE": [{"text": "15th January, 2024"}],
        "PROPERTY_ID": [{"text": "Survey No. 123/4"}]
    }
    
    result = generate_report(sample_text, entities)
    print(result["full_report"])
