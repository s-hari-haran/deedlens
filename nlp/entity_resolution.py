"""
DeedLens Entity Resolution
Resolves duplicate and inconsistent entity mentions.
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class ResolvedEntity:
    """A resolved/canonical entity."""
    canonical_name: str
    entity_type: str
    mentions: List[str]
    confidence: float
    metadata: Dict


class EntityResolver:
    """
    Resolves entity mentions to canonical forms.
    Uses embedding similarity and rule-based matching.
    """
    
    # Common name variations and abbreviations
    NAME_PREFIXES = {
        'mr', 'mrs', 'ms', 'dr', 'shri', 'smt', 'kumari', 'sri', 'late',
        's/o', 'd/o', 'w/o', 'c/o', 'son of', 'daughter of', 'wife of'
    }
    
    # Location normalizations
    LOCATION_NORMALIZATIONS = {
        'blr': 'Bangalore',
        'bengaluru': 'Bangalore',
        'b\'lore': 'Bangalore',
        'chn': 'Chennai',
        'hyd': 'Hyderabad',
        'mum': 'Mumbai',
        'del': 'Delhi',
        'ncr': 'Delhi NCR',
    }
    
    def __init__(
        self,
        use_embeddings: bool = True,
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self._model = None
    
    def _load_embedding_model(self):
        """Lazy load sentence transformer."""
        if self._model is None and EMBEDDINGS_AVAILABLE:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a person name."""
        # Convert to lowercase for processing
        normalized = name.lower().strip()
        
        # Remove prefixes
        for prefix in self.NAME_PREFIXES:
            pattern = rf'^{re.escape(prefix)}\.?\s*'
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Title case
        normalized = normalized.title()
        
        return normalized
    
    def _normalize_location(self, location: str) -> str:
        """Normalize a location name."""
        normalized = location.lower().strip()
        
        # Apply known normalizations
        for abbrev, full in self.LOCATION_NORMALIZATIONS.items():
            if normalized == abbrev or normalized.startswith(abbrev + ' '):
                normalized = normalized.replace(abbrev, full.lower(), 1)
        
        # Remove common suffixes that might vary
        normalized = re.sub(r'\s*-\s*\d+', '', normalized)  # Remove PIN codes
        
        return normalized.title()
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self.use_embeddings:
            # Fall back to simple string similarity
            return self._string_similarity(text1, text2)
        
        model = self._load_embedding_model()
        embeddings = model.encode([text1, text2])
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity based on common characters."""
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
        
        # Check if one is substring of another
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 0.9
        
        # Check initials match
        s1_words = s1_lower.split()
        s2_words = s2_lower.split()
        
        if len(s1_words) > 0 and len(s2_words) > 0:
            # Check if last names match
            if s1_words[-1] == s2_words[-1]:
                return 0.8
            
            # Check initials
            s1_initials = ''.join(w[0] for w in s1_words if w)
            s2_initials = ''.join(w[0] for w in s2_words if w)
            if s1_initials == s2_initials:
                return 0.75
        
        # Character-based similarity
        common = set(s1_lower) & set(s2_lower)
        total = set(s1_lower) | set(s2_lower)
        
        return len(common) / len(total) if total else 0.0
    
    def _cluster_entities(
        self,
        entities: List[str],
        entity_type: str
    ) -> List[List[str]]:
        """Cluster similar entities together."""
        if not entities:
            return []
        
        # Normalize based on type
        if entity_type == "PERSON":
            normalized = [self._normalize_name(e) for e in entities]
        elif entity_type == "LOCATION":
            normalized = [self._normalize_location(e) for e in entities]
        else:
            normalized = entities
        
        # Create clusters
        clusters = []
        used = set()
        
        for i, (entity, norm) in enumerate(zip(entities, normalized)):
            if i in used:
                continue
            
            cluster = [entity]
            used.add(i)
            
            for j in range(i + 1, len(entities)):
                if j in used:
                    continue
                
                # Check similarity
                sim = self._compute_similarity(norm, normalized[j])
                
                if sim >= self.similarity_threshold:
                    cluster.append(entities[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _select_canonical(self, mentions: List[str], entity_type: str) -> str:
        """Select the canonical form from a list of mentions."""
        if not mentions:
            return ""
        
        if len(mentions) == 1:
            return mentions[0]
        
        # Prefer longer, more complete names
        if entity_type == "PERSON":
            # Prefer names with more parts
            scored = [(m, len(m.split())) for m in mentions]
            scored.sort(key=lambda x: (-x[1], -len(x[0])))
            return scored[0][0]
        
        # Default: prefer the longest mention
        return max(mentions, key=len)
    
    def resolve(
        self,
        entities: Dict[str, List[str]]
    ) -> Dict[str, List[ResolvedEntity]]:
        """
        Resolve entities by type.
        
        Args:
            entities: Dictionary mapping entity type to list of mentions
        
        Returns:
            Dictionary mapping entity type to resolved entities
        """
        resolved = {}
        
        for entity_type, mentions in entities.items():
            # Get unique mentions
            unique_mentions = list(set(mentions))
            
            # Cluster similar entities
            clusters = self._cluster_entities(unique_mentions, entity_type)
            
            # Create resolved entities
            resolved_list = []
            for cluster in clusters:
                canonical = self._select_canonical(cluster, entity_type)
                
                resolved_list.append(ResolvedEntity(
                    canonical_name=canonical,
                    entity_type=entity_type,
                    mentions=cluster,
                    confidence=1.0 if len(cluster) == 1 else 0.9,
                    metadata={"cluster_size": len(cluster)}
                ))
            
            resolved[entity_type] = resolved_list
        
        return resolved


def resolve_entities(entities: Dict[str, List[str]]) -> Dict:
    """
    Convenience function to resolve entities.
    
    Args:
        entities: Dictionary mapping entity type to list of mentions
    
    Returns:
        Dictionary with resolved entities
    """
    resolver = EntityResolver()
    resolved = resolver.resolve(entities)
    
    result = {}
    for entity_type, resolved_list in resolved.items():
        result[entity_type] = [
            {
                "canonical": r.canonical_name,
                "mentions": r.mentions,
                "confidence": r.confidence
            }
            for r in resolved_list
        ]
    
    return result


if __name__ == "__main__":
    # Example usage
    entities = {
        "PERSON": [
            "Mr. Ramesh Kumar",
            "Ramesh Kumar",
            "R. Kumar",
            "Mrs. Priya Sharma",
            "Priya S.",
        ],
        "LOCATION": [
            "Bangalore",
            "Bengaluru",
            "Indiranagar, Bangalore",
            "Koramangala, Bangalore",
        ]
    }
    
    resolved = resolve_entities(entities)
    
    print("=== Resolved Entities ===")
    for entity_type, entities_list in resolved.items():
        print(f"\n{entity_type}:")
        for e in entities_list:
            print(f"  Canonical: {e['canonical']}")
            print(f"  Mentions: {e['mentions']}")
