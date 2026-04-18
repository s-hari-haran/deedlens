"""
DeedLens Search Engine
Hybrid search combining keyword and semantic search.

Scoring Formula:
    final_score = (α × vector_score) + (β × bm25_score) + (γ × recency_boost)
    
Where:
    α = 0.6 (semantic/vector weight)
    β = 0.3 (BM25/keyword weight)  
    γ = 0.1 (recency boost)

This is engineered search, not just "vector + keyword".
"""

import re
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

from .vector_index import VectorIndex, IndexedDocument, SearchResult

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Configurable search scoring weights."""
    # Core weights (must sum to 1.0)
    vector_weight: float = 0.6      # α - semantic similarity
    bm25_weight: float = 0.3        # β - keyword matching (BM25)
    recency_weight: float = 0.1     # γ - recency boost
    
    # BM25 parameters
    bm25_k1: float = 1.5            # Term frequency saturation
    bm25_b: float = 0.75            # Length normalization
    
    # Recency decay
    recency_decay_days: int = 365   # Full decay over 1 year
    
    # Result limits
    initial_fetch_multiplier: int = 3  # Fetch 3x results for reranking
    
    def validate(self):
        """Ensure weights sum to 1.0."""
        total = self.vector_weight + self.bm25_weight + self.recency_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


DEFAULT_CONFIG = SearchConfig()


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class HybridSearchResult:
    """Combined search result with detailed scoring breakdown."""
    doc_id: str
    document: IndexedDocument
    
    # Score components
    vector_score: float
    bm25_score: float
    recency_score: float
    
    # Final combined score
    combined_score: float
    
    # Explanation (for debugging/transparency)
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedQuery:
    """Parsed query with extracted filters."""
    original: str
    cleaned: str
    terms: List[str]
    
    # Extracted filters
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    location: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    doc_type: Optional[str] = None
    
    # Entities to boost
    boost_terms: List[str] = field(default_factory=list)


# =============================================================================
# BM25 IMPLEMENTATION
# =============================================================================

class BM25Index:
    """
    BM25 (Best Matching 25) keyword search.
    
    This is a proper implementation, not just term frequency counting.
    BM25 is the same algorithm used by Elasticsearch/Lucene.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Document data
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.doc_count: int = 0
        
        # Term frequencies
        self.term_freqs: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: freq}
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term -> num_docs_containing
        
        # Original texts for snippet generation
        self.doc_texts: Dict[str, str] = {}
    
    def add_document(self, doc_id: str, text: str):
        """Add a document to the BM25 index."""
        terms = self._tokenize(text)
        
        self.doc_lengths[doc_id] = len(terms)
        self.doc_texts[doc_id] = text
        self.doc_count += 1
        
        # Update average document length
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
        
        # Count term frequencies
        term_counts = defaultdict(int)
        for term in terms:
            term_counts[term] += 1
        
        # Update indexes
        for term, freq in term_counts.items():
            if doc_id not in self.term_freqs[term]:
                self.doc_freqs[term] += 1
            self.term_freqs[term][doc_id] = freq
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Lowercase and extract words
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        # Filter short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'it'}
        return [w for w in words if len(w) > 2 and w not in stop_words]
    
    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency."""
        if term not in self.doc_freqs:
            return 0
        
        df = self.doc_freqs[term]
        # Standard IDF formula with smoothing
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_id: str) -> float:
        """Calculate BM25 score for a query-document pair."""
        if doc_id not in self.doc_lengths:
            return 0
        
        query_terms = self._tokenize(query)
        doc_length = self.doc_lengths[doc_id]
        
        score = 0
        for term in query_terms:
            if term not in self.term_freqs or doc_id not in self.term_freqs[term]:
                continue
            
            tf = self.term_freqs[term][doc_id]
            idf = self._idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search and return top-k documents with BM25 scores."""
        query_terms = self._tokenize(query)
        
        # Find candidate documents (those containing at least one query term)
        candidates = set()
        for term in query_terms:
            if term in self.term_freqs:
                candidates.update(self.term_freqs[term].keys())
        
        # Score all candidates
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in candidates]
        
        # Sort by score descending
        scores.sort(key=lambda x: -x[1])
        
        return scores[:k]


# =============================================================================
# QUERY PARSER
# =============================================================================

class QueryParser:
    """
    Parse natural language queries to extract filters.
    
    Examples:
        "sale deeds in Bangalore above 50 lakhs" 
        -> location=Bangalore, price_min=5000000, doc_type=sale_deed
        
        "gift deeds from 2023"
        -> doc_type=gift_deed, date_from=2023-01-01
    """
    
    # Price patterns
    PRICE_PATTERNS = [
        (r'(?:above|over|more than|>\s*)\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr)', 'min'),
        (r'(?:below|under|less than|<\s*)\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr)', 'max'),
        (r'(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr)', 'range'),
    ]
    
    # Location keywords
    LOCATION_PREPOSITIONS = ['in', 'at', 'near', 'around']
    
    # Document types
    DOC_TYPES = {
        'sale deed': 'sale_deed',
        'sale deeds': 'sale_deed',
        'gift deed': 'gift_deed',
        'gift deeds': 'gift_deed',
        'lease': 'lease',
        'lease deed': 'lease',
        'mortgage': 'mortgage',
        'will': 'will',
        'poa': 'power_of_attorney',
        'power of attorney': 'power_of_attorney',
    }
    
    def parse(self, query: str) -> ParsedQuery:
        """Parse query and extract filters."""
        parsed = ParsedQuery(
            original=query,
            cleaned=query,
            terms=[]
        )
        
        working_query = query.lower()
        
        # Extract price filters
        parsed.price_min, parsed.price_max, working_query = self._extract_price(working_query)
        
        # Extract document type
        parsed.doc_type, working_query = self._extract_doc_type(working_query)
        
        # Extract location
        parsed.location, working_query = self._extract_location(working_query)
        
        # Extract date filters
        parsed.date_from, parsed.date_to, working_query = self._extract_date(working_query)
        
        # Clean remaining query
        parsed.cleaned = self._clean_query(working_query)
        parsed.terms = [t for t in parsed.cleaned.split() if len(t) > 2]
        
        return parsed
    
    def _extract_price(self, query: str) -> Tuple[Optional[float], Optional[float], str]:
        """Extract price filters from query."""
        price_min = None
        price_max = None
        
        for pattern, filter_type in self.PRICE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if filter_type == 'min':
                    amount, unit = match.groups()
                    price_min = self._parse_amount(float(amount), unit)
                elif filter_type == 'max':
                    amount, unit = match.groups()
                    price_max = self._parse_amount(float(amount), unit)
                elif filter_type == 'range':
                    min_amt, min_unit, max_amt, max_unit = match.groups()
                    price_min = self._parse_amount(float(min_amt), min_unit)
                    price_max = self._parse_amount(float(max_amt), max_unit)
                
                # Remove matched text
                query = query[:match.start()] + query[match.end():]
        
        return price_min, price_max, query
    
    def _parse_amount(self, amount: float, unit: str) -> float:
        """Convert amount to absolute value."""
        unit = unit.lower()
        if unit.startswith('cr'):
            return amount * 10000000  # 1 crore = 10 million
        elif unit.startswith('l') or unit.startswith('lac'):
            return amount * 100000  # 1 lakh = 100,000
        return amount
    
    def _extract_doc_type(self, query: str) -> Tuple[Optional[str], str]:
        """Extract document type from query."""
        for phrase, doc_type in self.DOC_TYPES.items():
            if phrase in query.lower():
                query = re.sub(re.escape(phrase), '', query, flags=re.IGNORECASE)
                return doc_type, query
        return None, query
    
    def _extract_location(self, query: str) -> Tuple[Optional[str], str]:
        """Extract location from query."""
        for prep in self.LOCATION_PREPOSITIONS:
            pattern = rf'\b{prep}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1)
                query = query[:match.start()] + query[match.end():]
                return location, query
        return None, query
    
    def _extract_date(self, query: str) -> Tuple[Optional[datetime], Optional[datetime], str]:
        """Extract date filters from query."""
        date_from = None
        date_to = None
        
        # Year patterns
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            
            # Check for "from" or "since"
            if re.search(rf'(?:from|since|after)\s+{year}', query, re.IGNORECASE):
                date_from = datetime(year, 1, 1)
            # Check for "before" or "until"
            elif re.search(rf'(?:before|until|by)\s+{year}', query, re.IGNORECASE):
                date_to = datetime(year, 12, 31)
            # Just year mentioned - assume that year
            else:
                date_from = datetime(year, 1, 1)
                date_to = datetime(year, 12, 31)
            
            query = re.sub(rf'\b{year}\b', '', query)
        
        return date_from, date_to, query
    
    def _clean_query(self, query: str) -> str:
        """Clean up query after filter extraction."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'with', 'for', 'from', 'in', 'at', 'to'}
        words = query.split()
        words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(words)


# =============================================================================
# MAIN SEARCH ENGINE
# =============================================================================

class SearchEngine:
    """
    Hybrid search engine with engineered scoring.
    
    Features:
    - BM25 for keyword search (not just term frequency)
    - Semantic search with sentence transformers
    - Query parsing for filter extraction
    - Configurable scoring weights
    - Recency boosting
    - Result explanation
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: SearchConfig = None
    ):
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        
        self._model = None
        self.bm25_index = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.query_parser = QueryParser()
        
        # Document metadata for filtering
        self.doc_metadata: Dict[str, Dict] = {}
        self.doc_timestamps: Dict[str, datetime] = {}
    
    def _load_model(self):
        """Lazy load embedding model."""
        if self._model is None and EMBEDDINGS_AVAILABLE:
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    def _get_embedding(self, text: str):
        """Generate embedding for text."""
        model = self._load_model()
        return model.encode(text, convert_to_numpy=True)
    
    def index_document(
        self,
        doc_id: str,
        text: str,
        embedding,
        file_path: str = "",
        title: str = "",
        entities: Dict = None,
        metadata: Dict = None,
        created_at: datetime = None
    ):
        """Index a document for hybrid search."""
        # Create document for vector index
        doc = IndexedDocument(
            doc_id=doc_id,
            file_path=file_path,
            title=title or doc_id,
            text_preview=text[:500] if text else "",
            entities=entities,
            metadata=metadata
        )
        
        # Add to vector index
        self.vector_index.add(embedding, doc)
        
        # Add to BM25 index
        self.bm25_index.add_document(doc_id, text)
        
        # Store metadata for filtering
        self.doc_metadata[doc_id] = metadata or {}
        self.doc_timestamps[doc_id] = created_at or datetime.utcnow()
    
    def add_document(self, doc_id: str, embedding):
        """Add document to vector index (compatibility method)."""
        # This is called by existing code that just wants vector search
        # We'll create a minimal IndexedDocument
        doc = IndexedDocument(
            doc_id=doc_id,
            file_path="",
            title=doc_id,
            text_preview="",
            entities={},
            metadata={}
        )
        self.vector_index.add(embedding, doc)
    
    def _calculate_recency_score(self, doc_id: str) -> float:
        """Calculate recency score (0-1, where 1 is most recent)."""
        if doc_id not in self.doc_timestamps:
            return 0.5  # Default for unknown timestamps
        
        doc_time = self.doc_timestamps[doc_id]
        now = datetime.utcnow()
        age_days = (now - doc_time).days
        
        # Exponential decay
        decay_rate = 1 / self.config.recency_decay_days
        score = math.exp(-decay_rate * age_days)
        
        return min(1.0, max(0.0, score))
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return {}
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores}
        
        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }
    
    def _apply_filters(self, doc_id: str, parsed_query: ParsedQuery) -> bool:
        """Check if document passes all filters."""
        metadata = self.doc_metadata.get(doc_id, {})
        
        # Price filter
        if parsed_query.price_min is not None:
            doc_price = metadata.get('price', 0)
            if doc_price < parsed_query.price_min:
                return False
        
        if parsed_query.price_max is not None:
            doc_price = metadata.get('price', float('inf'))
            if doc_price > parsed_query.price_max:
                return False
        
        # Location filter
        if parsed_query.location:
            doc_location = str(metadata.get('location', '')).lower()
            if parsed_query.location.lower() not in doc_location:
                return False
        
        # Document type filter
        if parsed_query.doc_type:
            doc_type = metadata.get('doc_type', '')
            if doc_type != parsed_query.doc_type:
                return False
        
        # Date filter
        if parsed_query.date_from or parsed_query.date_to:
            doc_time = self.doc_timestamps.get(doc_id)
            if doc_time:
                if parsed_query.date_from and doc_time < parsed_query.date_from:
                    return False
                if parsed_query.date_to and doc_time > parsed_query.date_to:
                    return False
        
        return True
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        explain: bool = False,
        filters: Dict = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search with engineered scoring.
        
        Score formula:
            final = (α × vector) + (β × bm25) + (γ × recency)
        
        Args:
            query: Search query
            k: Number of results
            explain: Include scoring explanation
            filters: Manual filters (overrides parsed filters)
        
        Returns:
            List of HybridSearchResult sorted by combined_score
        """
        # Parse query to extract filters
        parsed = self.query_parser.parse(query)
        search_query = parsed.cleaned or query
        
        # Fetch more results initially for reranking
        fetch_k = k * self.config.initial_fetch_multiplier
        
        # Get vector search results
        query_embedding = self._get_embedding(search_query)
        vector_results = self.vector_index.search(query_embedding, fetch_k)
        vector_scores = {r.doc_id: r.score for r in vector_results}
        
        # Get BM25 results
        bm25_results = self.bm25_index.search(search_query, fetch_k)
        bm25_scores = dict(bm25_results)
        
        # Normalize scores
        vector_scores = self._normalize_scores(vector_scores)
        bm25_scores = self._normalize_scores(bm25_scores)
        
        # Combine candidates
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        results = []
        for doc_id in all_doc_ids:
            # Apply filters
            if not self._apply_filters(doc_id, parsed):
                continue
            
            # Get component scores
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            r_score = self._calculate_recency_score(doc_id)
            
            # Calculate combined score
            combined = (
                self.config.vector_weight * v_score +
                self.config.bm25_weight * b_score +
                self.config.recency_weight * r_score
            )
            
            # Get document
            doc = self.vector_index.get_by_doc_id(doc_id)
            if doc is None:
                continue
            
            # Build explanation if requested
            explanation = {}
            if explain:
                explanation = {
                    'formula': f"({self.config.vector_weight} × {v_score:.3f}) + ({self.config.bm25_weight} × {b_score:.3f}) + ({self.config.recency_weight} × {r_score:.3f})",
                    'components': {
                        'vector': {'score': v_score, 'weight': self.config.vector_weight, 'contribution': v_score * self.config.vector_weight},
                        'bm25': {'score': b_score, 'weight': self.config.bm25_weight, 'contribution': b_score * self.config.bm25_weight},
                        'recency': {'score': r_score, 'weight': self.config.recency_weight, 'contribution': r_score * self.config.recency_weight},
                    },
                    'filters_applied': {
                        'price_min': parsed.price_min,
                        'price_max': parsed.price_max,
                        'location': parsed.location,
                        'doc_type': parsed.doc_type,
                    }
                }
            
            results.append(HybridSearchResult(
                doc_id=doc_id,
                document=doc,
                vector_score=v_score,
                bm25_score=b_score,
                recency_score=r_score,
                combined_score=combined,
                explanation=explanation
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: -x.combined_score)
        
        return results[:k]
    
    def semantic_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform semantic-only search."""
        query_embedding = self._get_embedding(query)
        return self.vector_index.search(query_embedding, k)
    
    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search."""
        return self.bm25_index.search(query, k)
    
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        k: int = 10,
        explain: bool = False,
        **kwargs
    ) -> List[Dict]:
        """
        Unified search interface.
        
        Args:
            query: Search query
            mode: 'semantic', 'keyword', or 'hybrid'
            k: Number of results
            explain: Include scoring explanation
            **kwargs: Additional arguments
        
        Returns:
            List of search results as dictionaries
        """
        if mode == "semantic":
            results = self.semantic_search(query, k)
            return [
                {
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "semantic_score": r.score,
                    "keyword_score": 0,
                    "title": r.document.title,
                    "preview": r.document.text_preview,
                    "entities": r.document.entities
                }
                for r in results
            ]
        
        elif mode == "keyword":
            results = self.keyword_search(query, k)
            output = []
            for doc_id, score in results:
                doc = self.vector_index.get_by_doc_id(doc_id)
                if doc:
                    output.append({
                        "doc_id": doc_id,
                        "score": score,
                        "semantic_score": 0,
                        "keyword_score": score,
                        "title": doc.title,
                        "preview": doc.text_preview,
                        "entities": doc.entities
                    })
            return output
        
        else:  # hybrid
            results = self.hybrid_search(query, k, explain=explain, **kwargs)
            return [
                {
                    "doc_id": r.doc_id,
                    "score": r.combined_score,
                    "semantic_score": r.vector_score,
                    "keyword_score": r.bm25_score,
                    "recency_score": r.recency_score,
                    "title": r.document.title,
                    "preview": r.document.text_preview,
                    "entities": r.document.entities,
                    "explanation": r.explanation if explain else None
                }
                for r in results
            ]


def create_search_engine(
    dimension: int = 384,
    embedding_model: str = "all-MiniLM-L6-v2",
    config: SearchConfig = None
) -> SearchEngine:
    """Create a new search engine with vector index."""
    index = VectorIndex(dimension=dimension)
    return SearchEngine(index, embedding_model, config)


if __name__ == "__main__":
    # Example usage
    print("Search Engine module loaded successfully")
    
    # Demo query parsing
    parser = QueryParser()
    test_queries = [
        "sale deeds in Bangalore above 50 lakhs",
        "gift deeds from 2023",
        "properties near Koramangala below 1 crore",
    ]
    
    for q in test_queries:
        parsed = parser.parse(q)
        print(f"\nQuery: {q}")
        print(f"  Cleaned: {parsed.cleaned}")
        print(f"  Filters: price_min={parsed.price_min}, price_max={parsed.price_max}, location={parsed.location}, doc_type={parsed.doc_type}")
