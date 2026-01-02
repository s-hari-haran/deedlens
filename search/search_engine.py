"""
DeedLens Search Engine
Hybrid search combining keyword and semantic search.
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from .vector_index import VectorIndex, IndexedDocument, SearchResult

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class HybridSearchResult:
    """Combined search result with scores."""
    doc_id: str
    document: IndexedDocument
    semantic_score: float
    keyword_score: float
    combined_score: float


class SearchEngine:
    """
    Hybrid search engine combining semantic and keyword search.
    Supports filtering by metadata.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self._model = None
        
        # In-memory keyword index
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)
        self.document_texts: Dict[str, str] = {}
    
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
        metadata: Dict = None
    ):
        """
        Index a document for both semantic and keyword search.
        
        Args:
            doc_id: Unique document ID
            text: Full document text
            embedding: Pre-computed embedding vector
            file_path: Path to original file
            title: Document title
            entities: Extracted entities
            metadata: Additional metadata
        """
        # Create document
        doc = IndexedDocument(
            doc_id=doc_id,
            file_path=file_path,
            title=title or doc_id,
            text_preview=text[:200] if text else "",
            entities=entities,
            metadata=metadata
        )
        
        # Add to vector index
        self.vector_index.add(embedding, doc)
        
        # Index for keyword search
        self.document_texts[doc_id] = text.lower()
        
        # Build keyword index
        words = re.findall(r'\w+', text.lower())
        for word in set(words):
            if len(word) > 2:  # Skip very short words
                self.keyword_index[word].append(doc_id)
    
    def semantic_search(
        self,
        query: str,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of SearchResult
        """
        query_embedding = self._get_embedding(query)
        return self.vector_index.search(query_embedding, k)
    
    def keyword_search(
        self,
        query: str,
        k: int = 10
    ) -> List[tuple]:
        """
        Perform keyword search.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of (doc_id, score) tuples
        """
        query_words = re.findall(r'\w+', query.lower())
        
        # Count matches per document
        doc_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.keyword_index:
                matching_docs = self.keyword_index[word]
                for doc_id in matching_docs:
                    doc_scores[doc_id] += 1
        
        # Normalize by query length
        for doc_id in doc_scores:
            doc_scores[doc_id] /= len(query_words)
        
        # Sort by score
        results = sorted(doc_scores.items(), key=lambda x: -x[1])
        
        return results[:k]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Dict = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining semantic and keyword.
        
        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            filters: Optional metadata filters
        
        Returns:
            List of HybridSearchResult
        """
        # Get semantic results
        semantic_results = self.semantic_search(query, k * 2)
        semantic_scores = {r.doc_id: r.score for r in semantic_results}
        
        # Get keyword results
        keyword_results = self.keyword_search(query, k * 2)
        keyword_scores = dict(keyword_results)
        
        # Combine results
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            sem_score = semantic_scores.get(doc_id, 0)
            kw_score = keyword_scores.get(doc_id, 0)
            
            combined = (sem_score * semantic_weight) + (kw_score * keyword_weight)
            
            # Get document
            doc = self.vector_index.get_by_doc_id(doc_id)
            if doc is None:
                continue
            
            # Apply filters
            if filters and not self._matches_filters(doc, filters):
                continue
            
            combined_results.append(HybridSearchResult(
                doc_id=doc_id,
                document=doc,
                semantic_score=sem_score,
                keyword_score=kw_score,
                combined_score=combined
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: -x.combined_score)
        
        return combined_results[:k]
    
    def _matches_filters(self, doc: IndexedDocument, filters: Dict) -> bool:
        """Check if document matches filters."""
        if not doc.metadata:
            return False
        
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            
            doc_value = doc.metadata[key]
            
            # Handle different filter types
            if isinstance(value, dict):
                # Range filters
                if "min" in value and doc_value < value["min"]:
                    return False
                if "max" in value and doc_value > value["max"]:
                    return False
            elif isinstance(value, list):
                # IN filter
                if doc_value not in value:
                    return False
            else:
                # Exact match
                if doc_value != value:
                    return False
        
        return True
    
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        k: int = 10,
        **kwargs
    ) -> List[Dict]:
        """
        Unified search interface.
        
        Args:
            query: Search query
            mode: 'semantic', 'keyword', or 'hybrid'
            k: Number of results
            **kwargs: Additional arguments for specific search modes
        
        Returns:
            List of search results as dictionaries
        """
        if mode == "semantic":
            results = self.semantic_search(query, k)
            return [
                {
                    "doc_id": r.doc_id,
                    "score": r.score,
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
                        "title": doc.title,
                        "preview": doc.text_preview,
                        "entities": doc.entities
                    })
            return output
        
        else:  # hybrid
            results = self.hybrid_search(query, k, **kwargs)
            return [
                {
                    "doc_id": r.doc_id,
                    "score": r.combined_score,
                    "semantic_score": r.semantic_score,
                    "keyword_score": r.keyword_score,
                    "title": r.document.title,
                    "preview": r.document.text_preview,
                    "entities": r.document.entities
                }
                for r in results
            ]


def create_search_engine(
    dimension: int = 384,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> SearchEngine:
    """Create a new search engine with vector index."""
    index = VectorIndex(dimension=dimension)
    return SearchEngine(index, embedding_model)


if __name__ == "__main__":
    # Example usage
    print("Search Engine module loaded successfully")
