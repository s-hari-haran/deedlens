"""
DeedLens Vector Index
FAISS-based vector store for document embeddings.
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class IndexedDocument:
    """Metadata for an indexed document."""
    doc_id: str
    file_path: str
    title: Optional[str] = None
    text_preview: Optional[str] = None
    entities: Optional[Dict] = None
    metadata: Optional[Dict] = None


@dataclass
class SearchResult:
    """A single search result."""
    doc_id: str
    score: float
    document: IndexedDocument


class VectorIndex:
    """
    FAISS-based vector index for semantic search.
    Stores document embeddings and metadata.
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat"
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install faiss-cpu.")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.documents: Dict[int, IndexedDocument] = {}
        self.id_counter = 0
    
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index."""
        if self.index_type == "flat":
            # Exact search (L2 distance)
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "cosine":
            # Inner product (for normalized vectors = cosine similarity)
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            return faiss.IndexFlatL2(self.dimension)
    
    def add(
        self,
        embedding: np.ndarray,
        document: IndexedDocument
    ) -> int:
        """
        Add a document to the index.
        
        Args:
            embedding: Document embedding vector
            document: Document metadata
        
        Returns:
            Internal ID assigned to the document
        """
        # Ensure correct shape
        embedding = np.array(embedding).astype('float32')
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store metadata
        internal_id = self.id_counter
        self.documents[internal_id] = document
        self.id_counter += 1
        
        return internal_id
    
    def add_batch(
        self,
        embeddings: List[np.ndarray],
        documents: List[IndexedDocument]
    ) -> List[int]:
        """Add multiple documents at once."""
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        ids = []
        for doc in documents:
            internal_id = self.id_counter
            self.documents[internal_id] = doc
            ids.append(internal_id)
            self.id_counter += 1
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
        
        Returns:
            List of SearchResult objects
        """
        # Ensure correct shape
        query = np.array(query_embedding).astype('float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.documents:
                # Convert L2 distance to similarity score
                score = 1 / (1 + dist)  # Simple conversion
                
                results.append(SearchResult(
                    doc_id=self.documents[idx].doc_id,
                    score=float(score),
                    document=self.documents[idx]
                ))
        
        return results
    
    def get_document(self, internal_id: int) -> Optional[IndexedDocument]:
        """Get document by internal ID."""
        return self.documents.get(internal_id)
    
    def get_by_doc_id(self, doc_id: str) -> Optional[IndexedDocument]:
        """Get document by document ID."""
        for doc in self.documents.values():
            if doc.doc_id == doc_id:
                return doc
        return None
    
    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return self.index.ntotal
    
    def save(self, directory: str):
        """
        Save index and metadata to disk.
        
        Args:
            directory: Directory to save to
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "id_counter": self.id_counter,
            "documents": {
                str(k): asdict(v)
                for k, v in self.documents.items()
            }
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, directory: str) -> "VectorIndex":
        """
        Load index from disk.
        
        Args:
            directory: Directory to load from
        
        Returns:
            Loaded VectorIndex
        """
        path = Path(directory)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"]
        )
        
        # Load FAISS index
        instance.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents
        instance.id_counter = metadata["id_counter"]
        instance.documents = {
            int(k): IndexedDocument(**v)
            for k, v in metadata["documents"].items()
        }
        
        return instance


def create_index(dimension: int = 384) -> VectorIndex:
    """Create a new vector index."""
    return VectorIndex(dimension=dimension)


if __name__ == "__main__":
    # Example usage
    import random
    
    # Create index
    index = VectorIndex(dimension=384)
    
    # Add some dummy documents
    for i in range(5):
        embedding = np.random.rand(384).astype('float32')
        doc = IndexedDocument(
            doc_id=f"doc_{i}",
            file_path=f"/path/to/doc_{i}.pdf",
            title=f"Document {i}",
            text_preview=f"This is document {i}..."
        )
        index.add(embedding, doc)
    
    print(f"Index size: {index.size}")
    
    # Search
    query = np.random.rand(384).astype('float32')
    results = index.search(query, k=3)
    
    print("\n=== Search Results ===")
    for r in results:
        print(f"  {r.doc_id}: score={r.score:.3f} - {r.document.title}")
