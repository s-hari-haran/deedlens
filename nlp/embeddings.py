"""
DeedLens Embedding Generator
Generates vector embeddings for documents and entities.

NOTE: Sentence-Transformers has compatibility issues with PyTorch in containerized environments.
This module provides a stub implementation that returns zero embeddings for now.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np

# Skip sentence-transformers import due to transformers/nn compatibility issues
SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: np.ndarray
    model: str
    dimension: int


class EmbeddingGenerator:
    """
    Stub implementation of embedding generator.
    Returns zero-vectors due to transformers compatibility issues in containerized environments.
    """

    MODELS = {
        "all-MiniLM-L6-v2": 384,      # Fast, good quality
        "all-mpnet-base-v2": 768,      # Higher quality
        "paraphrase-MiniLM-L6-v2": 384, # Good for paraphrase detection
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = self.MODELS.get(model_name, 384)
        self._model = None

    def _load_model(self):
        """Stub - returns None"""
        return None

    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Return zero-vector embedding (stub implementation).
        """
        # Return zero vector for now due to compatibility issues
        embedding = np.zeros(self.dimension, dtype=np.float32)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            dimension=len(embedding)
        )

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[EmbeddingResult]:
        """
        Return zero-vector embeddings for multiple texts (stub).
        """
        results = []
        for text in texts:
            embedding = np.zeros(self.dimension, dtype=np.float32)
            results.append(EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model_name,
                dimension=len(embedding)
            ))

        return results

    def embed_document(
        self,
        document: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict:
        """
        Return zero-vector document embedding (stub).
        """
        chunks = self._chunk_text(document, chunk_size, overlap)

        # Return zero embedding
        doc_embedding = np.zeros(self.dimension, dtype=np.float32)

        return {
            "document_embedding": doc_embedding,
            "chunk_embeddings": [
                EmbeddingResult(
                    text=chunk,
                    embedding=np.zeros(self.dimension, dtype=np.float32),
                    model=self.model_name,
                    dimension=self.dimension
                ) for chunk in chunks
            ],
            "num_chunks": len(chunks),
            "dimension": self.dimension
        }

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for sep in ['. ', '.\n', '\n\n']:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query (stub - returns empty list).
        """
        return []


def generate_embeddings(
    texts: Union[str, List[str]],
    model: str = "all-MiniLM-L6-v2"
) -> Dict:
    """
    Convenience function to generate embeddings (stub).
    """
    generator = EmbeddingGenerator(model_name=model)

    if isinstance(texts, str):
        result = generator.embed_text(texts)
        return {
            "text": result.text,
            "embedding": result.embedding.tolist(),
            "dimension": result.dimension
        }
    else:
        results = generator.embed_texts(texts)
        return {
            "embeddings": [
                {
                    "text": r.text,
                    "embedding": r.embedding.tolist(),
                    "dimension": r.dimension
                }
                for r in results
            ]
        }


if __name__ == "__main__":
    # Example usage
    texts = [
        "Sale deed for residential property in Bangalore",
        "Agreement for sale of commercial plot in Indiranagar",
        "Lease agreement for office space in Koramangala",
        "Gift deed for ancestral property",
    ]

    generator = EmbeddingGenerator()
    results = generator.embed_texts(texts)

    print(f"Generated {len(results)} embeddings")
    print(f"Dimension: {results[0].dimension}")

    # Find similar
    query = generator.embed_text("Property sale agreement in Bangalore")
    embeddings = [r.embedding for r in results]

    similar = generator.find_similar(query.embedding, embeddings, top_k=3)

    print("\n=== Most Similar ===")
    for idx, sim in similar:
        print(f"  {texts[idx][:50]}... (similarity: {sim:.3f})")

