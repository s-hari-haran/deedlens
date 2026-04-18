"""
DeedLens Document Service
Unified service for document processing - OCR, NER, embeddings, search.
Used by both Streamlit UI and FastAPI.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from config import settings
from core.database import Database, Document, get_db
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing."""
    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    ocr_confidence: float = 0.0
    processing_time: float = 0.0


@dataclass
class SearchResult:
    """A search result."""
    doc_id: str
    title: str
    preview: str
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    entities: Dict = None


class DocumentService:
    """
    Unified service for all document operations.
    Handles OCR, NER, embeddings, and search.
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()
        self._embedding_generator = None
        self._search_engine = None
        self._ocr_engine = None
        self._ner_model = None
        self._index_loaded = False
        
        logger.info("DocumentService initialized")
    
    # -------------------------------------------------------------------------
    # Lazy Loading
    # -------------------------------------------------------------------------
    
    def _get_ocr_engine(self):
        """Lazy load OCR engine."""
        if self._ocr_engine is None:
            try:
                from ocr.ocr_engine import OCREngine, OCRBackend
                backend = OCRBackend(settings.ocr_backend)
                self._ocr_engine = OCREngine(
                    backend=backend,
                    languages=settings.ocr_languages
                )
                logger.info(f"OCR engine loaded: {settings.ocr_backend}")
            except Exception as e:
                logger.error(f"Failed to load OCR engine: {e}")
                raise
        return self._ocr_engine
    
    def _get_ner_model(self):
        """Lazy load NER model."""
        if self._ner_model is None:
            try:
                from nlp.ner_model import PropertyNERModel
                self._ner_model = PropertyNERModel()
                logger.info("NER model loaded")
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                raise
        return self._ner_model
    
    def _get_embedding_generator(self):
        """Lazy load embedding generator."""
        if self._embedding_generator is None:
            try:
                from nlp.embeddings import EmbeddingGenerator
                self._embedding_generator = EmbeddingGenerator(
                    model_name=settings.embedding_model
                )
                logger.info(f"Embedding generator loaded: {settings.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding generator: {e}")
                raise
        return self._embedding_generator
    
    def _get_search_engine(self):
        """Lazy load search engine with FAISS index."""
        if self._search_engine is None:
            try:
                from search.search_engine import SearchEngine
                from search.vector_index import VectorIndex
                
                index = VectorIndex(
                    dimension=settings.embedding_dimension,
                    index_type=settings.faiss_index_type
                )
                self._search_engine = SearchEngine(
                    vector_index=index,
                    embedding_model=settings.embedding_model
                )
                logger.info("Search engine initialized")
                
                # Load existing documents into index
                self._load_index_from_db()
                
            except Exception as e:
                logger.error(f"Failed to load search engine: {e}")
                raise
        return self._search_engine
    
    def _load_index_from_db(self):
        """Load existing document embeddings into search index."""
        if self._index_loaded:
            return
        
        try:
            docs_with_embeddings = self.db.get_documents_with_embeddings()
            
            if not docs_with_embeddings:
                logger.info("No existing embeddings to load")
                self._index_loaded = True
                return
            
            search = self._get_search_engine()
            
            for doc_id, embedding in docs_with_embeddings:
                doc = self.db.get_document(doc_id)
                if doc and embedding:
                    from search.vector_index import IndexedDocument
                    indexed_doc = IndexedDocument(
                        doc_id=doc.id,
                        file_path=doc.file_path or "",
                        title=doc.name,
                        text_preview=doc.text[:200] if doc.text else "",
                        entities=doc.entities,
                        metadata=doc.metadata
                    )
                    search.vector_index.add(embedding, indexed_doc)
                    search.document_texts[doc_id] = doc.text.lower() if doc.text else ""
            
            logger.info(f"Loaded {len(docs_with_embeddings)} documents into search index")
            self._index_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load index from database: {e}")
    
    # -------------------------------------------------------------------------
    # Document Processing
    # -------------------------------------------------------------------------
    
    def process_document(
        self,
        file_path: str,
        file_name: str,
        doc_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a document: OCR -> Clean -> NER -> Embeddings -> Save.
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original filename
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            ProcessingResult with success status and document
        """
        import time
        start_time = time.time()
        
        doc_id = doc_id or str(uuid.uuid4())[:8]
        logger.info(f"Processing document: {file_name} (id: {doc_id})")
        
        try:
            # Step 1: OCR
            logger.debug("Step 1: OCR extraction")
            ocr_result = self._extract_text(file_path)
            raw_text = ocr_result.get('text', '')
            ocr_confidence = ocr_result.get('confidence', 0.0)
            
            if not raw_text.strip():
                logger.warning(f"No text extracted from {file_name}")
            
            # Step 2: Clean text
            logger.debug("Step 2: Text cleaning")
            cleaned_text = self._clean_text(raw_text)
            
            # Step 3: Extract entities
            logger.debug("Step 3: NER extraction")
            entities = self._extract_entities(cleaned_text)
            
            # Detect document type
            doc_type = self._detect_document_type(cleaned_text, entities)
            
            # Step 4: Generate embeddings
            logger.debug("Step 4: Generating embeddings")
            embedding = self._generate_embedding(cleaned_text)
            
            # Step 5: Create and save document
            doc = Document(
                id=doc_id,
                name=file_name,
                file_path=file_path,
                text=cleaned_text,
                entities=entities,
                embedding=embedding,
                ocr_confidence=ocr_confidence,
                doc_type=doc_type
            )
            
            saved_doc = self.db.save_document(doc)
            logger.info(f"Document saved: {doc_id}")
            
            # Step 6: Add to search index
            self._add_to_search_index(saved_doc)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processed in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                document=saved_doc,
                ocr_confidence=ocr_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing document {file_name}: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _extract_text(self, file_path: str) -> Dict:
        """Extract text using OCR."""
        try:
            ocr = self._get_ocr_engine()
            result = ocr.process_file(file_path)
            return {
                'text': result.full_text,
                'confidence': result.avg_confidence,
                'pages': result.total_pages
            }
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            # Try fallback: direct text extraction for PDFs
            try:
                import fitz
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                if text.strip():
                    logger.info("Used PyMuPDF text extraction as fallback")
                    return {'text': text, 'confidence': 0.9, 'pages': 1}
            except:
                pass
            raise
    
    def _clean_text(self, raw_text: str) -> str:
        """Clean OCR text."""
        try:
            from preprocessing.text_cleaner import TextCleaner
            cleaner = TextCleaner()
            result = cleaner.clean(raw_text)
            return result.cleaned
        except ImportError:
            logger.warning("TextCleaner not available, using raw text")
            return raw_text.strip()
    
    def _extract_entities(self, text: str) -> Dict[str, List[dict]]:
        """Extract entities from text."""
        try:
            ner = self._get_ner_model()
            result = ner.extract(text)
            
            # Convert to serializable format
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
            
            return grouped
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return {}
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if not text.strip():
            return None
        
        try:
            generator = self._get_embedding_generator()
            result = generator.embed_text(text)
            return result.embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _detect_document_type(self, text: str, entities: Dict) -> str:
        """Detect document type from text and entities."""
        text_lower = text.lower()
        
        if "sale deed" in text_lower:
            return "Sale Deed"
        elif "power of attorney" in text_lower:
            return "Power of Attorney"
        elif "agreement to sell" in text_lower or "agreement of sale" in text_lower:
            return "Agreement to Sell"
        elif "gift deed" in text_lower:
            return "Gift Deed"
        elif "will" in text_lower and "testament" in text_lower:
            return "Will"
        elif "lease" in text_lower:
            return "Lease Agreement"
        
        # Check entities for document type
        doc_types = entities.get("DOCUMENT_TYPE", [])
        if doc_types:
            return doc_types[0].get("text", "Unknown")
        
        return "Property Document"
    
    def _add_to_search_index(self, doc: Document):
        """Add document to search index."""
        if not doc.embedding:
            return
        
        try:
            search = self._get_search_engine()
            from search.vector_index import IndexedDocument
            
            indexed_doc = IndexedDocument(
                doc_id=doc.id,
                file_path=doc.file_path or "",
                title=doc.name,
                text_preview=doc.text[:200] if doc.text else "",
                entities=doc.entities,
                metadata=doc.metadata
            )
            
            search.vector_index.add(doc.embedding, indexed_doc)
            search.document_texts[doc.id] = doc.text.lower() if doc.text else ""
            
            logger.debug(f"Document {doc.id} added to search index")
            
        except Exception as e:
            logger.error(f"Failed to add document to search index: {e}")
    
    # -------------------------------------------------------------------------
    # Document Retrieval
    # -------------------------------------------------------------------------
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.db.get_document(doc_id)
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Get all documents with pagination."""
        return self.db.get_all_documents(limit=limit, offset=offset)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        success = self.db.delete_document(doc_id)
        if success:
            logger.info(f"Document deleted: {doc_id}")
        return success
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "total_documents": self.db.get_document_count(),
            "entity_counts": self.db.get_entity_stats()
        }
    
    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        k: int = None,
        filters: Dict = None
    ) -> List[SearchResult]:
        """
        Search documents.
        
        Args:
            query: Search query
            mode: 'semantic', 'keyword', or 'hybrid'
            k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult
        """
        k = k or settings.search_default_k
        logger.info(f"Searching: '{query}' (mode: {mode}, k: {k})")
        
        try:
            search = self._get_search_engine()
            
            results = search.search(
                query=query,
                mode=mode,
                k=k,
                semantic_weight=settings.search_semantic_weight,
                keyword_weight=settings.search_keyword_weight,
                filters=filters
            )
            
            search_results = []
            for r in results:
                search_results.append(SearchResult(
                    doc_id=r['doc_id'],
                    title=r['title'],
                    preview=r['preview'],
                    score=r['score'],
                    semantic_score=r.get('semantic_score', 0),
                    keyword_score=r.get('keyword_score', 0),
                    entities=r.get('entities', {})
                ))
            
            logger.info(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Fallback to simple text search from database
            return self._fallback_search(query, k)
    
    def _fallback_search(self, query: str, k: int) -> List[SearchResult]:
        """Fallback search using database FTS."""
        try:
            docs = self.db.search_text(query, limit=k)
            return [
                SearchResult(
                    doc_id=doc.id,
                    title=doc.name,
                    preview=doc.text[:200] if doc.text else "",
                    score=0.8,  # Default score for FTS results
                    entities=doc.entities
                )
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------
    
    def generate_report(self, doc_id: str) -> Dict:
        """Generate a report for a document."""
        doc = self.get_document(doc_id)
        if not doc:
            return {"error": "Document not found"}
        
        try:
            from reports.report_generator import generate_report
            return generate_report(doc.text, doc.entities)
        except ImportError:
            logger.warning("Report generator not available, using fallback")
            return self._generate_fallback_report(doc)
    
    def _generate_fallback_report(self, doc: Document) -> Dict:
        """Generate a basic report without LLM."""
        entities = doc.entities
        
        def get_first(entity_type: str) -> str:
            items = entities.get(entity_type, [])
            if items and isinstance(items[0], dict):
                return items[0].get('text', 'N/A')
            return items[0] if items else 'N/A'
        
        return {
            "title": f"Property Report: {doc.name}",
            "summary": f"Document type: {doc.doc_type or 'Unknown'}",
            "sections": {
                "Property Summary": f"Location: {get_first('LOCATION')}, Area: {get_first('AREA')}",
                "Parties Involved": ", ".join(
                    e.get('text', str(e)) if isinstance(e, dict) else str(e)
                    for e in entities.get('PERSON', [])[:4]
                ) or "Not identified",
                "Transaction Details": f"Value: {get_first('MONEY')}, Date: {get_first('DATE')}",
            }
        }


# Global service instance
_service_instance: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get the global DocumentService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = DocumentService()
    return _service_instance


if __name__ == "__main__":
    # Test the service
    from core.logger import setup_logging
    setup_logging(level="DEBUG")
    
    service = get_document_service()
    
    # Get stats
    stats = service.get_stats()
    print(f"Documents: {stats['total_documents']}")
    print(f"Entities: {stats['entity_counts']}")
