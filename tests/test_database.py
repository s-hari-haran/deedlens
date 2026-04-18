"""
Tests for core.database module.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import Database, Document


class TestDatabase:
    """Tests for Database class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = Database(db_path)
        yield db
        
        # Cleanup
        if db_path.exists():
            os.unlink(db_path)
    
    def test_save_and_retrieve_document(self, temp_db):
        """Test saving and retrieving a document."""
        doc = Document(
            id="test_001",
            name="test_document.pdf",
            text="This is test content",
            entities={"PERSON": [{"text": "John Doe"}]},
            ocr_confidence=0.95
        )
        
        saved = temp_db.save_document(doc)
        
        assert saved.created_at is not None
        assert saved.updated_at is not None
        
        retrieved = temp_db.get_document("test_001")
        
        assert retrieved is not None
        assert retrieved.id == "test_001"
        assert retrieved.name == "test_document.pdf"
        assert retrieved.text == "This is test content"
        assert "PERSON" in retrieved.entities
    
    def test_update_document(self, temp_db):
        """Test updating an existing document."""
        doc = Document(
            id="test_001",
            name="original.pdf",
            text="Original content"
        )
        temp_db.save_document(doc)
        
        # Update
        doc.name = "updated.pdf"
        doc.text = "Updated content"
        temp_db.save_document(doc)
        
        retrieved = temp_db.get_document("test_001")
        assert retrieved.name == "updated.pdf"
        assert retrieved.text == "Updated content"
    
    def test_delete_document(self, temp_db):
        """Test deleting a document."""
        doc = Document(id="test_001", name="test.pdf", text="content")
        temp_db.save_document(doc)
        
        assert temp_db.get_document("test_001") is not None
        
        result = temp_db.delete_document("test_001")
        assert result is True
        
        assert temp_db.get_document("test_001") is None
    
    def test_delete_nonexistent_document(self, temp_db):
        """Test deleting a document that doesn't exist."""
        result = temp_db.delete_document("nonexistent")
        assert result is False
    
    def test_get_all_documents(self, temp_db):
        """Test retrieving all documents."""
        for i in range(5):
            doc = Document(id=f"doc_{i}", name=f"doc_{i}.pdf", text=f"Content {i}")
            temp_db.save_document(doc)
        
        docs = temp_db.get_all_documents()
        assert len(docs) == 5
    
    def test_get_all_documents_pagination(self, temp_db):
        """Test pagination of document retrieval."""
        for i in range(10):
            doc = Document(id=f"doc_{i}", name=f"doc_{i}.pdf", text=f"Content {i}")
            temp_db.save_document(doc)
        
        page1 = temp_db.get_all_documents(limit=5, offset=0)
        page2 = temp_db.get_all_documents(limit=5, offset=5)
        
        assert len(page1) == 5
        assert len(page2) == 5
        
        # Ensure no overlap
        page1_ids = {d.id for d in page1}
        page2_ids = {d.id for d in page2}
        assert len(page1_ids & page2_ids) == 0
    
    def test_get_document_count(self, temp_db):
        """Test document count."""
        assert temp_db.get_document_count() == 0
        
        for i in range(3):
            doc = Document(id=f"doc_{i}", name=f"doc_{i}.pdf", text="content")
            temp_db.save_document(doc)
        
        assert temp_db.get_document_count() == 3
    
    def test_embedding_serialization(self, temp_db):
        """Test that embeddings are correctly serialized and deserialized."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        doc = Document(
            id="test_embed",
            name="test.pdf",
            text="content",
            embedding=embedding
        )
        temp_db.save_document(doc)
        
        retrieved = temp_db.get_document("test_embed")
        
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 5
        assert abs(retrieved.embedding[0] - 0.1) < 0.0001
    
    def test_entity_stats(self, temp_db):
        """Test entity statistics."""
        doc1 = Document(
            id="doc_1",
            name="doc1.pdf",
            text="content",
            entities={
                "PERSON": [{"text": "John"}, {"text": "Jane"}],
                "LOCATION": [{"text": "NYC"}]
            }
        )
        doc2 = Document(
            id="doc_2",
            name="doc2.pdf",
            text="content",
            entities={
                "PERSON": [{"text": "Bob"}],
                "MONEY": [{"text": "$100"}]
            }
        )
        
        temp_db.save_document(doc1)
        temp_db.save_document(doc2)
        
        stats = temp_db.get_entity_stats()
        
        assert stats["PERSON"] == 3
        assert stats["LOCATION"] == 1
        assert stats["MONEY"] == 1


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_to_dict(self):
        """Test Document to dictionary conversion."""
        doc = Document(
            id="test_001",
            name="test.pdf",
            text="content",
            entities={"PERSON": [{"text": "John"}]},
            ocr_confidence=0.9
        )
        
        d = doc.to_dict()
        
        assert d["id"] == "test_001"
        assert d["name"] == "test.pdf"
        assert d["text"] == "content"
        assert "PERSON" in d["entities"]
    
    def test_from_dict(self):
        """Test Document from dictionary creation."""
        data = {
            "id": "test_001",
            "name": "test.pdf",
            "text": "content",
            "entities": {"PERSON": [{"text": "John"}]},
            "ocr_confidence": 0.9
        }
        
        doc = Document.from_dict(data)
        
        assert doc.id == "test_001"
        assert doc.name == "test.pdf"
        assert doc.ocr_confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
