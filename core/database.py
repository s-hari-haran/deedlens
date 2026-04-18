"""
DeedLens Database Layer
SQLite persistence for documents and metadata.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

from config import settings


@dataclass
class Document:
    """Represents a processed document."""
    id: str
    name: str
    file_path: Optional[str] = None
    text: str = ""
    entities: Dict[str, List[dict]] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    ocr_confidence: float = 0.0
    doc_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "file_path": self.file_path,
            "text": self.text,
            "entities": self.entities,
            "embedding": self.embedding,
            "ocr_confidence": self.ocr_confidence,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create Document from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            file_path=data.get("file_path"),
            text=data.get("text", ""),
            entities=data.get("entities", {}),
            embedding=data.get("embedding"),
            ocr_confidence=data.get("ocr_confidence", 0.0),
            doc_type=data.get("doc_type"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )


class Database:
    """
    SQLite database for document storage.
    Thread-safe with connection per operation.
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        file_path TEXT,
        text TEXT,
        entities TEXT,
        embedding BLOB,
        ocr_confidence REAL DEFAULT 0.0,
        doc_type TEXT,
        metadata TEXT,
        created_at TEXT,
        updated_at TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(name);
    CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
    CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
    
    -- Jobs table for async processing
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        document_id TEXT,
        file_path TEXT,
        original_filename TEXT,
        status TEXT DEFAULT 'pending',
        current_step TEXT,
        progress INTEGER DEFAULT 0,
        error_message TEXT,
        retry_count INTEGER DEFAULT 0,
        created_at TEXT,
        started_at TEXT,
        completed_at TEXT,
        updated_at TEXT,
        -- Metrics
        ocr_duration_ms INTEGER,
        clean_duration_ms INTEGER,
        ner_duration_ms INTEGER,
        embed_duration_ms INTEGER,
        total_duration_ms INTEGER,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
    
    CREATE TABLE IF NOT EXISTS search_index (
        doc_id TEXT PRIMARY KEY,
        indexed_at TEXT,
        embedding_model TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        id,
        name,
        text,
        content='documents',
        content_rowid='rowid'
    );
    
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, id, name, text) 
        VALUES (new.rowid, new.id, new.name, new.text);
    END;
    
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, id, name, text) 
        VALUES('delete', old.rowid, old.id, old.name, old.text);
    END;
    
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, id, name, text) 
        VALUES('delete', old.rowid, old.id, old.name, old.text);
        INSERT INTO documents_fts(rowid, id, name, text) 
        VALUES (new.rowid, new.id, new.name, new.text);
    END;
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _serialize_embedding(self, embedding: Optional[List[float]]) -> Optional[bytes]:
        """Serialize embedding to bytes."""
        if embedding is None:
            return None
        import struct
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    def _deserialize_embedding(self, data: Optional[bytes]) -> Optional[List[float]]:
        """Deserialize embedding from bytes."""
        if data is None:
            return None
        import struct
        n = len(data) // 4
        return list(struct.unpack(f'{n}f', data))
    
    def save_document(self, doc: Document) -> Document:
        """
        Save or update a document.
        
        Args:
            doc: Document to save
            
        Returns:
            Saved document with timestamps
        """
        now = datetime.utcnow().isoformat()
        
        if doc.created_at is None:
            doc.created_at = now
        doc.updated_at = now
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, name, file_path, text, entities, embedding, 
                 ocr_confidence, doc_type, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id,
                doc.name,
                doc.file_path,
                doc.text,
                json.dumps(doc.entities),
                self._serialize_embedding(doc.embedding),
                doc.ocr_confidence,
                doc.doc_type,
                json.dumps(doc.metadata),
                doc.created_at,
                doc.updated_at
            ))
        
        return doc
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (doc_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return self._row_to_document(row)
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Get all documents with pagination."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            
            return [self._row_to_document(row) for row in rows]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ?",
                (doc_id,)
            )
            return cursor.rowcount > 0
    
    def search_text(self, query: str, limit: int = 10) -> List[Document]:
        """
        Full-text search using FTS5.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of matching documents
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT d.* FROM documents d
                JOIN documents_fts fts ON d.id = fts.id
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            
            return [self._row_to_document(row) for row in rows]
    
    def get_documents_with_embeddings(self) -> List[tuple]:
        """Get all documents that have embeddings."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, embedding FROM documents 
                WHERE embedding IS NOT NULL
            """).fetchall()
            
            return [
                (row['id'], self._deserialize_embedding(row['embedding']))
                for row in rows
            ]
    
    def get_document_count(self) -> int:
        """Get total document count."""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return result[0]
    
    def get_entity_stats(self) -> Dict[str, int]:
        """Get entity type counts across all documents."""
        docs = self.get_all_documents(limit=10000)
        stats = {}
        
        for doc in docs:
            for entity_type, entities in doc.entities.items():
                stats[entity_type] = stats.get(entity_type, 0) + len(entities)
        
        return stats
    
    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a database row to Document."""
        return Document(
            id=row['id'],
            name=row['name'],
            file_path=row['file_path'],
            text=row['text'] or "",
            entities=json.loads(row['entities']) if row['entities'] else {},
            embedding=self._deserialize_embedding(row['embedding']),
            ocr_confidence=row['ocr_confidence'] or 0.0,
            doc_type=row['doc_type'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )


# Global database instance
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Get the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


if __name__ == "__main__":
    # Test database
    db = get_db()
    
    # Create test document
    doc = Document(
        id="test_001",
        name="test_document.pdf",
        text="This is a test sale deed for property in Bangalore.",
        entities={
            "LOCATION": [{"text": "Bangalore"}],
            "MONEY": [{"text": "Rs. 50,00,000"}]
        },
        ocr_confidence=0.95
    )
    
    # Save
    saved = db.save_document(doc)
    print(f"Saved: {saved.id} at {saved.created_at}")
    
    # Retrieve
    retrieved = db.get_document("test_001")
    print(f"Retrieved: {retrieved.name}")
    
    # Count
    print(f"Total documents: {db.get_document_count()}")
    
    # Cleanup test
    db.delete_document("test_001")
    print("Test document deleted")
