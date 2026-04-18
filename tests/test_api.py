"""
Tests for FastAPI endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client."""
    from api.app import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestStatsEndpoint:
    """Tests for stats endpoint."""
    
    def test_get_stats(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "entity_counts" in data


class TestDocumentsEndpoint:
    """Tests for documents endpoints."""
    
    def test_list_documents_empty(self, client):
        """Test listing documents when empty."""
        response = client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_nonexistent_document(self, client):
        """Test getting a document that doesn't exist."""
        response = client.get("/documents/nonexistent_id")
        
        assert response.status_code == 404
    
    def test_delete_nonexistent_document(self, client):
        """Test deleting a document that doesn't exist."""
        response = client.delete("/documents/nonexistent_id")
        
        assert response.status_code == 404


class TestSearchEndpoint:
    """Tests for search endpoints."""
    
    def test_search_post(self, client):
        """Test search via POST."""
        response = client.post("/search", json={
            "query": "test query",
            "mode": "hybrid",
            "k": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_get(self, client):
        """Test search via GET."""
        response = client.get("/search?q=test&mode=keyword&k=5")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_missing_query(self, client):
        """Test search without query parameter."""
        response = client.get("/search")
        
        assert response.status_code == 422  # Validation error


class TestUploadEndpoint:
    """Tests for upload endpoint."""
    
    def test_upload_invalid_file_type(self, client):
        """Test uploading unsupported file type."""
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
