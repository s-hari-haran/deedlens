"""
DeedLens FastAPI Application
REST API for Property Document Intelligence.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Pydantic Models
class EntityResponse(BaseModel):
    text: str
    entity_type: str
    confidence: float


class DocumentResponse(BaseModel):
    id: str
    name: str
    text: str
    entities: Dict[str, List[dict]]
    processed_at: str


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    k: int = 10
    filters: Optional[Dict] = None


class SearchResultResponse(BaseModel):
    doc_id: str
    title: str
    preview: str
    score: float


class ReportResponse(BaseModel):
    title: str
    summary: str
    sections: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# Initialize FastAPI
app = FastAPI(
    title="DeedLens API",
    description="AI-Powered Property Document Intelligence & Semantic Search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
documents_store: Dict[str, dict] = {}
embeddings_store: Dict[str, list] = {}


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a property document.
    
    Supports PDF and image files (PNG, JPG, TIFF).
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process document
        result = await process_document(tmp_path, file.filename)
        
        # Store in memory
        documents_store[result['id']] = result
        
        return DocumentResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def process_document(file_path: str, file_name: str) -> dict:
    """Process an uploaded document."""
    doc_id = str(uuid.uuid4())[:8]
    
    try:
        from ocr.ocr_engine import extract_text_from_document
        from preprocessing.text_cleaner import clean_ocr_text
        from nlp.ner_model import extract_entities
        from nlp.embeddings import generate_embeddings
        
        # OCR
        ocr_result = extract_text_from_document(file_path)
        raw_text = ocr_result['text']
        
        # Clean
        cleaned = clean_ocr_text(raw_text)
        cleaned_text = cleaned['cleaned_text']
        
        # NER
        entities = extract_entities(cleaned_text)
        
        # Embeddings
        embeddings = generate_embeddings(cleaned_text)
        embeddings_store[doc_id] = embeddings.get('embedding', [])
        
        return {
            'id': doc_id,
            'name': file_name,
            'text': cleaned_text,
            'entities': entities.get('entities', {}),
            'processed_at': datetime.now().isoformat()
        }
    
    except ImportError:
        # Fallback for testing without all dependencies
        return {
            'id': doc_id,
            'name': file_name,
            'text': "Sample extracted text (modules not available)",
            'entities': {},
            'processed_at': datetime.now().isoformat()
        }


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    """List all processed documents."""
    return [DocumentResponse(**doc) for doc in documents_store.values()]


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(**documents_store[doc_id])


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del documents_store[doc_id]
    if doc_id in embeddings_store:
        del embeddings_store[doc_id]
    
    return {"status": "deleted", "id": doc_id}


@app.post("/search", response_model=List[SearchResultResponse])
async def search_documents(request: SearchRequest):
    """
    Search documents using semantic, keyword, or hybrid search.
    
    Modes:
    - semantic: Vector similarity search
    - keyword: Traditional keyword matching
    - hybrid: Combination of both (default)
    """
    if not documents_store:
        return []
    
    results = []
    query_lower = request.query.lower()
    
    # Simple search implementation
    for doc_id, doc in documents_store.items():
        text_lower = doc.get('text', '').lower()
        name_lower = doc.get('name', '').lower()
        
        # Calculate simple relevance score
        score = 0.0
        
        if query_lower in text_lower:
            score += 0.5
        if query_lower in name_lower:
            score += 0.3
        
        # Check entities
        for entity_type, entities in doc.get('entities', {}).items():
            for entity in entities:
                entity_text = entity.get('text', '').lower() if isinstance(entity, dict) else str(entity).lower()
                if query_lower in entity_text:
                    score += 0.2
        
        if score > 0:
            results.append(SearchResultResponse(
                doc_id=doc_id,
                title=doc['name'],
                preview=doc.get('text', '')[:200],
                score=min(score, 1.0)
            ))
    
    # Sort by score
    results.sort(key=lambda x: -x.score)
    
    return results[:request.k]


@app.get("/search")
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", description="Search mode: semantic, keyword, or hybrid"),
    k: int = Query(10, description="Number of results")
):
    """Search documents (GET endpoint)."""
    request = SearchRequest(query=q, mode=mode, k=k)
    return await search_documents(request)


@app.get("/reports/{doc_id}", response_model=ReportResponse)
async def generate_report(doc_id: str):
    """Generate an AI-powered property report for a document."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_store[doc_id]
    
    try:
        from reports.report_generator import generate_report as gen_report
        
        result = gen_report(
            doc.get('text', ''),
            doc.get('entities', {})
        )
        
        return ReportResponse(
            title=result.get('title', 'Property Report'),
            summary=result.get('summary', ''),
            sections=result.get('sections', {})
        )
    
    except ImportError:
        # Fallback
        entities = doc.get('entities', {})
        
        return ReportResponse(
            title="Property Transaction Report",
            summary="Report generated from extracted entities.",
            sections={
                "Property Summary": f"Document: {doc['name']}",
                "Parties": str(entities.get('PERSON', [])),
                "Location": str(entities.get('LOCATION', [])),
                "Value": str(entities.get('MONEY', []))
            }
        )


@app.get("/entities/{doc_id}", response_model=Dict[str, List[EntityResponse]])
async def get_entities(doc_id: str):
    """Get extracted entities for a document."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_store[doc_id]
    entities = doc.get('entities', {})
    
    result = {}
    for entity_type, entity_list in entities.items():
        result[entity_type] = [
            EntityResponse(
                text=e.get('text', str(e)) if isinstance(e, dict) else str(e),
                entity_type=entity_type,
                confidence=e.get('confidence', 0.9) if isinstance(e, dict) else 0.9
            )
            for e in entity_list
        ]
    
    return result


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    total_entities = 0
    entity_counts = {}
    
    for doc in documents_store.values():
        for entity_type, entities in doc.get('entities', {}).items():
            count = len(entities)
            total_entities += count
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + count
    
    return {
        "total_documents": len(documents_store),
        "total_entities": total_entities,
        "entity_counts": entity_counts
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
