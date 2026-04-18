"""
DeedLens FastAPI Application
REST API for Property Document Intelligence.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.logger import setup_logging, get_logger
from core.service import get_document_service, DocumentService

setup_logging(level=settings.log_level)
logger = get_logger(__name__)


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
    doc_type: Optional[str] = None
    ocr_confidence: float = 0.0
    created_at: Optional[str] = None


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
    semantic_score: float = 0.0
    keyword_score: float = 0.0


class ReportResponse(BaseModel):
    title: str
    summary: str
    sections: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    documents_count: int


class StatsResponse(BaseModel):
    total_documents: int
    entity_counts: Dict[str, int]


# Job Queue Models
class JobResponse(BaseModel):
    """Response for job status."""
    job_id: str
    status: str
    progress: int
    current_step: Optional[str] = None
    document_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobCreateResponse(BaseModel):
    """Response when creating a new job."""
    job_id: str
    status: str
    message: str


class JobMetricsResponse(BaseModel):
    """Response for job metrics."""
    total_jobs: int
    pending: int
    processing: int
    completed: int
    failed: int
    avg_duration_ms: Optional[float] = None


class StageMetrics(BaseModel):
    """Metrics for a single processing stage."""
    stage: str
    total_count: int
    success_count: int
    failure_count: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    avg_queue_wait_ms: Optional[float] = None
    failure_rate: float


class ProcessingMetricsResponse(BaseModel):
    """Response for /metrics endpoint - THE KILLER FEATURE."""
    time_window: str  # e.g., "24h"
    total_processed: int
    total_failed: int
    overall_success_rate: float
    avg_total_duration_ms: float
    throughput_per_hour: float
    stages: List[StageMetrics]
    bottleneck_stage: Optional[str] = None  # Slowest stage


class DailyThroughputResponse(BaseModel):
    """Daily throughput stats."""
    date: str
    completed_jobs: int
    failed_jobs: int
    avg_duration_ms: Optional[float] = None


# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Verify API key if configured."""
    if not settings.api_key:
        # No API key configured, allow all requests
        return True
    
    if api_key and api_key == settings.api_key:
        return True
    
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key"
    )


# Initialize FastAPI
app = FastAPI(
    title="DeedLens API",
    description="AI-Powered Property Document Intelligence & Semantic Search",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - use configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# Initialize database schema on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        from core.db_init import init_postgres
        init_postgres()
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")


def get_service() -> DocumentService:
    """Dependency to get document service."""
    return get_document_service()


@app.get("/", response_model=HealthResponse)
async def health_check(service: DocumentService = Depends(get_service)):
    """Health check endpoint."""
    stats = service.get_stats()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now().isoformat(),
        documents_count=stats['total_documents']
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """Get system statistics."""
    stats = service.get_stats()
    return StatsResponse(**stats)


@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """
    Upload and process a property document (synchronous).
    
    Supports PDF and image files (PNG, JPG, TIFF).
    For async processing, use POST /jobs/upload instead.
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
        # Process document using service
        result = service.process_document(tmp_path, file.filename)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        doc = result.document
        return DocumentResponse(
            id=doc.id,
            name=doc.name,
            text=doc.text,
            entities=doc.entities,
            doc_type=doc.doc_type,
            ocr_confidence=doc.ocr_confidence,
            created_at=doc.created_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# JOB QUEUE ENDPOINTS (Async Processing)
# ============================================================================

@app.post("/jobs/upload", response_model=JobCreateResponse)
async def upload_document_async(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """
    Upload a document for async processing.
    
    Returns immediately with a job_id. Poll GET /jobs/{job_id} for status.
    When job completes, document_id will be available in the response.
    """
    from core.jobs import get_job_manager
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    # Ensure upload directory exists
    upload_dir = settings.data_dir / "uploads" if settings.data_dir else Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file permanently (needed by worker later)
    import uuid
    file_id = str(uuid.uuid4())
    permanent_path = upload_dir / f"{file_id}{file_ext}"
    
    content = await file.read()
    with open(permanent_path, "wb") as f:
        f.write(content)
    
    # Create job
    job_manager = get_job_manager()
    job = job_manager.create_job(
        original_filename=file.filename,
        file_path=str(permanent_path)
    )
    
    # Queue the task
    try:
        from worker.tasks import process_document_task
        process_document_task.delay(job.id)
        logger.info(f"Queued job {job.id} for processing")
    except Exception as e:
        # Celery not available - log warning but still return job
        logger.warning(f"Could not queue job (Celery unavailable?): {e}")
        # Mark as pending for manual processing
    
    return JobCreateResponse(
        job_id=job.id,
        status=job.status.value,
        message="Document queued for processing. Poll GET /jobs/{job_id} for status."
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Get the status of a processing job."""
    from core.jobs import get_job_manager
    
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        current_step=job.current_step.value if job.current_step else None,
        document_id=job.document_id,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    _: bool = Depends(verify_api_key)
):
    """List all jobs, optionally filtered by status."""
    from core.jobs import get_job_manager, JobStatus
    
    job_manager = get_job_manager()
    
    # Get jobs by status or all
    if status:
        try:
            job_status = JobStatus(status)
            jobs = job_manager.get_jobs_by_status(job_status, limit=limit)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid: {[s.value for s in JobStatus]}"
            )
    else:
        jobs = job_manager.get_all_jobs(limit=limit)
    
    return [
        JobResponse(
            job_id=job.id,
            status=job.status.value,
            progress=job.progress,
            current_step=job.current_step.value if job.current_step else None,
            document_id=job.document_id,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
        for job in jobs
    ]


@app.get("/jobs/metrics/summary", response_model=JobMetricsResponse)
async def get_job_metrics(
    _: bool = Depends(verify_api_key)
):
    """Get job processing metrics."""
    from core.jobs import get_job_manager
    
    job_manager = get_job_manager()
    metrics = job_manager.get_metrics_summary()
    
    return JobMetricsResponse(**metrics)


@app.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Cancel a pending or processing job."""
    from core.jobs import get_job_manager, JobStatus
    
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}"
        )
    
    # TODO: Revoke Celery task if possible
    job_manager.update_job(job_id, status=JobStatus.FAILED.value, error_message="Cancelled by user")
    
    return {"status": "cancelled", "job_id": job_id}


# =============================================================================
# OBSERVABILITY ENDPOINTS - THE KILLER FEATURE
# =============================================================================

@app.get("/metrics", response_model=ProcessingMetricsResponse)
async def get_processing_metrics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    _: bool = Depends(verify_api_key)
):
    """
    Get detailed processing metrics for observability.
    
    This is the killer feature that differentiates DeedLens from typical ML demos.
    
    Returns:
    - Per-stage latency (avg, p50, p95, p99)
    - Failure rates per stage
    - Queue wait times
    - Throughput metrics
    - Bottleneck identification
    """
    from core.database import get_db
    
    db = get_db()
    
    try:
        # Query processing_logs for stage metrics
        stage_query = """
            SELECT 
                stage,
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE status = 'completed') as success_count,
                COUNT(*) FILTER (WHERE status = 'failed') as failure_count,
                COALESCE(AVG(duration_ms) FILTER (WHERE status = 'completed'), 0) as avg_duration_ms,
                COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) FILTER (WHERE status = 'completed'), 0) as p50_duration_ms,
                COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) FILTER (WHERE status = 'completed'), 0) as p95_duration_ms,
                COALESCE(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) FILTER (WHERE status = 'completed'), 0) as p99_duration_ms,
                COALESCE(AVG(queue_wait_ms), 0) as avg_queue_wait_ms
            FROM processing_logs
            WHERE created_at > NOW() - INTERVAL '%s hours'
            GROUP BY stage
            ORDER BY 
                CASE stage 
                    WHEN 'ocr' THEN 1 
                    WHEN 'clean' THEN 2 
                    WHEN 'ner' THEN 3 
                    WHEN 'embed' THEN 4 
                    WHEN 'store' THEN 5 
                    ELSE 6 
                END
        """
        
        stage_results = db.fetch_all(stage_query, (hours,))
        
        stages = []
        bottleneck_stage = None
        max_duration = 0
        
        for row in stage_results:
            total = row['total_count']
            failures = row['failure_count']
            failure_rate = (failures / total * 100) if total > 0 else 0
            
            stage_metric = StageMetrics(
                stage=row['stage'],
                total_count=total,
                success_count=row['success_count'],
                failure_count=failures,
                avg_duration_ms=round(row['avg_duration_ms'], 2),
                p50_duration_ms=round(row['p50_duration_ms'], 2),
                p95_duration_ms=round(row['p95_duration_ms'], 2),
                p99_duration_ms=round(row['p99_duration_ms'], 2),
                avg_queue_wait_ms=round(row['avg_queue_wait_ms'], 2) if row['avg_queue_wait_ms'] else None,
                failure_rate=round(failure_rate, 2)
            )
            stages.append(stage_metric)
            
            # Track bottleneck
            if row['avg_duration_ms'] > max_duration:
                max_duration = row['avg_duration_ms']
                bottleneck_stage = row['stage']
        
        # Get overall job metrics
        job_query = """
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                COUNT(*) FILTER (WHERE status = 'failed') as failed,
                COALESCE(AVG(duration_ms) FILTER (WHERE status = 'completed'), 0) as avg_duration_ms
            FROM jobs
            WHERE created_at > NOW() - INTERVAL '%s hours'
        """
        
        job_result = db.fetch_one(job_query, (hours,))
        
        total_processed = job_result['completed'] if job_result else 0
        total_failed = job_result['failed'] if job_result else 0
        total = total_processed + total_failed
        
        success_rate = (total_processed / total * 100) if total > 0 else 0
        throughput = total_processed / hours if hours > 0 else 0
        
        return ProcessingMetricsResponse(
            time_window=f"{hours}h",
            total_processed=total_processed,
            total_failed=total_failed,
            overall_success_rate=round(success_rate, 2),
            avg_total_duration_ms=round(job_result['avg_duration_ms'], 2) if job_result else 0,
            throughput_per_hour=round(throughput, 2),
            stages=stages,
            bottleneck_stage=bottleneck_stage
        )
        
    except Exception as e:
        logger.warning(f"PostgreSQL metrics not available, using fallback: {e}")
        
        # Fallback for SQLite or when PostgreSQL isn't configured
        return ProcessingMetricsResponse(
            time_window=f"{hours}h",
            total_processed=0,
            total_failed=0,
            overall_success_rate=0,
            avg_total_duration_ms=0,
            throughput_per_hour=0,
            stages=[],
            bottleneck_stage=None
        )


@app.get("/metrics/daily", response_model=List[DailyThroughputResponse])
async def get_daily_throughput(
    days: int = Query(30, ge=1, le=90, description="Number of days"),
    _: bool = Depends(verify_api_key)
):
    """Get daily throughput for trend analysis."""
    from core.database import get_db
    
    db = get_db()
    
    try:
        query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
                COALESCE(AVG(duration_ms) FILTER (WHERE status = 'completed'), 0) as avg_duration_ms
            FROM jobs
            WHERE created_at > NOW() - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """
        
        results = db.fetch_all(query, (days,))
        
        return [
            DailyThroughputResponse(
                date=str(row['date']),
                completed_jobs=row['completed_jobs'],
                failed_jobs=row['failed_jobs'],
                avg_duration_ms=round(row['avg_duration_ms'], 2) if row['avg_duration_ms'] else None
            )
            for row in results
        ]
        
    except Exception as e:
        logger.warning(f"Daily throughput query failed: {e}")
        return []


@app.get("/metrics/stage/{stage_name}")
async def get_stage_details(
    stage_name: str,
    hours: int = Query(24, ge=1, le=168),
    _: bool = Depends(verify_api_key)
):
    """Get detailed metrics for a specific processing stage."""
    from core.database import get_db
    
    valid_stages = ['ocr', 'clean', 'ner', 'embed', 'store']
    if stage_name not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Valid stages: {valid_stages}"
        )
    
    db = get_db()
    
    try:
        # Get recent logs for this stage
        query = """
            SELECT 
                job_id,
                started_at,
                completed_at,
                duration_ms,
                status,
                error_type,
                error_message,
                metrics
            FROM processing_logs
            WHERE stage = %s
              AND created_at > NOW() - INTERVAL '%s hours'
            ORDER BY created_at DESC
            LIMIT 100
        """
        
        results = db.fetch_all(query, (stage_name, hours))
        
        # Calculate distribution
        durations = [r['duration_ms'] for r in results if r['duration_ms'] and r['status'] == 'completed']
        
        histogram = {}
        for d in durations:
            bucket = (d // 100) * 100  # 100ms buckets
            histogram[f"{bucket}-{bucket+100}ms"] = histogram.get(f"{bucket}-{bucket+100}ms", 0) + 1
        
        errors = [
            {"type": r['error_type'], "message": r['error_message'][:200] if r['error_message'] else None}
            for r in results if r['status'] == 'failed'
        ][:10]  # Last 10 errors
        
        return {
            "stage": stage_name,
            "time_window": f"{hours}h",
            "total_executions": len(results),
            "duration_histogram": histogram,
            "recent_errors": errors
        }
        
    except Exception as e:
        logger.warning(f"Stage details query failed: {e}")
        return {"stage": stage_name, "error": str(e)}


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """List all processed documents."""
    documents = service.get_all_documents(limit=limit, offset=offset)
    return [
        DocumentResponse(
            id=doc.id,
            name=doc.name,
            text=doc.text,
            entities=doc.entities,
            doc_type=doc.doc_type,
            ocr_confidence=doc.ocr_confidence,
            created_at=doc.created_at
        )
        for doc in documents
    ]


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """Get a specific document by ID."""
    doc = service.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=doc.id,
        name=doc.name,
        text=doc.text,
        entities=doc.entities,
        doc_type=doc.doc_type,
        ocr_confidence=doc.ocr_confidence,
        created_at=doc.created_at
    )


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """Delete a document."""
    success = service.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"status": "deleted", "id": doc_id}


@app.post("/search", response_model=List[SearchResultResponse])
async def search_documents(
    request: SearchRequest,
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """
    Search documents using semantic, keyword, or hybrid search.
    
    Modes:
    - semantic: Vector similarity search using FAISS
    - keyword: Traditional keyword matching
    - hybrid: Combination of both (default)
    """
    results = service.search(
        query=request.query,
        mode=request.mode,
        k=request.k,
        filters=request.filters
    )
    
    return [
        SearchResultResponse(
            doc_id=r.doc_id,
            title=r.title,
            preview=r.preview,
            score=r.score,
            semantic_score=r.semantic_score,
            keyword_score=r.keyword_score
        )
        for r in results
    ]


@app.get("/search")
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", description="Search mode: semantic, keyword, or hybrid"),
    k: int = Query(10, description="Number of results"),
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """Search documents (GET endpoint)."""
    request = SearchRequest(query=q, mode=mode, k=k)
    return await search_documents(request, service, _)


@app.get("/reports/{doc_id}", response_model=ReportResponse)
async def generate_report(
    doc_id: str,
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """Generate an AI-powered property report for a document."""
    doc = service.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    result = service.generate_report(doc_id)
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
    
    return ReportResponse(
        title=result.get('title', 'Property Report'),
        summary=result.get('summary', ''),
        sections=result.get('sections', {})
    )


@app.get("/entities/{doc_id}", response_model=Dict[str, List[EntityResponse]])
async def get_entities(
    doc_id: str,
    service: DocumentService = Depends(get_service),
    _: bool = Depends(verify_api_key)
):
    """Get extracted entities for a document."""
    doc = service.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    result = {}
    for entity_type, entity_list in doc.entities.items():
        result[entity_type] = [
            EntityResponse(
                text=e.get('text', str(e)) if isinstance(e, dict) else str(e),
                entity_type=entity_type,
                confidence=e.get('confidence', 0.9) if isinstance(e, dict) else 0.9
            )
            for e in entity_list
        ]
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000))
    )
