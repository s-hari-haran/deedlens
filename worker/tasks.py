"""
Celery Tasks for Document Processing Pipeline

Architecture:
    - Each stage is a separate task for granular control
    - Tasks are chained: OCR → Clean → NER → Embed → Store
    - Per-stage metrics logged to processing_logs
    - Exponential backoff on failures
    - Partial failure recovery

Usage:
    from celery import chain
    from worker.tasks import ocr_task, clean_task, ner_task, embed_task, store_task
    
    pipeline = chain(
        ocr_task.s(job_id),
        clean_task.s(),
        ner_task.s(),
        embed_task.s(),
        store_task.s()
    )
    pipeline.apply_async()
"""

import time
import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps

from .celery_app import celery_app


# =============================================================================
# OBSERVABILITY DECORATOR
# =============================================================================

def log_stage(stage_name: str):
    """
    Decorator that logs stage metrics to processing_logs.
    
    Tracks:
    - Duration
    - Status (completed/failed)
    - Error details
    - Custom metrics from return value
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, data: Dict[str, Any], *args, **kwargs):
            from core.logger import get_logger
            logger = get_logger(f"worker.{stage_name}")
            
            job_id = data.get('job_id')
            started_at = datetime.utcnow()
            start_time = time.time()
            
            logger.info(f"[{job_id}] Starting {stage_name}")
            
            # Log stage start
            log_id = _log_stage_start(job_id, stage_name, started_at)
            
            try:
                # Execute the stage
                result = func(self, data, *args, **kwargs)
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Log stage completion
                _log_stage_complete(
                    log_id=log_id,
                    duration_ms=duration_ms,
                    metrics=result.get('_metrics', {})
                )
                
                logger.info(f"[{job_id}] {stage_name} completed in {duration_ms}ms")
                
                # Update job progress
                _update_job_progress(job_id, stage_name, result)
                
                return result
                
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_msg = str(e)
                stack = traceback.format_exc()
                
                # Log stage failure
                _log_stage_failure(
                    log_id=log_id,
                    duration_ms=duration_ms,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    stack_trace=stack
                )
                
                logger.error(f"[{job_id}] {stage_name} failed: {error_msg}")
                
                # Re-raise for Celery retry handling
                raise
        
        return wrapper
    return decorator


def _log_stage_start(job_id: str, stage: str, started_at: datetime) -> str:
    """Log stage start to processing_logs (PostgreSQL)."""
    try:
        from config import settings
        import psycopg2

        log_id = str(uuid.uuid4())

        # Only log if using PostgreSQL
        if not settings.database_url or not settings.database_url.startswith('postgresql'):
            return log_id

        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO processing_logs (id, job_id, stage, started_at, status)
            VALUES (%s, %s, %s, %s, 'started')
        """, (log_id, job_id, stage, started_at))
        conn.commit()
        cur.close()
        conn.close()

        return log_id
    except Exception:
        # Don't fail the task if logging fails
        return str(uuid.uuid4())


def _log_stage_complete(log_id: str, duration_ms: int, metrics: Dict = None):
    """Log stage completion."""
    try:
        from config import settings
        import psycopg2
        import json

        # Only log if using PostgreSQL
        if not settings.database_url or not settings.database_url.startswith('postgresql'):
            return

        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        cur.execute("""
            UPDATE processing_logs
            SET completed_at = NOW(),
                duration_ms = %s,
                status = 'completed',
                metrics = %s
            WHERE id = %s
        """, (duration_ms, json.dumps(metrics or {}), log_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def _log_stage_failure(
    log_id: str,
    duration_ms: int,
    error_type: str,
    error_message: str,
    stack_trace: str
):
    """Log stage failure."""
    try:
        from config import settings
        import psycopg2

        # Only log if using PostgreSQL
        if not settings.database_url or not settings.database_url.startswith('postgresql'):
            return

        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        cur.execute("""
            UPDATE processing_logs
            SET completed_at = NOW(),
                duration_ms = %s,
                status = 'failed',
                error_type = %s,
                error_message = %s,
                stack_trace = %s
            WHERE id = %s
        """, (duration_ms, error_type, error_message, stack_trace, log_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def _update_job_progress(job_id: str, stage: str, result: Dict):
    """Update job progress based on completed stage."""
    stage_progress = {
        'ocr': 20,
        'clean': 35,
        'ner': 55,
        'embed': 75,
        'store': 100
    }
    
    progress = stage_progress.get(stage, 0)
    
    try:
        from core.jobs import get_job_manager
        job_manager = get_job_manager()
        
        updates = {
            'progress': progress,
            'current_step': stage
        }
        
        # Add stage-specific metrics
        if stage == 'ocr':
            updates['ocr_duration_ms'] = result.get('_metrics', {}).get('duration_ms')
        elif stage == 'ner':
            updates['ner_duration_ms'] = result.get('_metrics', {}).get('duration_ms')
        elif stage == 'embed':
            updates['embed_duration_ms'] = result.get('_metrics', {}).get('duration_ms')
        
        job_manager.update_job(job_id, **updates)
    except Exception:
        pass


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

RETRY_CONFIG = {
    'autoretry_for': (Exception,),
    'retry_backoff': True,  # Exponential backoff
    'retry_backoff_max': 600,  # Max 10 minutes between retries
    'retry_jitter': True,  # Add randomness to prevent thundering herd
    'max_retries': 3,
}


# =============================================================================
# PIPELINE TASKS
# =============================================================================

@celery_app.task(bind=True, name='tasks.ocr', **RETRY_CONFIG)
@log_stage('ocr')
def ocr_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 1: OCR - Extract text from document.
    
    Input:
        {
            'job_id': str,
            'file_path': str,
            'filename': str
        }
    
    Output:
        {
            'job_id': str,
            'file_path': str,
            'filename': str,
            'raw_text': str,
            'ocr_confidence': float,
            'page_count': int,
            '_metrics': {'pages': int, 'confidence': float, 'backend': str}
        }
    """
    from ocr.ocr_engine import OCREngine, OCRBackend
    from config import settings
    
    backend = OCRBackend(settings.ocr_backend)
    engine = OCREngine(backend=backend, languages=settings.ocr_languages)
    result = engine.process_file(data['file_path'])
    
    return {
        **data,
        'raw_text': result.full_text,
        'ocr_confidence': result.avg_confidence,
        'page_count': len(result.pages) if hasattr(result, 'pages') else 1,
        '_metrics': {
            'pages': len(result.pages) if hasattr(result, 'pages') else 1,
            'confidence': result.avg_confidence,
            'backend': settings.ocr_backend,
            'text_length': len(result.full_text)
        }
    }


@celery_app.task(bind=True, name='tasks.clean', **RETRY_CONFIG)
@log_stage('clean')
def clean_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: Clean - Apply OCR post-processing corrections.
    
    Input:
        {..., 'raw_text': str}
    
    Output:
        {..., 'cleaned_text': str, 'corrections_applied': int}
    """
    from preprocessing.text_cleaner import TextCleaner
    
    cleaner = TextCleaner()
    result = cleaner.clean(data['raw_text'])
    
    corrections_count = len(result.corrections) if hasattr(result, 'corrections') else 0
    
    return {
        **data,
        'cleaned_text': result.cleaned,
        'corrections_applied': corrections_count,
        '_metrics': {
            'corrections': corrections_count,
            'input_length': len(data['raw_text']),
            'output_length': len(result.cleaned)
        }
    }


@celery_app.task(bind=True, name='tasks.ner', **RETRY_CONFIG)
@log_stage('ner')
def ner_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 3: NER - Extract named entities.

    Input:
        {..., 'cleaned_text': str}

    Output:
        {..., 'entities': dict}
    """
    from nlp.ner_model import PropertyNERModel

    model = PropertyNERModel()
    result = model.extract(data['cleaned_text'])
    
    # Group by entity type
    grouped = {}
    for entity in result.entities:
        type_name = entity.entity_type.value
        if type_name not in grouped:
            grouped[type_name] = []
        grouped[type_name].append({
            "text": entity.text,
            "start": entity.start,
            "end": entity.end,
            "confidence": entity.confidence,
            "subtype": getattr(entity, 'subtype', None)
        })
    
    total_entities = sum(len(v) for v in grouped.values())
    
    return {
        **data,
        'entities': grouped,
        '_metrics': {
            'total_entities': total_entities,
            'entity_types': list(grouped.keys()),
            'entities_per_type': {k: len(v) for k, v in grouped.items()}
        }
    }


@celery_app.task(bind=True, name='tasks.embed', **RETRY_CONFIG)
@log_stage('embed')
def embed_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 4: Embed - Generate semantic embeddings.

    Input:
        {..., 'cleaned_text': str}

    Output:
        {..., 'embedding': list}
    """
    # Import torch first to ensure nn module is available for transformers
    try:
        import torch
        import torch.nn
        # Patch transformers namespace to ensure nn is available
        import transformers
        if not hasattr(transformers, 'nn'):
            transformers.nn = torch.nn
    except ImportError:
        pass

    from nlp.embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator()
    result = generator.embed_text(data['cleaned_text'])
    embedding_list = result.embedding.tolist()
    
    return {
        **data,
        'embedding': embedding_list,
        '_metrics': {
            'embedding_dim': len(embedding_list),
            'model': 'all-MiniLM-L6-v2'
        }
    }


@celery_app.task(bind=True, name='tasks.store', **RETRY_CONFIG)
@log_stage('store')
def store_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 5: Store - Save document to PostgreSQL database.

    Input:
        {..., 'cleaned_text': str, 'entities': dict, 'embedding': list}

    Output:
        {'job_id': str, 'document_id': str, 'status': 'completed'}
    """
    from config import settings
    import json
    import uuid as uuid_module

    # Try PostgreSQL first (production), fall back to SQLite (dev)
    doc_id = str(uuid_module.uuid4())

    # Determine doc type from entities
    doc_type = None
    if 'DOCUMENT_TYPE' in data['entities'] and data['entities']['DOCUMENT_TYPE']:
        doc_type = data['entities']['DOCUMENT_TYPE'][0].get('text')

    # Try to save to PostgreSQL
    try:
        from core.logger import get_logger
        import psycopg2
        logger = get_logger('store_task')
        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()

        # Insert document
        cur.execute("""
            INSERT INTO documents (
                id, filename, file_path, cleaned_text,
                document_type, metadata, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, (
            doc_id,
            data['filename'],
            data.get('file_path', ''),
            data['cleaned_text'],
            doc_type,
            json.dumps({})
        ))
        logger.info(f"Document inserted: {doc_id}")

        # Insert entities
        entity_count = 0
        for entity_type, entity_list in data['entities'].items():
            for entity in entity_list:
                cur.execute("""
                    INSERT INTO entities (
                        document_id, entity_type, value, confidence,
                        extraction_method, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (
                    doc_id,
                    entity_type,
                    entity.get('text', ''),
                    1.0,
                    'spacy'
                ))
                entity_count += 1
        logger.info(f"Entities inserted: {entity_count} for doc {doc_id}")

        conn.commit()
        logger.info(f"Transaction committed for doc {doc_id}")
        conn.close()

    except Exception as e:
        # Fall back to SQLite
        from core.database import get_db, Document
        from core.logger import get_logger
        logger = get_logger('store_task')
        logger.error(f"PostgreSQL save failed: {e}, falling back to SQLite")

        db = get_db()
        doc = Document(
            id=doc_id,
            name=data['filename'],
            file_path=data.get('file_path', ''),
            text=data['cleaned_text'],
            entities=data['entities'],
            embedding=data['embedding'],
            ocr_confidence=data.get('ocr_confidence', 0.0),
            doc_type=doc_type
        )
        db.save_document(doc)

    # Add to search index
    try:
        from core.service import get_document_service
        from core.logger import get_logger
        service = get_document_service()
        if service._search_engine and data['embedding']:
            service._search_engine.add_document(doc_id, data['embedding'])
    except Exception as e:
        from core.logger import get_logger
        logger = get_logger()
        logger.warning(f"Failed to add to search index: {e}")

    # Mark job completed in PostgreSQL
    from core.jobs import get_job_manager, JobStatus
    job_manager = get_job_manager()
    job_manager.update_job(
        data['job_id'],
        status=JobStatus.COMPLETED.value,
        progress=100,
        document_id=doc_id
    )

    return {
        'job_id': data['job_id'],
        'document_id': doc_id,
        'status': 'completed',
        '_metrics': {
            'document_id': doc_id,
            'text_length': len(data['cleaned_text']),
            'entity_count': sum(len(v) for v in data['entities'].values())
        }
    }


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

def create_pipeline(job_id: str, file_path: str, filename: str):
    """
    Create a chained processing pipeline.
    
    Usage:
        pipeline = create_pipeline(job_id, file_path, filename)
        result = pipeline.apply_async()
    """
    from celery import chain
    
    initial_data = {
        'job_id': job_id,
        'file_path': file_path,
        'filename': filename
    }
    
    return chain(
        ocr_task.s(initial_data),
        clean_task.s(),
        ner_task.s(),
        embed_task.s(),
        store_task.s()
    )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_document_task(self, job_id: str):
    """
    Legacy single-task processor (for backwards compatibility).
    
    For new code, use create_pipeline() instead.
    """
    from core.jobs import get_job_manager, JobStatus
    from core.logger import get_logger
    
    logger = get_logger("worker.tasks")
    job_manager = get_job_manager()
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return {"error": "Job not found"}
    
    logger.info(f"Starting pipeline for job {job_id}")
    
    # Mark as processing
    job_manager.update_job(job_id, status=JobStatus.PROCESSING.value, progress=5)
    
    try:
        # Create and execute pipeline
        pipeline = create_pipeline(job_id, job.file_path, job.original_filename)
        result = pipeline.apply_async()
        
        return {
            "status": "pipeline_started",
            "job_id": job_id,
            "chain_id": str(result.id)
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Pipeline creation failed: {error_msg}")
        
        if self.request.retries < self.max_retries:
            job_manager.update_job(job_id, status=JobStatus.RETRYING.value)
            raise self.retry(exc=e)
        else:
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                error_message=error_msg
            )
            return {"status": "failed", "job_id": job_id, "error": error_msg}


# =============================================================================
# PRIORITY QUEUES
# =============================================================================

@celery_app.task(bind=True, queue='high_priority', **RETRY_CONFIG)
def process_urgent_document(self, job_id: str, file_path: str, filename: str):
    """
    Process document with high priority.
    
    Use for:
    - Single document uploads (user waiting)
    - Small files
    - Premium users
    """
    pipeline = create_pipeline(job_id, file_path, filename)
    return pipeline.apply_async(queue='high_priority')


@celery_app.task(bind=True, queue='batch', **RETRY_CONFIG)
def process_batch_document(self, job_id: str, file_path: str, filename: str):
    """
    Process document with low priority.
    
    Use for:
    - Batch uploads
    - Large files
    - Background reprocessing
    """
    pipeline = create_pipeline(job_id, file_path, filename)
    return pipeline.apply_async(queue='batch')

