"""
DeedLens Job Manager
Handles job lifecycle: create, update, query, retry.
Uses PostgreSQL for persistence.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from config import settings


class JobStatus(Enum):
    """Job status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ProcessingStep(Enum):
    """Processing pipeline steps."""
    UPLOAD = "upload"
    OCR = "ocr"
    CLEAN = "clean"
    NER = "ner"
    EMBED = "embed"
    SAVE = "save"


@dataclass
class Job:
    """Represents a processing job."""
    id: str
    file_path: str
    original_filename: str
    status: JobStatus = JobStatus.PENDING
    document_id: Optional[str] = None
    current_step: Optional[ProcessingStep] = None
    progress: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None
    # Metrics
    ocr_duration_ms: Optional[int] = None
    clean_duration_ms: Optional[int] = None
    ner_duration_ms: Optional[int] = None
    embed_duration_ms: Optional[int] = None
    total_duration_ms: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "original_filename": self.original_filename,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "document_id": self.document_id,
            "current_step": self.current_step.value if isinstance(self.current_step, ProcessingStep) else self.current_step,
            "progress": self.progress,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "ocr_duration_ms": self.ocr_duration_ms,
            "clean_duration_ms": self.clean_duration_ms,
            "ner_duration_ms": self.ner_duration_ms,
            "embed_duration_ms": self.embed_duration_ms,
            "total_duration_ms": self.total_duration_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create Job from dictionary."""
        status = data.get("status", JobStatus.PENDING.value)
        if isinstance(status, str):
            status = JobStatus(status)
        
        current_step = data.get("current_step")
        if current_step and isinstance(current_step, str):
            try:
                current_step = ProcessingStep(current_step)
            except ValueError:
                current_step = None
        
        # Helper function to convert datetime to ISO string
        def to_iso_string(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value
            # Handle datetime objects from PostgreSQL
            try:
                from datetime import datetime
                if isinstance(value, datetime):
                    return value.isoformat()
            except:
                pass
            return str(value)
        
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            original_filename=data["original_filename"],
            status=status,
            document_id=data.get("document_id"),
            current_step=current_step,
            progress=data.get("progress", 0),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            created_at=to_iso_string(data.get("created_at")),
            started_at=to_iso_string(data.get("started_at")),
            completed_at=to_iso_string(data.get("completed_at")),
            updated_at=to_iso_string(data.get("updated_at")),
            ocr_duration_ms=data.get("ocr_duration_ms"),
            clean_duration_ms=data.get("clean_duration_ms"),
            ner_duration_ms=data.get("ner_duration_ms"),
            embed_duration_ms=data.get("embed_duration_ms"),
            total_duration_ms=data.get("total_duration_ms"),
        )


class JobManager:
    """Manages job lifecycle in PostgreSQL database."""
    
    def __init__(self):
        """Initialize with PostgreSQL connection."""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        self.psycopg2 = psycopg2
        self.RealDictCursor = RealDictCursor
        self.db_url = settings.database_url or "postgresql://deedlens:deedlens_secret@localhost:5432/deedlens"
    
    def _get_conn(self):
        """Get PostgreSQL connection."""
        try:
            conn = self.psycopg2.connect(self.db_url)
            return conn
        except Exception as e:
            print(f"Warning: PostgreSQL not available: {e}")
            return None
    
    def create_job(self, file_path: str, original_filename: str) -> Job:
        """Create a new job."""
        now = datetime.utcnow().isoformat()
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            file_path=file_path,
            original_filename=original_filename,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now
        )
        
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO jobs (
                        id, filename, file_path, status, 
                        progress, retry_count, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    job.id, job.original_filename, job.file_path,
                    job.status.value, job.progress, job.retry_count, 
                    job.created_at
                ))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Warning: Failed to save job to PostgreSQL: {e}")
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor(cursor_factory=self.RealDictCursor)
                cur.execute("""
                    SELECT 
                        id, filename as original_filename, file_path, status, 
                        document_id, current_stage as current_step, progress, 
                        error_message, retry_count, created_at, started_at, 
                        completed_at, created_at as updated_at
                    FROM jobs WHERE id = %s
                """, (job_id,))
                row = cur.fetchone()
                conn.close()
                
                if not row:
                    return None
                
                return Job.from_dict(dict(row))
        except Exception as e:
            print(f"Warning: Failed to get job from PostgreSQL: {e}")
        
        return None
    
    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_step: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
        document_id: Optional[str] = None,
        **metrics
    ) -> Optional[Job]:
        """Update job status and metrics."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor()
                
                updates = []
                values = []
                
                if status:
                    updates.append("status = %s")
                    values.append(status)
                    
                    if status == JobStatus.PROCESSING.value:
                        updates.append("started_at = %s")
                        values.append(datetime.utcnow().isoformat())
                    elif status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                        updates.append("completed_at = %s")
                        values.append(datetime.utcnow().isoformat())
                
                if current_step:
                    updates.append("current_stage = %s")
                    values.append(current_step)
                
                if progress is not None:
                    updates.append("progress = %s")
                    values.append(progress)
                
                if error_message is not None:
                    updates.append("error_message = %s")
                    values.append(error_message)
                
                if document_id:
                    updates.append("document_id = %s")
                    values.append(document_id)
                
                # Add metrics (store in metadata JSONB)
                if metrics:
                    import json
                    updates.append("metadata = metadata || %s::jsonb")
                    values.append(json.dumps(metrics))
                
                if updates:
                    values.append(job_id)
                    query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = %s"
                    cur.execute(query, values)
                    conn.commit()
                conn.close()
        except Exception as e:
            print(f"Warning: Failed to update job in PostgreSQL: {e}")
        
        return self.get_job(job_id)
    
    def increment_retry(self, job_id: str) -> Optional[Job]:
        """Increment retry count."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE jobs SET retry_count = retry_count + 1, status = %s WHERE id = %s",
                    (JobStatus.RETRYING.value, job_id)
                )
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Warning: Failed to increment retry: {e}")
        
        return self.get_job(job_id)
    
    def get_pending_jobs(self, limit: int = 10) -> List[Job]:
        """Get pending jobs for processing."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor(cursor_factory=self.RealDictCursor)
                cur.execute("""
                    SELECT 
                        id, filename as original_filename, file_path, status, 
                        document_id, current_stage as current_step, progress, 
                        error_message, retry_count, created_at, started_at, 
                        completed_at, created_at as updated_at
                    FROM jobs 
                    WHERE status = %s OR status = %s
                    ORDER BY created_at ASC 
                    LIMIT %s
                """, (JobStatus.PENDING.value, JobStatus.RETRYING.value, limit))
                rows = cur.fetchall()
                conn.close()
                
                return [Job.from_dict(dict(row)) for row in rows]
        except Exception as e:
            print(f"Warning: Failed to get pending jobs: {e}")
        
        return []
    
    def get_jobs_by_status(self, status: JobStatus, limit: int = 50) -> List[Job]:
        """Get all jobs with given status."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor(cursor_factory=self.RealDictCursor)
                cur.execute("""
                    SELECT 
                        id, filename as original_filename, file_path, status, 
                        document_id, current_stage as current_step, progress, 
                        error_message, retry_count, created_at, started_at, 
                        completed_at, created_at as updated_at
                    FROM jobs 
                    WHERE status = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (status.value, limit))
                rows = cur.fetchall()
                conn.close()
                
                return [Job.from_dict(dict(row)) for row in rows]
        except Exception as e:
            print(f"Warning: Failed to get jobs by status: {e}")
        
        return []
    
    def get_all_jobs(self, limit: int = 50) -> List[Job]:
        """Get all jobs, sorted by most recent first."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor(cursor_factory=self.RealDictCursor)
                cur.execute("""
                    SELECT 
                        id, filename as original_filename, file_path, status, 
                        document_id, current_stage as current_step, progress, 
                        error_message, retry_count, created_at, started_at, 
                        completed_at, created_at as updated_at
                    FROM jobs 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
                conn.close()
                
                return [Job.from_dict(dict(row)) for row in rows]
        except Exception as e:
            print(f"Warning: Failed to get all jobs: {e}")
        
        return []
    
    def get_recent_jobs(self, limit: int = 20) -> List[Job]:
        """Get recent jobs."""
        return self.get_all_jobs(limit)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get aggregate metrics."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor(cursor_factory=self.RealDictCursor)
                
                # Status counts
                cur.execute("SELECT status, COUNT(*) as count FROM jobs GROUP BY status")
                status_results = cur.fetchall()
                status_counts = {row['status']: row['count'] for row in status_results}
                
                # Average durations
                cur.execute("""
                    SELECT 
                        COALESCE(AVG(duration_ms), 0) as avg_total_ms
                    FROM jobs WHERE status = %s
                """, (JobStatus.COMPLETED.value,))
                avg_result = cur.fetchone()
                
                conn.close()
                
                total = sum(status_counts.values())
                
                return {
                    "total_jobs": total,
                    "pending": status_counts.get(JobStatus.PENDING.value, 0),
                    "processing": status_counts.get(JobStatus.PROCESSING.value, 0),
                    "completed": status_counts.get(JobStatus.COMPLETED.value, 0),
                    "failed": status_counts.get(JobStatus.FAILED.value, 0),
                    "avg_duration_ms": round(avg_result['avg_total_ms'] or 0, 1) if avg_result else None,
                }
        except Exception as e:
            print(f"Warning: Failed to get metrics: {e}")
        
        return {
            "total_jobs": 0,
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "avg_duration_ms": None,
        }
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        try:
            conn = self._get_conn()
            if conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
                affected = cur.rowcount
                conn.commit()
                conn.close()
                return affected > 0
        except Exception as e:
            print(f"Warning: Failed to delete job: {e}")
        
        return False
    
    def cancel_job(self, job_id: str) -> Optional[Job]:
        """Cancel a pending job."""
        job = self.get_job(job_id)
        if job and job.status in [JobStatus.PENDING, JobStatus.RETRYING]:
            return self.update_job(job_id, status=JobStatus.CANCELLED.value)
        return job


# Singleton instance
_job_manager = None

def get_job_manager() -> JobManager:
    """Get or create job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
