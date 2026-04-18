# Analytics Module
# Provides metrics and observability utilities

# NOTE: clustering.py and regression.py have been deprecated
# They added "ML demo" features that diluted the backend positioning
# 
# This module now focuses on operational analytics:
# - Processing metrics
# - Throughput analysis
# - Quality metrics

from typing import Dict, Any, Optional
from datetime import datetime, timedelta


def get_processing_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get processing summary for observability.
    
    Returns:
        Dictionary with processing metrics
    """
    try:
        from core.jobs import get_job_manager
        job_manager = get_job_manager()
        return job_manager.get_metrics_summary()
    except Exception:
        return {
            "total_jobs": 0,
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }

