"""
DeedLens Worker Module
Celery-based async task processing for document pipeline.
"""

# Lazy imports to avoid errors when Celery is not installed
CELERY_AVAILABLE = False

def get_celery_app():
    """Get Celery app if available."""
    global CELERY_AVAILABLE
    try:
        from .celery_app import celery_app
        CELERY_AVAILABLE = True
        return celery_app
    except ImportError:
        return None


def get_process_document_task():
    """Get process_document_task if available."""
    try:
        from .tasks import process_document_task
        return process_document_task
    except ImportError:
        return None


__all__ = ['get_celery_app', 'get_process_document_task', 'CELERY_AVAILABLE']
