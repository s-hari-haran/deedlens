"""
Celery Application Configuration
"""

import os

# Disable accelerate before any transformers imports
os.environ['HF_ACCELERATE_AVAILABLE'] = '0'

# Ensure PyTorch nn module is available
try:
    import torch
    import torch.nn
except ImportError:
    pass

from celery import Celery

# Redis URL from environment or default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "deedlens",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,  # Acknowledge after task completes (safer)
    task_reject_on_worker_lost=True,

    # Retry settings
    task_default_retry_delay=5,  # 5 seconds between retries
    task_max_retries=3,

    # Result backend
    result_expires=3600,  # Results expire after 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time (for heavy ML tasks)
    worker_concurrency=2,  # Number of concurrent workers

    # Task routing - prioritize high_priority and batch queues
    task_routes={
        'worker.tasks.process_urgent_document': {'queue': 'high_priority'},
        'worker.tasks.process_batch_document': {'queue': 'batch'},
    },

    # Define queue names
    task_queues={
        'celery': {'exchange': 'celery', 'routing_key': 'celery'},
        'high_priority': {'exchange': 'high_priority', 'routing_key': 'high_priority'},
        'batch': {'exchange': 'batch', 'routing_key': 'batch'},
    },
)

# Optional: Configure for Windows/Docker compatibility
celery_app.conf.update(
    worker_pool='solo'  # Use solo pool for compatibility
)
