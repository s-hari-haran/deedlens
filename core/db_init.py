"""
Initialize PostgreSQL database schema on startup.
"""

import os
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def init_postgres():
    """Initialize PostgreSQL schema from core/schema.sql"""
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed, skipping PostgreSQL initialization")
        return
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url or not db_url.startswith("postgresql://"):
        logger.info("DATABASE_URL not set or not PostgreSQL, skipping schema initialization")
        return
    
    # Read schema file
    schema_file = Path(__file__).parent / "schema.sql"
    if not schema_file.exists():
        logger.warning(f"Schema file not found: {schema_file}")
        return
    
    with open(schema_file, 'r') as f:
        schema = f.read()
    
    # Retry logic for database startup
    max_retries = 10
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            # Execute schema
            cur.execute(schema)
            conn.commit()

            logger.info("✅ PostgreSQL schema initialized successfully")
            conn.close()
            return
        except Exception as e:
            # Treat "already exists" as success (idempotent)
            if "already exists" in str(e).lower():
                logger.info("✅ PostgreSQL schema already initialized (tables exist)")
                try:
                    conn.close()
                except:
                    pass
                return

            if attempt < max_retries - 1:
                logger.warning(f"⏳ PostgreSQL not ready, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ Failed to initialize PostgreSQL after {max_retries} attempts: {e}")
                raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_postgres()
