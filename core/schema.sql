-- DeedLens PostgreSQL Schema
-- Designed for production: indexes, relationships, JSONB, audit trails

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- ============================================================================
-- DOCUMENTS TABLE
-- Core document storage with full metadata
-- ============================================================================
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Core fields
    filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    checksum VARCHAR(64),  -- SHA-256 for deduplication
    
    -- Extracted content
    raw_text TEXT,
    cleaned_text TEXT,
    page_count INTEGER DEFAULT 1,
    
    -- Document classification
    document_type VARCHAR(100),  -- sale_deed, gift_deed, lease, etc.
    confidence_score DECIMAL(5,4),  -- 0.0000 to 1.0000
    
    -- Flexible metadata (JSONB for extensibility)
    metadata JSONB DEFAULT '{}',
    
    -- Vector embedding reference
    embedding_id VARCHAR(100),  -- Reference to FAISS index
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    
    -- Soft delete
    deleted_at TIMESTAMPTZ
);

-- Indexes for documents
CREATE INDEX idx_documents_filename ON documents(filename);
CREATE INDEX idx_documents_document_type ON documents(document_type);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_checksum ON documents(checksum);  -- For dedup lookups
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);  -- JSONB queries
CREATE INDEX idx_documents_cleaned_text_trgm ON documents USING GIN(cleaned_text gin_trgm_ops);  -- Fuzzy search

-- ============================================================================
-- ENTITIES TABLE
-- Normalized entity storage (one row per entity)
-- ============================================================================
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Entity classification
    entity_type VARCHAR(50) NOT NULL,  -- PERSON, LOCATION, MONEY, DATE, SURVEY_NUMBER, etc.
    entity_subtype VARCHAR(50),  -- buyer, seller, witness, village, district, etc.
    
    -- Entity value
    value TEXT NOT NULL,
    normalized_value TEXT,  -- Cleaned/standardized version
    
    -- Position in document
    start_offset INTEGER,
    end_offset INTEGER,
    
    -- Confidence
    confidence DECIMAL(5,4),
    extraction_method VARCHAR(50),  -- spacy, regex, llm
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for entities
CREATE INDEX idx_entities_document_id ON entities(document_id);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_type_subtype ON entities(entity_type, entity_subtype);
CREATE INDEX idx_entities_value ON entities(value);
CREATE INDEX idx_entities_normalized ON entities(normalized_value);

-- ============================================================================
-- JOBS TABLE
-- Job lifecycle with full audit trail
-- ============================================================================
CREATE TYPE job_status AS ENUM (
    'pending',
    'queued',
    'processing',
    'completed',
    'failed',
    'cancelled',
    'retrying'
);

CREATE TYPE processing_stage AS ENUM (
    'upload',
    'ocr',
    'clean',
    'ner',
    'embed',
    'store',
    'complete'
);

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Job identification
    celery_task_id VARCHAR(100),
    
    -- Document reference
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000),
    
    -- Status tracking
    status job_status DEFAULT 'pending',
    current_stage processing_stage DEFAULT 'upload',
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Error handling
    error_message TEXT,
    error_stage processing_stage,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Priority (lower = higher priority)
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Calculated fields
    duration_ms INTEGER GENERATED ALWAYS AS (
        CASE 
            WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
            THEN EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000
            ELSE NULL 
        END
    ) STORED
);

-- Indexes for jobs
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_status_priority ON jobs(status, priority, created_at);  -- Queue ordering
CREATE INDEX idx_jobs_document_id ON jobs(document_id);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX idx_jobs_celery_task_id ON jobs(celery_task_id);

-- ============================================================================
-- PROCESSING_LOGS TABLE
-- Per-stage metrics for observability (THIS IS THE KILLER FEATURE)
-- ============================================================================
CREATE TABLE processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Stage identification
    stage processing_stage NOT NULL,
    
    -- Timing
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    
    -- Queue metrics
    queue_wait_ms INTEGER,  -- Time spent waiting in queue
    
    -- Status
    status VARCHAR(20) NOT NULL CHECK (status IN ('started', 'completed', 'failed', 'skipped')),
    
    -- Error details
    error_type VARCHAR(100),
    error_message TEXT,
    stack_trace TEXT,
    
    -- Stage-specific metrics (JSONB for flexibility)
    metrics JSONB DEFAULT '{}',
    -- Example metrics:
    -- OCR: {"pages": 5, "confidence": 0.92, "backend": "easyocr"}
    -- NER: {"entities_found": 23, "processing_time_per_page": 120}
    -- Embed: {"embedding_dim": 384, "model": "all-MiniLM-L6-v2"}
    
    -- Resource usage
    memory_mb INTEGER,
    cpu_percent DECIMAL(5,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for processing_logs (critical for metrics queries)
CREATE INDEX idx_processing_logs_job_id ON processing_logs(job_id);
CREATE INDEX idx_processing_logs_stage ON processing_logs(stage);
CREATE INDEX idx_processing_logs_status ON processing_logs(status);
CREATE INDEX idx_processing_logs_created_at ON processing_logs(created_at DESC);
CREATE INDEX idx_processing_logs_stage_status ON processing_logs(stage, status);

-- ============================================================================
-- SEARCH_METADATA TABLE
-- Hybrid search support (vector + keyword + filters)
-- ============================================================================
CREATE TABLE search_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Precomputed search fields (denormalized for speed)
    
    -- Price range (extracted from entities)
    min_price DECIMAL(15,2),
    max_price DECIMAL(15,2),
    
    -- Location hierarchy
    state VARCHAR(100),
    district VARCHAR(100),
    taluk VARCHAR(100),
    village VARCHAR(100),
    
    -- Date range
    execution_date DATE,
    registration_date DATE,
    
    -- Document type
    document_type VARCHAR(100),
    
    -- Parties (denormalized for filtering)
    buyer_names TEXT[],
    seller_names TEXT[],
    
    -- Survey numbers (for exact matching)
    survey_numbers TEXT[],
    
    -- Full-text search vector
    search_vector TSVECTOR,
    
    -- BM25-style stats
    word_count INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for search_metadata
CREATE INDEX idx_search_metadata_document_id ON search_metadata(document_id);
CREATE INDEX idx_search_metadata_price ON search_metadata(min_price, max_price);
CREATE INDEX idx_search_metadata_location ON search_metadata(state, district, taluk, village);
CREATE INDEX idx_search_metadata_date ON search_metadata(execution_date, registration_date);
CREATE INDEX idx_search_metadata_type ON search_metadata(document_type);
CREATE INDEX idx_search_metadata_survey ON search_metadata USING GIN(survey_numbers);
CREATE INDEX idx_search_metadata_fts ON search_metadata USING GIN(search_vector);

-- ============================================================================
-- API_KEYS TABLE
-- API authentication
-- ============================================================================
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of actual key
    name VARCHAR(100) NOT NULL,
    
    -- Permissions
    scopes TEXT[] DEFAULT ARRAY['read'],  -- read, write, admin
    
    -- Rate limiting
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_day INTEGER DEFAULT 10000,
    
    -- Tracking
    last_used_at TIMESTAMPTZ,
    request_count BIGINT DEFAULT 0,
    
    -- Validity
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- ============================================================================
-- VIEWS
-- Prebuilt queries for common operations
-- ============================================================================

-- Job queue view (what to process next)
CREATE VIEW job_queue AS
SELECT 
    j.id,
    j.filename,
    j.priority,
    j.created_at,
    j.retry_count,
    EXTRACT(EPOCH FROM (NOW() - j.created_at)) AS wait_seconds
FROM jobs j
WHERE j.status IN ('pending', 'retrying')
ORDER BY j.priority ASC, j.created_at ASC;

-- Processing metrics view (for /metrics endpoint)
CREATE VIEW processing_metrics AS
SELECT 
    stage,
    status,
    COUNT(*) AS total_count,
    AVG(duration_ms) AS avg_duration_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_duration_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99_duration_ms,
    AVG(queue_wait_ms) AS avg_queue_wait_ms,
    MIN(created_at) AS first_seen,
    MAX(created_at) AS last_seen
FROM processing_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY stage, status;

-- Daily throughput view
CREATE VIEW daily_throughput AS
SELECT 
    DATE(created_at) AS date,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_jobs,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_jobs,
    AVG(duration_ms) FILTER (WHERE status = 'completed') AS avg_duration_ms
FROM jobs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update search vector on document change
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.buyer_names, ' '), '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.seller_names, ' '), '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.village, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.district, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.survey_numbers, ' '), '')), 'A');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_search_vector
    BEFORE INSERT OR UPDATE ON search_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_search_metadata_updated_at
    BEFORE UPDATE ON search_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- COMMENTS (for documentation)
-- ============================================================================
COMMENT ON TABLE documents IS 'Core document storage with OCR text and metadata';
COMMENT ON TABLE entities IS 'Normalized entity extraction results';
COMMENT ON TABLE jobs IS 'Async job queue with lifecycle tracking';
COMMENT ON TABLE processing_logs IS 'Per-stage metrics for observability';
COMMENT ON TABLE search_metadata IS 'Denormalized search index with precomputed filters';
COMMENT ON VIEW processing_metrics IS 'Real-time processing performance metrics';
COMMENT ON VIEW job_queue IS 'Next jobs to process, ordered by priority';
