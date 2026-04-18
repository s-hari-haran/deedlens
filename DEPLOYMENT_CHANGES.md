# DeedLens PostgreSQL Migration - Deployment Changes

## What Was Changed

### 1. **core/jobs.py** - Migrated to PostgreSQL
- Changed from SQLite to psycopg2 (PostgreSQL driver)
- Updated JobManager to use PostgreSQL connections
- Rewrote all SQL queries from SQLite syntax (?) to PostgreSQL syntax (%s)
- Updated schema mapping to match PostgreSQL tables:
  - `jobs.filename` (was `original_filename`)
  - `jobs.current_stage` (was `current_step`)
  - Metadata stored in JSONB format for metrics
- Added connection pooling and error handling

### 2. **core/db_init.py** - New File
- Created database initialization script
- Reads `core/schema.sql` and executes it on PostgreSQL startup
- Includes retry logic (max 10 attempts with 2-second delays)
- Gracefully handles connection failures with warnings

### 3. **api/app.py** - Added Startup Hook
- Added FastAPI `@app.on_event("startup")` handler
- Calls `init_postgres()` on application startup
- Initializes PostgreSQL schema automatically

## How to Deploy

### Option A: Using Docker (Recommended)

```bash
cd "c:\projects\iyal-06 Proper Analyser"

# Clean up old containers/volumes
docker-compose down -v

# Rebuild with fresh images
docker-compose build --no-cache

# Start all services (PostgreSQL will be initialized automatically)
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Option B: Manual Verification

After starting Docker:

```bash
# Check if PostgreSQL is healthy
docker exec deedlens-postgres pg_isready -U deedlens

# Check if API initialized schema
docker logs deedlens-api | grep -E "(PostgreSQL|schema|initialized)"

# Verify jobs table exists
docker exec deedlens-postgres psql -U deedlens -d deedlens -c "\dt jobs"
```

## Testing the Pipeline

### 1. Upload a Document (via Streamlit UI)
- Open http://localhost:8501
- Upload a document
- You should see a job created immediately

### 2. Verify Job in PostgreSQL
```bash
docker exec deedlens-postgres psql -U deedlens -d deedlens << 'EOF'
SELECT id, filename, status, progress, created_at FROM jobs ORDER BY created_at DESC LIMIT 5;
EOF
```

Expected output:
```
                  id                  |     filename      | status  | progress |       created_at
--------------------------------------+-------------------+---------+----------+---------------------
 12345678-1234-1234-1234-123456789abc | sample_deed.png   | pending |        0 | 2026-04-08 10:50:03
```

### 3. Verify Job in Redis Queue
```bash
docker exec deedlens-redis redis-cli LLEN celery
```

Expected output: `1` (or number of queued jobs)

### 4. Watch Worker Process the Job
```bash
docker-compose logs worker --follow
```

Expected output:
```
deedlens-worker  | [2026-04-08 10:50:05,100: INFO/MainProcess] Task worker.tasks.process_document_task[...] received
deedlens-worker  | [12345678...] Starting ocr
...
deedlens-worker  | [12345678...] ocr completed in 2500ms
```

### 5. Check Job Completion
```bash
# After ~30 seconds, check job status
docker exec deedlens-postgres psql -U deedlens -d deedlens << 'EOF'
SELECT id, status, progress FROM jobs WHERE id = '12345678-...';
EOF
```

Expected progression:
- `pending` → `processing` → `completed`
- `progress` from 0 → 50 → 100

### 6. View Metrics
```bash
curl http://localhost:8000/metrics
```

Expected response:
```json
{
  "time_window": "24h",
  "total_processed": 1,
  "total_failed": 0,
  "overall_success_rate": 1.0,
  "avg_total_duration_ms": 5000.0,
  "throughput_per_hour": 120.0,
  "stages": [...]
}
```

## Troubleshooting

### Issue: Jobs not appearing in PostgreSQL

**Check 1: Is the API calling the right database?**
```bash
docker exec deedlens-api cat /app/config.py | grep DATABASE
```

**Check 2: Is schema initialized?**
```bash
docker exec deedlens-postgres psql -U deedlens -d deedlens \
  -c "SELECT count(*) FROM information_schema.tables WHERE table_name='jobs';"
```

Should return `1`. If `0`, schema wasn't created.

**Check 3: Are there errors in the API logs?**
```bash
docker logs deedlens-api | grep -E "(ERROR|postgres|database)"
```

---

### Issue: Celery worker not picking up tasks

**Check 1: Is the worker connected?**
```bash
docker logs deedlens-worker | grep "Connected to redis://"
```

**Check 2: Is Redis queue empty?**
```bash
docker exec deedlens-redis redis-cli KEYS "celery*"
```

**Check 3: Are tasks registered?**
```bash
docker logs deedlens-worker | grep "\[tasks\]" -A 10
```

Should see:
```
[tasks]
  . tasks.clean
  . tasks.embed
  . tasks.ner
  . tasks.ocr
  . tasks.store
  . worker.tasks.process_document_task
```

---

### Issue: "psycopg2" import error

This means requirements.txt wasn't installed properly. Rebuild:

```bash
docker-compose build --no-cache deedlens-api
docker-compose up -d deedlens-api
```

---

## Configuration Reference

### PostgreSQL Connection String
Default: `postgresql://deedlens:deedlens_secret@deedlens-postgres:5432/deedlens`

Set via environment variable `DATABASE_URL` in docker-compose.yml

### Database Schema Location
`core/schema.sql` - Contains:
- `documents` table (core documents)
- `entities` table (extracted entities)
- `jobs` table (processing jobs) ← **CRITICAL**
- `processing_logs` table (per-stage metrics)
- Indexes, triggers, views for observability

### Job Status Flow
```
pending → queued → processing → completed
                               ↓
                             failed (with error_message)
```

### Processing Stages
- `upload` - File received
- `ocr` - Text extraction
- `clean` - Text cleaning
- `ner` - Entity extraction
- `embed` - Vector embedding
- `store` - Save to database

---

## Key Files Changed

| File | Changes | Impact |
|------|---------|--------|
| `core/jobs.py` | Complete rewrite to use PostgreSQL | Jobs now persist in DB |
| `core/db_init.py` | NEW - Schema initialization | Automatic setup on startup |
| `api/app.py` | Added startup hook | PostgreSQL schema created on API start |
| `core/schema.sql` | Already existed | Now properly used |
| `docker-compose.yml` | Already updated | DATABASE_URL environment set |

---

## Next Steps After Deployment

1. ✅ Deploy and verify jobs are being created
2. Watch the Celery worker logs as documents are processed
3. Check `/metrics` endpoint for processing statistics
4. Monitor `processing_logs` table for per-stage metrics
5. (Optional) Set up database backups for PostgreSQL

---

## Command Reference

```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Clean everything (delete volumes too)
docker-compose down -v

# Rebuild specific service
docker-compose build --no-cache deedlens-api

# View logs
docker-compose logs [service-name] --follow

# Access PostgreSQL
docker exec deedlens-postgres psql -U deedlens -d deedlens

# Access Redis
docker exec deedlens-redis redis-cli

# Restart everything
docker-compose restart
```

---

## Success Criteria

After deployment, you should see:

1. ✅ All 5 containers running: `docker-compose ps`
2. ✅ Schema initialized: `SELECT count(*) FROM jobs;` returns `0` (but table exists)
3. ✅ API health check: `curl http://localhost:8000/`
4. ✅ Upload document → job created in PostgreSQL
5. ✅ Worker picks up task from Redis
6. ✅ Job status changes from `pending` → `completed`
7. ✅ Metrics appear in `/metrics` endpoint
8. ✅ Processing logs in PostgreSQL `processing_logs` table

---

**Questions or issues?** Check the logs first, then check the troubleshooting section above.
