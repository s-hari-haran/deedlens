# DeedLens: AI-Powered Property Document Intelligence

> **Production-grade AI system for automated extraction, analysis, and intelligent search of legal property documents.**

DeedLens transforms unstructured property documents (deeds, agreements, leases) into structured, queryable data. Built with async job queues, ML-powered extraction, semantic search, and comprehensive observability.

---

## ✨ Key Features

- **Multi-Backend OCR** - EasyOCR (local), Tesseract, or Groq Vision (cloud)
- **Named Entity Recognition (NER)** - Extract persons, locations, dates, amounts, survey numbers
- **Semantic Search** - Hybrid BM25 + vector similarity (384-dim embeddings)
- **Async Processing Pipeline** - 5-stage Celery chain with granular metrics
- **Production Database** - PostgreSQL with JSONB, GIN indexes, full-text search
- **Job Management** - Status tracking, retry logic, priority queues
- **Observability** - Per-stage latency, throughput, error tracking
- **REST API** - Document upload, search, job management, metrics
- **Streamlit UI** - Interactive document interface
- **Docker Deployment** - Multi-container setup with health checks

---

## 🏗️ Architecture

### Processing Pipeline

```
Document Upload
      ↓
┌─────────────────────────────────────────────────────┐
│              ASYNC CELERY PIPELINE                  │
├─────────────────────────────────────────────────────┤
│  Stage 1: OCR         → Extract text from image    │
│  Stage 2: Clean       → Apply corrections + fixes  │
│  Stage 3: NER         → Extract entities           │
│  Stage 4: Embed       → Generate vectors (384-dim) │
│  Stage 5: Store       → Save to DB + search index  │
└─────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────┐
│           DATA PERSISTENCE & SEARCH                 │
├──────────────────┬──────────────────┬──────────────┤
│   PostgreSQL     │   FAISS Index    │  File Store  │
│  - documents     │  - embeddings    │  - originals │
│  - entities      │  - similarity    │              │
│  - jobs          │                  │              │
│  - logs          │                  │              │
└──────────────────┴──────────────────┴──────────────┘
```

### System Diagram

```
┌─────────────────────────────────────────────────────────┐
│              CLIENT LAYER                               │
├──────────────────────────┬────────────────────────────┤
│   Streamlit UI           │     FastAPI REST API        │
│   (port 8501)            │     (port 8000)             │
│   - Upload               │     - /upload               │
│   - Search               │     - /search               │
│   - Results              │     - /jobs                 │
│                          │     - /metrics              │
└──────────────┬───────────┴────────────┬────────────────┘
               │                        │
               ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│         MESSAGE QUEUE & PROCESSING                      │
├─────────────────────────────────────────────────────────┤
│              Redis (port 6379)                          │
│  - Job queue (priority, batch, default)                 │
│  - Task results & state                                 │
│                ↓                                         │
│         Celery Workers                                   │
│  - 2 concurrent workers                                 │
│  - Exponential backoff retry                            │
│  - Per-stage observability logging                      │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│         DATA LAYER                                      │
├──────────────────┬──────────────────┬────────────────┤
│  PostgreSQL      │  FAISS           │  File System   │
│  (port 5432)     │  Vector Index    │  (/data)       │
│                  │                  │                │
│  Tables:         │  384-dim         │  - Uploads    │
│  - documents     │  flat index      │  - Raw OCR    │
│  - entities      │  cosine search   │  - Embeddings │
│  - jobs          │                  │                │
│  - jobs_queue    │                  │                │
│  - processing    │                  │                │
│    _logs         │                  │                │
│  - search_meta   │                  │                │
└──────────────────┴──────────────────┴────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Latest |
| **Backend** | FastAPI | Latest |
| **Language** | Python | 3.11 |
| **Database** | PostgreSQL | 16 (prod) |
| **Cache/Queue** | Redis | 7 |
| **Task Queue** | Celery | 5.3 |
| **OCR** | EasyOCR / Tesseract / Groq | Latest |
| **NER** | spaCy | en_core_web_sm |
| **Embeddings** | SentenceTransformers | all-MiniLM-L6-v2 |
| **Vector Search** | FAISS | Latest |
| **Container** | Docker | Latest |

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Verify services
docker-compose ps

# Access:
# - Streamlit UI: http://localhost:8501
# - FastAPI docs: http://localhost:8000/docs
# - API: http://localhost:8000
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Terminal 1: Redis
docker run -d -p 6379:6379 --name deedlens-redis redis:7-alpine

# Terminal 2: Celery Worker
celery -A worker.celery_app worker --loglevel=info --pool=solo

# Terminal 3: FastAPI
uvicorn api.app:app --port 8000

# Terminal 4: Streamlit
streamlit run app.py
```

---

## 📋 Configuration

### Environment Variables

Create `.env` file (copy from `.env.example`):

```env
# Database
DATABASE_URL=postgresql://deedlens:deedlens_secret@localhost:5432/deedlens

# Redis
REDIS_URL=redis://localhost:6379/0

# OCR Backend (tesseract, easyocr, groq)
OCR_BACKEND=easyocr
OCR_DPI=200

# API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search
DEFAULT_SEARCH_MODE=hybrid
SEARCH_K=10

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

### Settings (`config.py`)

All settings are loaded via Pydantic Settings from `.env`:

```python
from config import settings

print(settings.database_url)
print(settings.ocr_backend)
print(settings.embedding_dimension)  # 384
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Health Check
```bash
GET /health
```

### Document Upload (Async)
```bash
POST /jobs/upload
Content-Type: multipart/form-data

file: <document.pdf or .jpg>
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2024-04-18T10:30:00Z"
}
```

### Get Job Status
```bash
GET /jobs/{job_id}
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 35,
  "current_step": "clean",
  "created_at": "2024-04-18T10:30:00Z",
  "started_at": "2024-04-18T10:30:05Z"
}
```

### List Jobs
```bash
GET /jobs?status=completed&limit=50
```

### Search Documents
```bash
POST /search
Content-Type: application/json

{
  "query": "sale deed bangalore",
  "mode": "hybrid",
  "limit": 10
}
```

Response:
```json
{
  "results": [
    {
      "document_id": "doc-123",
      "filename": "deed_001.pdf",
      "score": 0.87,
      "entities": {
        "PERSON": ["John Doe"],
        "LOCATION": ["Bangalore"],
        "AMOUNT": ["₹50,00,000"]
      },
      "excerpt": "Sale deed executed at Bangalore..."
    }
  ]
}
```

### Get Metrics
```bash
GET /metrics
```

Response:
```json
{
  "total_documents": 1250,
  "processing_stats": {
    "avg_duration_ms": 3450,
    "success_rate": 0.98,
    "failure_rate": 0.02
  },
  "stage_metrics": {
    "ocr": {"avg_ms": 1200, "success": 1225, "failed": 5},
    "clean": {"avg_ms": 450, "success": 1225, "failed": 0},
    "ner": {"avg_ms": 850, "success": 1220, "failed": 5},
    "embed": {"avg_ms": 600, "success": 1220, "failed": 0},
    "store": {"avg_ms": 350, "success": 1220, "failed": 0}
  }
}
```

Full API docs available at: http://localhost:8000/docs

---

## 🔍 Search Capabilities

### 1. Semantic Search (Vector-based)
Uses embeddings to find conceptually similar documents.

```python
# Find documents about property sales
results = search_engine.semantic_search(
    query="sale of residential property",
    top_k=10
)
```

### 2. Keyword Search (BM25)
Traditional full-text search on extracted entities and text.

```python
# Find documents with specific location
results = search_engine.keyword_search(
    query="Bangalore Indiranagar",
    top_k=10
)
```

### 3. Hybrid Search (Default)
Combines vector + keyword with configurable weights:
```
score = (0.6 × vector_score) + (0.3 × bm25_score) + (0.1 × recency)
```

---

## 🔧 Database Schema

### Documents Table
```sql
documents
├── id (UUID PK)
├── filename (text)
├── file_path (text)
├── cleaned_text (text, full-text indexed)
├── document_type (text)
├── metadata (JSONB, GIN indexed)
├── created_at (timestamp)
└── updated_at (timestamp)
```

### Entities Table
```sql
entities
├── id (UUID PK)
├── document_id (FK → documents)
├── entity_type (ENUM: PERSON, LOCATION, DATE, AMOUNT, etc.)
├── value (text)
├── confidence (float 0-1)
├── extraction_method (text)
└── created_at (timestamp)
```

### Jobs Table
```sql
jobs
├── id (UUID PK)
├── filename (text)
├── file_path (text)
├── status (ENUM: pending, processing, completed, failed)
├── progress (int 0-100)
├── document_id (FK → documents)
├── error_message (text)
├── retry_count (int)
├── metadata (JSONB)
├── created_at (timestamp)
└── updated_at (timestamp)
```

### Processing Logs Table
```sql
processing_logs
├── id (UUID PK)
├── job_id (FK → jobs)
├── stage (text: ocr, clean, ner, embed, store)
├── status (text: started, completed, failed)
├── duration_ms (int)
├── metrics (JSONB)
├── error_details (text)
├── started_at (timestamp)
└── completed_at (timestamp)
```

---

## 📊 Observability

### Processing Metrics

Track per-stage performance:

```bash
# Get stage-specific metrics
GET /metrics/stage/ocr
GET /metrics/stage/ner
GET /metrics/stage/embed

# Get daily throughput
GET /metrics/daily?days=30
```

### Processing Logs

Every document passes through 5 observable stages:

1. **OCR** - Image → Text (1-2 seconds)
2. **Clean** - Remove noise, fix Indian OCR errors (0.2-0.5s)
3. **NER** - Extract entities (0.8-1.2s)
4. **Embed** - Generate vectors (0.5-1s)
5. **Store** - Save to DB (0.2-0.5s)

Total average: **3-5 seconds per document**

---

## 🐳 Docker Deployment

### Services

```yaml
postgres:16-alpine
  - Port: 5432
  - Volume: postgres_data
  - Health check: enabled

redis:7-alpine
  - Port: 6379
  - Health check: enabled

worker (Celery)
  - 2 concurrent workers
  - Depends on: postgres, redis
  - Health check: task processing

api (FastAPI)
  - Port: 8000
  - Depends on: postgres, redis, worker
  - Health check: /health

streamlit
  - Port: 8501
  - Depends on: api
  - Health check: HTTP 200
```

### Build & Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f worker
docker-compose logs -f api
docker-compose logs -f streamlit

# Stop services
docker-compose down

# Clean up volumes
docker-compose down -v
```

---

## 🚢 Deployment Options

### Option 1: Railway (Recommended for Beginners)
```bash
# $5-15/month, simple setup
railway link
railway up

# Access: https://your-project.railway.app
```

### Option 2: Render
```bash
# Free tier available
# Connect GitHub repo → auto-deploy
# Native PostgreSQL support
```

### Option 3: Hetzner Cloud
```bash
# $5-10/month for VPS
# Full control, best for production
docker-compose up -d
```

### Option 4: AWS / GCP / Azure
- Use managed databases (RDS, Cloud SQL)
- Deploy containers on ECS/Cloud Run
- CDN for static assets

---

## 📈 Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **OCR Speed** | 1-2 sec/page | Depends on image quality |
| **NER Speed** | 0.8-1.2 sec/doc | spaCy model inference |
| **Embed Speed** | 0.5-1 sec/doc | 384-dim SentenceTransformers |
| **End-to-end** | 3-5 sec/doc | All stages combined |
| **Search Speed** | <100ms | With 1M documents indexed |
| **Concurrent Jobs** | 2-10 | Limited by worker pool |
| **DB Storage** | ~50KB/doc | Varies by document length |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=. tests/

# Run integration tests
pytest tests/test_api.py::test_upload_and_search -v
```

Test Coverage:
- ✅ API endpoints (test_api.py)
- ✅ Database operations (test_database.py)
- ✅ NER model (test_ner_model.py)
- ✅ Text cleaning (test_text_cleaner.py)

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest`
3. Commit: `git commit -m "feat: your feature"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

---

## 📝 License

MIT License - See LICENSE file

---

## 📞 Support

- **Issues**: GitHub Issues
- **Docs**: See `/docs` folder
- **API Docs**: http://localhost:8000/docs (when running)

---

## 🗺️ Roadmap

- [ ] Multi-language OCR support
- [ ] Custom NER model training
- [ ] Batch document processing UI
- [ ] Document versioning & change tracking
- [ ] Advanced analytics dashboard
- [ ] Webhook notifications
- [ ] GraphQL API
- [ ] Mobile app

---

**Made with ❤️ for property document intelligence**
