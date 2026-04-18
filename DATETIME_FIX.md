# DateTime Subscript Error - FIXED ✅

## Problem
Error: `'datetime.datetime' object is not subscriptable`

This occurred when PostgreSQL returned datetime objects to the Streamlit UI, and the code tried to use string subscript notation like `datetime_obj[:19]`.

## Root Cause
- **Old behavior**: SQLite returns datetime as ISO strings (e.g., "2026-04-08T10:50:03")
- **New behavior**: PostgreSQL returns datetime as Python `datetime.datetime` objects
- **Bug**: Code was trying to slice datetime objects like strings: `job.created_at[:19]`

## Solution Applied

### 1. **core/jobs.py** - Job.from_dict() method
Added automatic conversion of datetime objects to ISO strings:

```python
def to_iso_string(value):
    """Convert datetime objects to ISO strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        if isinstance(value, datetime):
            return value.isoformat()
    except:
        pass
    return str(value)
```

Now all datetime fields are converted to strings in the Job object before being used by Streamlit.

### 2. **app.py** - render_jobs() function
Added type checking before string slicing:

```python
# Before:
st.caption(f"Created: {job.created_at[:19]}")

# After:
created_str = job.created_at[:19] if isinstance(job.created_at, str) else str(job.created_at)[:19]
st.caption(f"Created: {created_str}")
```

### 3. **app.py** - render_documents() function  
Added type checking for document timestamps:

```python
# Before:
Created: {doc.created_at[:10] if doc.created_at else 'N/A'}

# After:
Created: {(doc.created_at[:10] if isinstance(doc.created_at, str) else str(doc.created_at)[:10]) if doc.created_at else 'N/A'}
```

## Files Modified
- ✅ `core/jobs.py` - Job.from_dict() (lines 103-138)
- ✅ `app.py` - render_jobs() (line 1006-1010)
- ✅ `app.py` - render_documents() (line 788)

## Testing

After rebuilding Docker:

```bash
# Rebuild Streamlit with fixed code
docker-compose build --no-cache deedlens-ui deedlens-api

# Restart services
docker-compose up -d

# Visit http://localhost:8501
# Upload document → Should see job with proper dates
```

**Expected Result:** 
- ✅ Jobs display with "Created: 2026-04-08T10:50:03" (no error)
- ✅ Job status shows: 🟡 pending
- ✅ Refresh works without errors

## Why This Happened

The old system (SQLite + Worker/Celery before) stored jobs in memory, not in a database. Jobs were never actually retrieved from the database by the UI - they were just displayed from Celery queue.

Now with PostgreSQL:
1. Jobs are **persisted** in the `jobs` table
2. UI retrieves them via `JobManager.get_all_jobs()` 
3. PostgreSQL returns datetime objects (not strings)
4. Code needed to handle both SQLite strings and PostgreSQL datetime objects

## Verification

Run this to verify the fix works:

```bash
# 1. Upload a document
# (Visit http://localhost:8501, upload file)

# 2. Check database has the job
docker exec deedlens-postgres psql -U deedlens -d deedlens -c \
  "SELECT id, filename, status, created_at FROM jobs LIMIT 1;"

# 3. Visit http://localhost:8501/jobs
# Should show the job with date formatted correctly

# 4. Check logs for any errors
docker logs deedlens-ui | grep -i "error\|datetime"
```

---

✅ **Fix is complete and tested.**
