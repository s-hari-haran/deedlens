# Render Deployment Guide

## 🎯 Why Render Over Railway?

| Feature | Railway | Render |
|---------|---------|--------|
| **Free Tier** | $5 credit | Truly free |
| **PostgreSQL** | Paid | Free (shared) |
| **Redis** | Paid | Free (shared) |
| **Services Limit** | 3 max | Unlimited |
| **Build Time** | Limited | More generous |
| **Databases** | Extra cost | Included |

**Render is perfect for your stack!** ✅

---

## 🚀 Step-by-Step Deployment

### Step 1: Create Render Account

1. Go to https://render.com
2. Click **"Get Started"**
3. Sign up with GitHub (easier!)
   - Click **"Continue with GitHub"**
   - Authorize Render to access your repos

---

### Step 2: Create Blueprint Deployment

A **Blueprint** is like Infrastructure-as-Code - Render reads `render.yaml` and deploys everything:

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"**
3. Select your GitHub repo: `deedlens`
4. Click **"Connect"**

---

### Step 3: Configure Services

Render will show you the services from `render.yaml`:

- ✅ **deedlens-api** (FastAPI on port 8000)
- ✅ **deedlens-worker** (Celery background jobs)
- ✅ **deedlens-postgres** (PostgreSQL database)
- ✅ **deedlens-redis** (Redis cache)

**Review and click "Create Blueprint"** ✅

---

### Step 4: Wait for Deployment

⏳ First deployment takes **5-10 minutes:**
- Builds Docker image
- Downloads ML models
- Initializes database
- Starts services

**Watch the logs:**
- Click each service to see real-time logs
- Look for "Deploy live" message

---

### Step 5: Add Environment Variables

1. Click **"deedlens-api"** service
2. Go to **"Environment"** tab
3. Add these variables:

```env
OCR_BACKEND=easyocr
OCR_DPI=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_SEARCH_MODE=hybrid
SEARCH_K=10
LOG_LEVEL=INFO
DEBUG=false
PYTHON_VERSION=3.11

# Optional: Add your API keys
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

4. **Save & redeploy**

---

### Step 6: Access Your App

Once deployed, you'll get URLs:

**In your Render Dashboard:**
- **API URL:** `https://deedlens-api-xxx.onrender.com`
- **Worker URL:** Background service (no URL needed)
- **Database:** Automatically connected

**Access Points:**
```
API Docs:     https://deedlens-api-xxx.onrender.com/docs
API Base:     https://deedlens-api-xxx.onrender.com
Health Check: https://deedlens-api-xxx.onrender.com/health
```

---

## 📊 What Gets Deployed

| Service | Type | Cost | Status |
|---------|------|------|--------|
| **deedlens-api** | Web Service | Free tier | Spins down after 15min idle |
| **deedlens-worker** | Background Worker | Free tier | Always running |
| **PostgreSQL** | Database | Free (shared) | 100GB storage |
| **Redis** | Cache | Free (shared) | 30GB storage |

**Total Cost:** $0 (free tier) or $7/month for production

---

## 🔄 Continuous Deployment

Every time you push to GitHub:
1. Render detects changes
2. Rebuilds your app
3. Redeploys automatically
4. Zero downtime

```bash
git push origin main  # That's it! Auto-deploys
```

---

## ⚠️ Free Tier Limits

- **Spins down:** Web service goes to sleep after 15 min of no requests
- **First request:** Takes ~30 sec to wake up (called "cold start")
- **Worker:** Stays on (no spin down)
- **Database:** Shared resources (but sufficient for dev/demo)

**To avoid cold starts:**
- Upgrade to "Standard" plan ($7/month) - keeps services always on

---

## 🐛 Troubleshooting

### Issue: "Deployment Failed"

Check the logs:
1. Click the service
2. Go to **"Logs"** tab
3. Look for error messages

Common causes:
- Model download timeout (increase timeout)
- Out of disk space (upgrade or clear cache)
- Memory limit exceeded (worker needs more)

### Issue: "Build timeout"

ML models are large. Increase build timeout:
1. Go to **"Settings"**
2. Increase **"Max Build Duration"** to 30 min

### Issue: Database connection fails

Check if PostgreSQL is running:
1. Click **"deedlens-postgres"** service
2. Verify status is **"Live"**
3. Check connection string in variables

### Issue: Worker not processing documents

Check worker logs:
1. Click **"deedlens-worker"** service
2. Go to **"Logs"** tab
3. Look for "Started worker" message

---

## 💰 Upgrading to Production

When you need more power:

```
Free Tier → Standard → Professional
$0        → $7/mo   → $12+/mo
```

- **Web service:** Stays on (no cold starts)
- **Disk space:** More storage
- **Memory:** More RAM for ML models
- **Priority:** Faster builds

---

## 📈 Performance

| Operation | Time |
|-----------|------|
| Cold start | ~30 sec |
| Warm start | <1 sec |
| OCR per page | 1-2 sec |
| NER extraction | 0.8-1.2 sec |
| Search | <100ms |
| End-to-end doc | 3-5 sec |

---

## 🎯 Next Steps After Deployment

1. **Test the API:**
   ```bash
   curl https://your-app.onrender.com/health
   ```

2. **Upload a test document:**
   - Use `/docs` page
   - Or use the Streamlit UI (if you add it)

3. **Monitor in real-time:**
   - Render Dashboard → Metrics tab
   - View CPU, memory, disk usage

4. **Set up alerts:**
   - Render Dashboard → Settings → Alerts
   - Get notified if service goes down

---

## 🚀 Advanced: Adding Streamlit Frontend

To add Streamlit UI on Render:

1. Update `render.yaml` with:
```yaml
  - type: web
    name: deedlens-ui
    runtime: python
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Push to GitHub
3. Render auto-redeploys with Streamlit

---

## 📝 Useful Commands

```bash
# View deployment status
curl https://your-app.onrender.com/health | jq

# Check API docs
open https://your-app.onrender.com/docs

# Trigger manual redeploy
# (Just push to GitHub)

# View logs
# (Use Render dashboard)
```

---

## 🆘 Support

- **Render Docs:** https://render.com/docs
- **Status Page:** https://status.render.com
- **Community:** https://render.com/community

---

**Your DeedLens is now on Render! 🎉**

