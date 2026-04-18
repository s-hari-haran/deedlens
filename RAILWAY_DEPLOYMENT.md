# Railway Deployment Guide

## 📋 Prerequisites

- GitHub account with your repo pushed ✅ (you already have this)
- Railway account (you have this)
- Railway CLI (we'll install)

---

## 🚀 Step-by-Step Deployment

### Step 1: Install Railway CLI

**Windows (PowerShell):**
```powershell
iwr https://releases.railway.app/setup.ps1 -useb | iex
```

**macOS/Linux:**
```bash
curl -L railway.app/install.sh | bash
```

**Verify installation:**
```bash
railway --version
```

---

### Step 2: Login to Railway

```bash
railway login
```

This will open a browser window to authenticate. Copy the token and paste it in the terminal.

```bash
# Verify login
railway whoami
```

---

### Step 3: Create Railway Project

**Option A: Using Web Dashboard (Easier)**

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Authorize Railway to access your GitHub
5. Select `s-hari-haran/deedlens` repo
6. Click "Deploy Now"

**Option B: Using Railway CLI**

```bash
cd /c/projects/iyal-06\ Proper\ Analyser

# Create project
railway init

# You'll be prompted to:
# - Enter project name: "deedlens"
# - Select environment: choose "production"

# This creates railway.json
```

---

### Step 4: Add Services (Via Web Dashboard - Easiest)

1. **Go to your project dashboard:** https://railway.app/dashboard
2. **Create PostgreSQL database:**
   - Click "New"
   - Select "Database" → "PostgreSQL"
   - Wait for deployment (1-2 min)
   - Copy the `DATABASE_URL` variable

3. **Create Redis cache:**
   - Click "New"
   - Select "Database" → "Redis"
   - Wait for deployment
   - Copy the `REDIS_URL` variable

4. **Deploy your app:**
   - Click "New"
   - Select "GitHub Repo"
   - Choose your `deedlens` repo
   - Wait for deployment

---

### Step 5: Configure Environment Variables

**Via Web Dashboard:**

1. Go to your app service
2. Click "Variables" tab
3. Add these variables:

```env
# Database (auto-provided by PostgreSQL service)
# DATABASE_URL=postgresql://... (already set)

# Redis (auto-provided by Redis service)
# REDIS_URL=redis://... (already set)

# OCR Settings
OCR_BACKEND=easyocr
OCR_DPI=200

# API Keys (OPTIONAL - get your own)
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search
DEFAULT_SEARCH_MODE=hybrid
SEARCH_K=10

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

**Important:** Railway automatically provides:
- `DATABASE_URL` (from PostgreSQL service)
- `REDIS_URL` (from Redis service)

You don't need to set these manually!

---

### Step 6: Deploy Your Application

**Method 1: Via Web Dashboard (Recommended)**
- Changes to your GitHub repo automatically trigger deployment
- Watch the deployment logs in real-time

**Method 2: Via Railway CLI**
```bash
railway up
```

---

## 🔍 Verify Deployment

### Check Service Status
```bash
railway status
```

### View Logs
```bash
# View all services
railway logs

# View specific service
railway logs --service api
railway logs --service worker
railway logs --service streamlit
```

### Access Your App
- **API Docs:** `https://your-project-api.railway.app/docs`
- **Streamlit UI:** `https://your-project-streamlit.railway.app`
- **API Base:** `https://your-project-api.railway.app`

---

## 🐛 Troubleshooting

### Issue: "No space left on device"

Railway's free tier has limited disk space. Solution:
1. Go to Variables
2. Set `OCR_BACKEND=groq` (uses cloud OCR, no local models)
3. Redeploy

### Issue: Database connection fails

Check if PostgreSQL service is running:
```bash
railway logs --service postgres
```

Verify `DATABASE_URL` is set correctly in Variables.

### Issue: Redis connection fails

Check Redis service:
```bash
railway logs --service redis
```

Verify `REDIS_URL` is set in Variables.

### Issue: Deployment timeout

Some models are large. Increase timeout:
1. Go to Deployment settings
2. Set build timeout to 30 minutes

---

## 💰 Pricing

**Free Tier ($5 credit/month):**
- 500 MB disk
- Limited memory
- Shared CPU

**After Free Credit:**
- PostgreSQL: ~$7/month
- Redis: ~$7/month  
- API service: ~$5/month
- Worker service: ~$5/month
- **Total: ~$24/month**

---

## 📊 Production Settings

For production, update variables:

```env
DEBUG=false
LOG_LEVEL=WARNING
SEARCH_K=50
worker_concurrency=4
```

---

## 🔄 Continuous Deployment

Every time you push to GitHub:
1. Railway detects changes
2. Rebuilds Docker image
3. Runs database migrations
4. Deploys new version
5. Zero-downtime deployment

No manual deploy needed!

---

## 📈 Monitor Performance

In Railway Dashboard:
- **Metrics** tab: CPU, Memory, Network usage
- **Logs** tab: Real-time application logs
- **Deployments** tab: Deployment history

---

## 🆘 Need Help?

- Railway Docs: https://docs.railway.app
- Railway Community: https://discord.gg/railway
- Project Issues: GitHub Issues

