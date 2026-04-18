# DeedLens Docker Image
FROM python:3.11-slim

WORKDIR /app

# Install all dependencies (build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libopenblas0 \
    libomp-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install dependencies in order
RUN pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Copy workaround for transformers issue
COPY sitecustomize.py /usr/local/lib/python3.11/site-packages/

# Create data directory
RUN mkdir -p data/index data/uploads

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV OCR_BACKEND=easyocr
ENV LOG_LEVEL=INFO
# Disable HF accelerate to avoid nn import issue in transformers
ENV HF_ACCELERATE_AVAILABLE=0

# Expose ports
EXPOSE 8501 8000

# Default command (Streamlit)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
