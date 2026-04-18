"""
DeedLens - Property Document Intelligence
Streamlit UI with Warm Brutalism Design System
"""

import streamlit as st
import os
import sys
import tempfile
import base64
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Initialize config and logging first
from config import settings
from core.logger import setup_logging, get_logger

setup_logging(level=settings.log_level)
logger = get_logger(__name__)

# Import core service
from core.service import get_document_service, DocumentService

# Page config must be first Streamlit command
st.set_page_config(
    page_title="DeedLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    theme={
        "primaryColor": "#FF6B35",
        "backgroundColor": "#F5F0E8",
        "secondaryBackgroundColor": "#EDE8DF",
        "textColor": "#1A1A1A",
        "font": "sans serif"
    }
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []


# Cached stats wrapper
@st.cache_data(ttl=30)
def get_cached_stats():
    """Get stats with 30-second cache."""
    service = get_service()
    return service.get_stats()


def get_service() -> DocumentService:
    """Get document service."""
    if 'document_service' not in st.session_state:
        st.session_state.document_service = get_document_service()
    return st.session_state.document_service


def get_img_as_base64(file_path):
    """Convert an image to base64 string."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None


# ============================================================================
# WARM BRUTALISM CSS INJECTION - Main Entry Point
# ============================================================================

def inject_css():
    """Inject complete Warm Brutalism CSS using st.markdown()."""
    st.markdown("""
    <style id="brutalism-css">
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

    /* ========== CONTAINER && LAYOUT ========== */
    html {
        background: #F5F0E8 !important;
    }

    body {
        background: #F5F0E8 !important;
    }

    /* Target Streamlit's main wrapper div */
    .stApp {
        background: #F5F0E8 !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #F5F0E8 !important;
    }

    [data-testid="stMainBlockContainer"] {
        background: #F5F0E8 !important;
    }


    /* ========== HIDE STREAMLIT CHROME ========== */
    header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu {
        display: none !important;
    }

    /* ========== FONTS ========== */
    * {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    button, [role="button"] {
        font-family: 'Space Mono', 'Courier New', monospace !important;
    }

    /* ========== BUTTONS - ALL VARIANTS ========== */
    button,
    [data-testid="stButton"] button,
    .stButton button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"],
    [role="button"] {
        border-radius: 0 !important;
        border: 2px solid #1A1A1A !important;
        background-color: #1A1A1A !important;
        color: #F5F0E8 !important;
        font-family: 'Space Mono', 'Courier New', monospace !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        padding: 12px 24px !important;
        cursor: pointer !important;
        transition: all 0.1s ease !important;
    }

    button:hover,
    [data-testid="stButton"] button:hover,
    .stButton button:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    [role="button"]:hover {
        background-color: #FF6B35 !important;
        border-color: #FF6B35 !important;
        color: #1A1A1A !important;
    }

    button:active,
    [data-testid="stButton"] button:active,
    .stButton button:active,
    [data-testid="baseButton-primary"]:active,
    [data-testid="baseButton-secondary"]:active {
        background-color: #E85A1F !important;
        border-color: #E85A1F !important;
    }

    button:focus,
    [data-testid="stButton"] button:focus,
    .stButton button:focus,
    [data-testid="baseButton-primary"]:focus,
    [data-testid="baseButton-secondary"]:focus {
        outline: none !important;
        border-color: #FF6B35 !important;
        box-shadow: none !important;
    }

    /* ========== TEXT INPUTS ========== */
    input[type="text"],
    input[type="password"],
    input[type="email"],
    input[type="number"],
    input[type="search"],
    textarea,
    select,
    [data-testid="textInput"] input,
    .stTextInput input {
        border-radius: 0 !important;
        border: 2px solid #1A1A1A !important;
        background-color: #F5F0E8 !important !important;
        color: #1A1A1A !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 14px !important;
        padding: 14px 18px !important;
    }

    input[type="text"]:focus,
    input[type="password"]:focus,
    input[type="email"]:focus,
    input[type="number"]:focus,
    input[type="search"]:focus,
    textarea:focus,
    select:focus,
    [data-testid="textInput"] input:focus,
    .stTextInput input:focus {
        border-color: #FF6B35 !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploadDropzone"],
    .stFileUploader {
        border-radius: 0 !important;
        border: 2px dashed #1A1A1A !important;
        background-color: #EDE8DF !important;
    }

    /* ========== HEADINGS ========== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #1A1A1A !important;
        font-weight: 700 !important;
        letter-spacing: -1px !important;
    }

    /* ========== RADIO & CHECKBOX ========== */
    input[type="radio"],
    input[type="checkbox"] {
        accent-color: #FF6B35 !important;
        border: 2px solid #1A1A1A !important;
    }

    /* ========== EXPANDERS ========== */
    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        border-radius: 0 !important;
        border: 2px solid #1A1A1A !important;
        background-color: #1A1A1A !important;
        color: #F5F0E8 !important;
        padding: 12px 16px !important;
    }

    [data-testid="stExpander"] summary:hover {
        background-color: #FF6B35 !important;
        color: #1A1A1A !important;
    }

    /* ========== TABS ========== */
    [data-testid="stTabs"] button {
        border-radius: 0 !important;
        border: 2px solid #1A1A1A !important;
        background-color: transparent !important;
        color: #1A1A1A !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #1A1A1A !important;
        color: #F5F0E8 !important;
        border-color: #1A1A1A !important;
    }

    /* ========== ALERTS && MESSAGES ========== */
    [data-testid="stAlert"] {
        border-radius: 0 !important;
        border: 2px solid #1A1A1A !important;
    }

    /* ========== HORIZONTAL DIVIDER ========== */
    hr {
        border: none !important;
        border-bottom: 2px solid #1A1A1A !important;
        margin: 2rem 0 !important;
    }

    /* ========== CUSTOM CLASSES (HTML components) ========== */
    .wb-nav-bar { background: #1A1A1A !important; border-bottom: 3px solid #FF6B35 !important; }
    .wb-nav-item { background: #1A1A1A !important; color: #888888 !important; border: 2px solid #333333 !important; }
    .wb-nav-item:hover { background: #FF6B35 !important; color: #1A1A1A !important; border-color: #FF6B35 !important; }
    .wb-nav-item.active { background: #FF6B35 !important; color: #1A1A1A !important; border-color: #FF6B35 !important; }
    .wb-section-header { border-bottom: 2px solid #1A1A1A !important; color: #1A1A1A !important; }
    .wb-card { border: 2px solid #1A1A1A !important; border-left: 4px solid #FF6B35 !important; }
    .wb-entity-tag { border: 2px solid #1A1A1A !important; color: #1A1A1A !important; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# NAVIGATION
# ============================================================================

def render_top_nav():
    """Render the Warm Brutalism top navigation bar with clickable nav items."""

    # Create JavaScript to handle navigation clicks
    nav_html = """
    <style>
    .wb-nav-bar {
        background: #1A1A1A;
        border-bottom: 3px solid #FF6B35;
        padding: 0 1rem;
        display: flex;
        align-items: center;
        height: 56px;
        margin: -1rem -1rem 1rem -1rem;
        justify-content: space-between;
    }

    .wb-nav-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 18px;
        display: flex;
        gap: 0;
    }

    .wb-nav-logo-deed { color: #F5F0E8; }
    .wb-nav-logo-lens { color: #FF6B35; }

    .wb-nav-links {
        display: flex;
        gap: 0;
    }

    .wb-nav-item {
        padding: 12px 24px;
        border: 2px solid #1A1A1A;
        background: #1A1A1A;
        color: #888888;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
        transition: all 0.1s;
        border-right: 2px solid #333333;
    }

    .wb-nav-item:last-child {
        border-right: none;
    }

    .wb-nav-item:hover {
        background: #FF6B35;
        color: #1A1A1A;
        border-color: #FF6B35;
    }

    .wb-nav-item.active {
        background: #FF6B35;
        color: #1A1A1A;
        border-color: #FF6B35;
    }
    </style>

    <div class="wb-nav-bar">
        <div class="wb-nav-logo">
            <span class="wb-nav-logo-deed">Deed</span><span class="wb-nav-logo-lens">Lens</span>
        </div>
        <div class="wb-nav-links">
            <div class="wb-nav-item {}" onclick="window.location.href='?page=home'">HOME</div>
            <div class="wb-nav-item {}" onclick="window.location.href='?page=upload'">UPLOAD</div>
            <div class="wb-nav-item {}" onclick="window.location.href='?page=search'">SEARCH</div>
            <div class="wb-nav-item {}" onclick="window.location.href='?page=documents'">DOCUMENTS</div>
            <div class="wb-nav-item {}" onclick="window.location.href='?page=jobs'">JOBS</div>
        </div>
    </div>
    """.format(
        'active' if st.session_state.page == 'home' else '',
        'active' if st.session_state.page == 'upload' else '',
        'active' if st.session_state.page == 'search' else '',
        'active' if st.session_state.page == 'documents' else '',
        'active' if st.session_state.page == 'jobs' else '',
    )

    st.markdown(nav_html, unsafe_allow_html=True)

    # Handle query params for navigation
    query_params = st.query_params
    if 'page' in query_params:
        page = query_params['page']
        if page in ['home', 'upload', 'search', 'documents', 'jobs', 'reports']:
            st.session_state.page = page
            st.query_params.clear()
            st.rerun()


# ============================================================================
# HOME PAGE
# ============================================================================

def render_home():
    """Render the home page with Warm Brutalism hero and stats."""
    stats = get_cached_stats()
    total_entities = sum(stats['entity_counts'].values())

    # Two-column hero
    st.markdown("""
    <div class="wb-hero">
        <div class="wb-hero-text">
            <h1>From Scanned Deeds<br>to <span class="accent">Structured Intelligence</span></h1>
            <p class="wb-hero-tagline">Extract, understand, and search property documents with AI-powered precision.</p>
        </div>
        <div class="wb-stats-strip">
            <div class="wb-stat-cell">
                <div class="wb-stat-label">Total Documents</div>
                <div class="wb-stat-value">{}</div>
            </div>
            <div class="wb-stat-cell">
                <div class="wb-stat-label">Total Entities</div>
                <div class="wb-stat-value orange">{}</div>
            </div>
            <div class="wb-stat-cell">
                <div class="wb-stat-label">OCR Backend</div>
                <div class="wb-stat-value">{}</div>
            </div>
        </div>
    </div>
    """.format(
        stats['total_documents'],
        total_entities,
        settings.ocr_backend.upper()
    ), unsafe_allow_html=True)

    # Feature cards
    st.markdown('<div class="wb-feature-cards">', unsafe_allow_html=True)

    features = [
        ("01 / OCR", "Extraction", "Extract text from scanned PDFs and images with high accuracy."),
        ("02 / NER", "Entity Recognition", "Identify owners, locations, values, and dates automatically."),
        ("03 / Search", "Semantic Search", "Find documents by meaning, not just keywords.")
    ]

    for idx, (label, title, desc) in enumerate(features):
        st.markdown(f"""
        <div class="wb-card">
            <div class="wb-card-index">{label}</div>
            <div class="wb-card-title">{title}</div>
            <p class="wb-card-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        if st.button("Initialize →", key="get_started", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()


# ============================================================================
# UPLOAD PAGE
# ============================================================================

def render_upload():
    """Render the upload page with flat toggle and dashed drop zone."""
    import uuid
    from core.jobs import get_job_manager

    service = get_service()

    st.markdown('<div class="wb-section-header">Upload Documents</div>', unsafe_allow_html=True)

    # Stats strip
    stats = get_cached_stats()
    total_entities = sum(stats['entity_counts'].values())

    st.markdown(f"""
    <div class="wb-stats-strip">
        <div class="wb-stat-cell">
            <div class="wb-stat-label">Total Documents</div>
            <div class="wb-stat-value">{stats['total_documents']}</div>
        </div>
        <div class="wb-stat-cell">
            <div class="wb-stat-label">Total Entities</div>
            <div class="wb-stat-value orange">{total_entities}</div>
        </div>
        <div class="wb-stat-cell">
            <div class="wb-stat-label">OCR Backend</div>
            <div class="wb-stat-value">{settings.ocr_backend.upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    # Processing mode toggle
    processing_mode = st.radio(
        "Processing Mode",
        ["Synchronous", "Asynchronous"],
        horizontal=True,
        help="Sync: Get results immediately. Async: Queue for background processing.",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Property Document",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

        if "Synchronous" in processing_mode:
            # Sync processing
            with st.spinner("Processing document with AI..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    result = service.process_document(tmp_path, uploaded_file.name)

                    if result.success and result.document:
                        st.session_state.current_doc = result.document.to_dict()

                        st.success(f"Document processed in {result.processing_time:.2f}s!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("OCR Confidence", f"{result.ocr_confidence:.1%}")
                        with col2:
                            st.metric("Processing Time", f"{result.processing_time:.2f}s")

                        display_document_results(result.document.to_dict())
                    else:
                        st.error(f"Processing failed: {result.error}")

                except Exception as e:
                    logger.error(f"Upload error: {e}", exc_info=True)
                    st.error(f"Error processing document: {str(e)}")

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        else:
            # Async processing
            try:
                job_manager = get_job_manager()

                upload_dir = settings.data_dir / "uploads" if settings.data_dir else Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)

                file_id = str(uuid.uuid4())
                file_ext = Path(uploaded_file.name).suffix
                permanent_path = upload_dir / f"{file_id}{file_ext}"

                with open(permanent_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                job = job_manager.create_job(
                    original_filename=uploaded_file.name,
                    file_path=str(permanent_path)
                )

                try:
                    import os
                    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                    logger.info(f"Queuing job {job.id} with REDIS_URL={redis_url}")

                    from worker.celery_app import celery_app
                    from worker.tasks import process_document_task

                    logger.info(f"Celery broker: {celery_app.conf.broker_url}")

                    result = process_document_task.delay(job.id)
                    logger.info(f"Task queued with ID: {result.id}")

                    st.success(f"Job queued for background processing!")
                    st.info(f"Job ID: `{job.id}`")
                    st.caption(f"Task ID: `{result.id}`")
                    st.markdown("Go to **Jobs** page to monitor progress.")
                except ImportError as e:
                    logger.error(f"Celery import error: {e}")
                    st.warning(f"Celery not installed: {e}")
                    st.info(f"Job ID: `{job.id}`")
                except Exception as e:
                    logger.error(f"Could not queue job: {e}", exc_info=True)
                    st.warning(f"Could not queue job (Celery/Redis not running?): {e}")
                    st.info(f"Job created with ID: `{job.id}`. Start Celery worker to process.")

            except Exception as e:
                logger.error(f"Async upload error: {e}", exc_info=True)
                st.error(f"Error creating job: {str(e)}")


def display_document_results(doc: dict):
    """Display results for a processed document."""
    st.markdown('<div class="wb-section-header">Extracted Information</div>', unsafe_allow_html=True)

    entities = doc.get('entities', {})

    entity_type_map = {
        'PERSON': 'person',
        'LOCATION': 'location',
        'MONEY': 'money',
        'DATE': 'date',
        'AREA': 'area',
        'PROPERTY_ID': 'property'
    }

    cols = st.columns(3)
    col_idx = 0

    for entity_type, entity_list in entities.items():
        if entity_list:
            with cols[col_idx % 3]:
                st.markdown(f"**{entity_type.replace('_', ' ')}**")

                for e in entity_list[:5]:
                    text = e.get('text', e.get('canonical', str(e))) if isinstance(e, dict) else str(e)
                    tag_class = entity_type_map.get(entity_type, 'property')
                    st.markdown(f'<span class="wb-entity-tag {tag_class}">[{text}]</span>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            col_idx += 1

    with st.expander("📝 View Extracted Text"):
        st.text_area("Full Text", doc.get('text', ''), height=300, disabled=True, label_visibility="collapsed")


# ============================================================================
# SEARCH PAGE
# ============================================================================

def render_search():
    """Render the search page with joined input+button bar."""
    service = get_service()

    st.markdown('<div class="wb-section-header">Search Documents</div>', unsafe_allow_html=True)

    stats = get_cached_stats()
    if stats['total_documents'] == 0:
        st.warning("⚠️ No documents indexed yet. Upload some documents first!")
        return

    st.markdown(f'<div class="wb-info-bar">🔍 Searching across <strong>{stats["total_documents"]}</strong> documents</div>', unsafe_allow_html=True)

    # Search bar with joined input + button
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input(
            "Search",
            placeholder="e.g., 'Sale deeds in Indiranagar above 1 crore'",
            label_visibility="collapsed",
            key="search_input"
        )

    with col2:
        search_mode = st.radio(
            "Mode",
            ["Hybrid", "Semantic", "Keyword"],
            horizontal=True,
            label_visibility="collapsed",
            key="search_mode"
        )

    with col3:
        num_results = st.slider("Results", 5, 50, 10, label_visibility="collapsed")

    col_search, _ = st.columns([1, 4])
    with col_search:
        if st.button("Search →", key="search_btn", use_container_width=True):
            if query:
                with st.spinner("Searching with AI..."):
                    results = service.search(query, mode=search_mode.lower(), k=num_results)
                    st.session_state.search_results = [
                        {
                            'doc_id': r.doc_id,
                            'title': r.title,
                            'preview': r.preview,
                            'score': r.score,
                            'semantic_score': r.semantic_score,
                            'keyword_score': r.keyword_score,
                            'entities': r.entities
                        }
                        for r in results
                    ]

    # Display results
    if st.session_state.search_results:
        st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"### Found {len(st.session_state.search_results)} results")

        for i, result in enumerate(st.session_state.search_results):
            col1, col2 = st.columns([4, 1])

            with col1:
                entities = result.get('entities', {})
                entity_tags = ""
                for etype, elist in list(entities.items())[:3]:
                    if elist:
                        text = elist[0].get('text', str(elist[0])) if isinstance(elist[0], dict) else str(elist[0])
                        entity_tags += f'<span class="wb-entity-tag">[{text}]</span>'

                st.markdown(f"""
                <div class="wb-bordered-row">
                    <div style="font-weight: 700; margin-bottom: 0.5rem;">📄 {result.get('title', 'Document')}</div>
                    <div style="font-size: 13px; color: #666; margin-bottom: 0.5rem;">{result.get('preview', '')[:250]}...</div>
                    <div style="margin-bottom: 0.5rem;">{entity_tags}</div>
                    <div style="font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700;">
                        Score: {result.get('score', 0):.3f} | Semantic: {result.get('semantic_score', 0):.2f} | Keyword: {result.get('keyword_score', 0):.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("View", key=f"view_search_{result['doc_id']}_{i}", use_container_width=True):
                    doc = service.get_document(result['doc_id'])
                    if doc:
                        st.session_state.current_doc = doc.to_dict()
                        st.session_state.page = 'reports'
                        st.rerun()


# ============================================================================
# DOCUMENTS PAGE
# ============================================================================

def render_documents():
    """Render the documents page with bordered rows."""
    service = get_service()

    st.markdown('<div class="wb-section-header">Documents</div>', unsafe_allow_html=True)

    documents = service.get_all_documents(limit=100)

    if not documents:
        st.info("No documents uploaded yet. Go to Upload to add documents.")
        return

    st.markdown(f"**{len(documents)} documents in database**")
    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    for idx, doc in enumerate(documents):
        row_class = "even" if idx % 2 == 0 else ""
        doc_type_badge = doc.doc_type or "Document"

        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            created_date = (doc.created_at[:10] if isinstance(doc.created_at, str) else str(doc.created_at)[:10]) if doc.created_at else 'N/A'

            st.markdown(f"""
            <div class="wb-bordered-row {row_class}">
                <div style="font-weight: 700; margin-bottom: 0.25rem;">📄 {doc.name} <span style="background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 0; font-size: 0.8rem; font-family: 'Space Mono', monospace;">{doc_type_badge}</span></div>
                <div style="font-size: 13px; color: #666; margin-bottom: 0.5rem;">{(doc.text or '')[:150]}...</div>
                <div style="font-family: 'Space Mono', monospace; font-size: 10px; color: var(--wb-muted);">
                    OCR Confidence: {doc.ocr_confidence:.1%} · Created: {created_date}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if st.button("View", key=f"view_{doc.id}", use_container_width=True):
                st.session_state.current_doc = doc.to_dict()
                st.session_state.page = 'reports'
                st.rerun()

        with col3:
            if st.button("✕", key=f"del_{doc.id}", use_container_width=True):
                if service.delete_document(doc.id):
                    st.success(f"Deleted {doc.name}")
                    st.rerun()


# ============================================================================
# JOBS PAGE
# ============================================================================

def render_jobs():
    """Render the jobs page with dark table header and metrics."""
    from core.jobs import get_job_manager, JobStatus
    import requests

    st.markdown('<div class="wb-section-header">Processing Jobs</div>', unsafe_allow_html=True)

    job_manager = get_job_manager()

    # Metrics summary
    try:
        metrics = job_manager.get_metrics_summary()

        st.markdown(f"""
        <div class="wb-stats-strip">
            <div class="wb-stat-cell">
                <div class="wb-stat-label">Total Jobs</div>
                <div class="wb-stat-value">{metrics['total_jobs']}</div>
            </div>
            <div class="wb-stat-cell">
                <div class="wb-stat-label">Pending</div>
                <div class="wb-stat-value" style="color: #BA7517;">{metrics['pending']}</div>
            </div>
            <div class="wb-stat-cell">
                <div class="wb-stat-label">Processing</div>
                <div class="wb-stat-value" style="color: #185FA5;">{metrics['processing']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completed", metrics['completed'])
        with col2:
            st.metric("Failed", metrics['failed'])
        with col3:
            if metrics['avg_duration_ms']:
                st.metric("Avg Duration", f"{metrics['avg_duration_ms']:.0f}ms")

    except Exception as e:
        st.warning(f"Could not load metrics: {e}")

    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    # Status filter + refresh
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        status_filter = st.selectbox(
            "Filter",
            ["All", "pending", "processing", "completed", "failed"],
            label_visibility="collapsed"
        )

    with col2:
        if st.button("Refresh", key="refresh_jobs", use_container_width=True):
            st.rerun()

    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    # Job table
    try:
        if status_filter == "All":
            jobs = job_manager.get_all_jobs(limit=50)
        else:
            jobs = job_manager.get_jobs_by_status(JobStatus(status_filter), limit=50)

        if not jobs:
            st.info("No jobs found. Upload documents to create processing jobs.")
        else:
            # HTML table header
            st.markdown("""
            <div class="wb-table-header" style="grid-template-columns: 1fr 1fr 0.5fr 0.8fr 1fr;">
                <div>#</div>
                <div>Filename</div>
                <div>Status</div>
                <div>Progress</div>
                <div>Created</div>
            </div>
            """, unsafe_allow_html=True)

            for idx, job in enumerate(jobs):
                status = job.status.value if hasattr(job.status, 'value') else job.status
                status_colors = {
                    'pending': '#BA7517',
                    'processing': '#185FA5',
                    'completed': '#3B6D11',
                    'failed': '#A32D2D',
                    'retrying': '#FF6B35'
                }
                status_color = status_colors.get(status, '#888888')

                created_str = job.created_at[:19] if isinstance(job.created_at, str) else str(job.created_at)[:19]

                row_class = "even" if idx % 2 == 0 else ""

                col1, col2, col3, col4, col5 = st.columns([1, 1, 0.5, 0.8, 1])

                with col1:
                    st.markdown(f'<div class="wb-bordered-row {row_class}" style="display: flex; align-items: center; height: 100%;">{idx + 1}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'<div class="wb-bordered-row {row_class}">{job.original_filename}</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown(f'<div class="wb-bordered-row {row_class}"><span style="color: {status_color};">●</span> {status.upper()}</div>', unsafe_allow_html=True)

                with col4:
                    st.markdown(f'<div class="wb-bordered-row {row_class}">{job.progress}%</div>', unsafe_allow_html=True)

                with col5:
                    st.markdown(f'<div class="wb-bordered-row {row_class}">{created_str}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading jobs: {e}")

    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    # Pipeline metrics expander
    with st.expander("Pipeline Metrics", expanded=False):
        try:
            metrics_response = requests.get('http://localhost:8000/metrics')
            if metrics_response.status_code == 200:
                metrics_data = metrics_response.json()

                # Render as bordered table
                st.markdown("""
                <div class="wb-table-header" style="grid-template-columns: 1fr 1fr 1fr 1fr;">
                    <div>Stage</div>
                    <div>Avg (ms)</div>
                    <div>P95 (ms)</div>
                    <div>Failure %</div>
                </div>
                """, unsafe_allow_html=True)

                for stage, data in metrics_data.items():
                    avg_ms = data.get('avg_duration_ms', 0)
                    p95_ms = data.get('p95_duration_ms', 0)
                    failure_pct = data.get('failure_rate', 0)

                    st.markdown(f"""
                    <div class="wb-bordered-row" style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; padding: 1rem;">
                        <div style="font-weight: 700;">{stage.upper()}</div>
                        <div>{avg_ms:.0f}</div>
                        <div>{p95_ms:.0f}</div>
                        <div>{failure_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not fetch pipeline metrics")
        except Exception as e:
            st.warning(f"Pipeline metrics unavailable: {e}")


# ============================================================================
# REPORTS PAGE
# ============================================================================

def render_reports():
    """Render the reports page with bordered layout."""
    service = get_service()

    st.markdown('<div class="wb-section-header">Property Report</div>', unsafe_allow_html=True)

    doc = st.session_state.current_doc

    if not doc:
        documents = service.get_all_documents(limit=1)
        if documents:
            doc = documents[0].to_dict()
            st.session_state.current_doc = doc
        else:
            st.info("No document selected. Upload a document first.")
            return

    st.markdown(f'<p style="color: var(--wb-muted); font-family: \'Space Mono\', monospace; font-size: 11px; text-transform: uppercase; margin-bottom: 1rem;">Report for: {doc["name"]}</p>', unsafe_allow_html=True)

    if st.button("Generate AI Report", key="gen_report", use_container_width=False):
        with st.spinner("Generating report with AI..."):
            report = service.generate_report(doc['id'])
            st.session_state.current_report = report
            if 'error' not in report:
                st.success("Report generated!")

    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    entities = doc.get('entities', {})

    # Property Summary
    st.markdown('<div style="border: 2px solid var(--wb-ink); padding: 1.5rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; margin-bottom: 1rem; font-size: 16px;">📋 Property Summary</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        locations = entities.get('LOCATION', [])
        loc_text = locations[0].get('text', 'N/A') if locations else 'N/A'
        st.markdown(f"**Location:** {loc_text}")

        areas = entities.get('AREA', [])
        area_text = areas[0].get('text', 'N/A') if areas else 'N/A'
        st.markdown(f"**Area:** {area_text}")

    with col2:
        money = entities.get('MONEY', [])
        money_text = money[0].get('text', 'N/A') if money else 'N/A'
        st.markdown(f"**Transaction Value:** {money_text}")

        dates = entities.get('DATE', [])
        date_text = dates[0].get('text', 'N/A') if dates else 'N/A'
        st.markdown(f"**Date:** {date_text}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Parties Involved
    persons = entities.get('PERSON', [])
    if persons:
        st.markdown('<div style="border: 2px solid var(--wb-ink); padding: 1.5rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 700; margin-bottom: 1rem; font-size: 16px;">👥 Parties Involved</div>', unsafe_allow_html=True)

        for p in persons:
            text = p.get('text', str(p)) if isinstance(p, dict) else str(p)
            st.markdown(f"• {text}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Property Details
    prop_ids = entities.get('PROPERTY_ID', [])
    if prop_ids:
        st.markdown('<div style="border: 2px solid var(--wb-ink); padding: 1.5rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 700; margin-bottom: 1rem; font-size: 16px;">🏠 Property Details</div>', unsafe_allow_html=True)

        for p in prop_ids:
            text = p.get('text', str(p)) if isinstance(p, dict) else str(p)
            st.markdown(f"• **ID:** {text}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Download report
    st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

    report_text = f"""
PROPERTY TRANSACTION REPORT
===========================

Document: {doc['name']}
Document Type: {doc.get('doc_type', 'Unknown')}
Created: {doc.get('created_at', 'N/A')}

PROPERTY SUMMARY
----------------
Location: {entities.get('LOCATION', [{}])[0].get('text', 'N/A') if entities.get('LOCATION') else 'N/A'}
Area: {entities.get('AREA', [{}])[0].get('text', 'N/A') if entities.get('AREA') else 'N/A'}
Value: {entities.get('MONEY', [{}])[0].get('text', 'N/A') if entities.get('MONEY') else 'N/A'}
Date: {entities.get('DATE', [{}])[0].get('text', 'N/A') if entities.get('DATE') else 'N/A'}

PARTIES INVOLVED
----------------
{chr(10).join(['• ' + (p.get('text', str(p)) if isinstance(p, dict) else str(p)) for p in entities.get('PERSON', [])])}

EXTRACTED TEXT
--------------
{doc.get('text', '')[:1000]}...
"""

    st.download_button(
        "📥 Download Report",
        report_text,
        file_name=f"report_{doc['id']}.txt",
        mime="text/plain",
        use_container_width=False
    )


# ============================================================================
# MAIN APP ENTRY POINT
# ============================================================================

def main():
    """Main app entry point."""
    # Render navigation FIRST
    render_top_nav()

    # Route to pages
    page = st.session_state.page

    if page == 'home':
        render_home()
    elif page == 'upload':
        render_upload()
    elif page == 'search':
        render_search()
    elif page == 'documents':
        render_documents()
    elif page == 'jobs':
        render_jobs()
    elif page == 'reports':
        render_reports()

    # ⚠️ INJECT CSS LAST - After all components are rendered
    # This ensures our CSS overrides Streamlit's defaults
    inject_css()



if __name__ == "__main__":
    main()
