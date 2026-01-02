"""
DeedLens - Property Document Intelligence
Streamlit UI with Typography-First Design
"""

import streamlit as st
import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config must be first
st.set_page_config(
    page_title="DeedLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Colors from Logo
PRIMARY_BLUE = "#0F3D5E"
PRIMARY_ORANGE = "#FF6B35"
BACKGROUND_COLOR = "#FFFFFF"

# Custom CSS for Typography-First Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: white;
    }
    
    /* Remove top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit branding and decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stDecoration"] {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        color: #0F3D5E;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .accent-text {
        color: #FF6B35;
    }
    
    .tagline {
        font-size: 1.3rem;
        font-weight: 400;
        color: #666;
        margin-bottom: 2rem;
        max-width: 500px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .section-subheader {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    /* Cards */
    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #eee;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0F3D5E;
    }
    
    .card-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Entity Tags */
    .entity-tag {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.25rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #eee;
    }
    
    .entity-person {
        background: #F0F9FF;
        color: #0F3D5E;
        border-color: #BAE6FD;
    }
    
    .entity-location {
        background: #ECFDF5;
        color: #047857;
        border-color: #A7F3D0;
    }
    
    .entity-money {
        background: #FFF7ED;
        color: #FF6B35;
        border-color: #FED7AA;
    }
    
    .entity-date {
        background: #FDF4FF;
        color: #86198F;
        border-color: #F0ABFC;
    }
    
    .entity-area {
        background: #DBEAFE;
        color: #1E40AF;
    }
    
    .entity-property {
        background: #E5E7EB;
        color: #374151;
    }
    
    /* Buttons */
    .stButton > button {
        background: #0F3D5E !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #FF6B35 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.2) !important;
    }

    /* Top Nav Button Styling */
    .nav-btn > button {
        background: transparent !important;
        color: #666 !important;
        box-shadow: none !important;
    }
    .nav-btn > button:hover {
        background: transparent !important;
        color: #0F3D5E !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .nav-btn-active > button {
        background: transparent !important;
        color: #0F3D5E !important;
        font-weight: 700 !important;
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed #4F46E5;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(79, 70, 229, 0.03);
        margin: 2rem 0;
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Search Box */
    .stTextInput > div > div > input {
        border-radius: 50px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1rem !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.2s !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1a1a1a !important;
    }
    
    section[data-testid="stSidebar"] {
        background: #1a1a1a;
        padding: 2rem 1rem;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 2rem;
    }
    
    .nav-item {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
        color: #999;
        font-weight: 500;
    }
    
    .nav-item:hover, .nav-item.active {
        background: rgba(79, 70, 229, 0.2);
        color: white;
    }
    
    /* Results */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4F46E5;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .result-title {
        font-weight: 700;
        font-size: 1.2rem;
        color: #1a1a1a;
    }
    
    .result-preview {
        color: #666;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    .result-score {
        font-weight: 700;
        color: #4F46E5;
    }
    
    /* Report Section */
    .report-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .report-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a1a;
        border-bottom: 2px solid #4F46E5;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Geometric Background Pattern */
    .bg-pattern {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        opacity: 0.02;
        z-index: -1;
        background-image: radial-gradient(#0F3D5E 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Stats Row */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-box {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #4F46E5;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []


def render_top_nav():
    """Render the custom top navigation bar."""
    st.markdown('<div class="bg-pattern"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
    
    with col1:
        # Logo Area
        col_logo, col_text = st.columns([1, 4])
        with col_logo:
            st.image("assets/logo.png", width=50)
        with col_text:
            st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
                    <span style="font-size: 1.5rem; font-weight: 800; color: #0F3D5E;">Deed</span>
                    <span style="font-size: 1.5rem; font-weight: 800; color: #FF6B35;">Lens</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Navigation Links
    pages = {
        'home': 'Home',
        'upload': 'Upload',
        'search': 'Search',
        'documents': 'Documents',
        'reports': 'Reports'
    }
    
    cols = [col2, col3, col4, col5, col6]
    
    for idx, (page_id, page_name) in enumerate(pages.items()):
        if idx < len(cols):
            with cols[idx]:
                # Use a unique key for each button to avoid conflicts
                if st.button(page_name, key=f"nav_top_{page_id}", use_container_width=True):
                    st.session_state.page = page_id
                    st.rerun()
    
    st.markdown("---")


def get_img_as_base64(file_path):
    """Convert an image to base64 string."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        # st.error(f"Failed to load {file_path}: {e}") # debug
        return None


def render_home():
    """Render the home page."""
    st.markdown("""
        <div class="main-header">
            From Scanned Deeds<br>
            to <span class="accent-text">Structured Intelligence</span>
        </div>
        <p class="tagline">
            Extract, understand, and search property documents with AI-powered precision.
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Robust absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    
    cards = [
        {
            "title": "OCR Extraction",
            "desc": "Extract text from scanned PDFs and images with high accuracy.",
            "img": os.path.join(assets_dir, "ocr_icon.png"),
            "fallback": "📄"
        },
        {
            "title": "Entity Recognition",
            "desc": "Identify owners, locations, values, and dates automatically.",
            "img": os.path.join(assets_dir, "entity_icon.png"),
            "fallback": "🏷️"
        },
        {
            "title": "Semantic Search",
            "desc": "Find documents by meaning, not just keywords.",
            "img": os.path.join(assets_dir, "logo.png"), # Fallback to logo as search icon is missing
            "fallback": "🔍"
        }
    ]
    
    for i, col in enumerate([col1, col2, col3]):
        with col:
            card = cards[i]
            img_b64 = get_img_as_base64(card['img'])
            
            # Use a slightly larger width for better visibility
            img_html = f'<img src="data:image/png;base64,{img_b64}" width="100" style="margin-bottom: 1.5rem; border-radius: 12px;">' if img_b64 else f'<div style="font-size: 3rem; margin-bottom: 1rem;">{card["fallback"]}</div>'
            
            st.markdown(f"""
                <div class="custom-card" style="text-align: center; height: 100%; padding: 2.5rem 1.5rem;">
                    <div style="display: flex; justify-content: center; align-items: center;">
                        {img_html}
                    </div>
                    <div class="card-title">{card['title']}</div>
                    <p style="color: #666; font-size: 0.95rem; line-height: 1.5;">{card['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Get Started →", key="get_started"):
        st.session_state.page = 'upload'
        st.rerun()


def render_upload():
    """Render the upload page."""
    st.markdown('<div class="section-header">Upload Documents</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Drag and drop your property documents for AI analysis</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
        help="Upload PDF or image files"
    )
    
    if uploaded_file:
        st.markdown("---")
        
        with st.spinner("🔍 Processing document..."):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                # Process document
                result = process_document(tmp_path, uploaded_file.name)
                
                if result:
                    st.session_state.documents.append(result)
                    st.session_state.current_doc = result
                    
                    st.success("✅ Document processed successfully!")
                    
                    # Show results
                    display_document_results(result)
            
            except Exception as e:
                import traceback
                st.error(f"Error processing document: {str(e)}")
                st.code(traceback.format_exc())
            
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


def process_document(file_path: str, file_name: str) -> dict:
    """Process an uploaded document."""
    try:
        # Try to import our modules
        from ocr.ocr_engine import extract_text_from_document
        from preprocessing.text_cleaner import clean_ocr_text
        from nlp.ner_model import extract_entities
    except ImportError:
        # Fallback with mock data
        return {
            'id': f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'name': file_name,
            'text': "Sample extracted text from document...",
            'entities': {
                'PERSON': [{'text': 'Ramesh Kumar'}, {'text': 'Priya Sharma'}],
                'LOCATION': [{'text': 'Indiranagar, Bangalore'}],
                'MONEY': [{'text': 'Rs. 1,50,00,000/-'}],
                'AREA': [{'text': '2400 Sq.Ft.'}],
                'DATE': [{'text': '15th January, 2024'}],
                'PROPERTY_ID': [{'text': 'Survey No. 123/4'}]
            },
            'processed_at': datetime.now().isoformat()
        }
    
    # Extract text
    backend = os.getenv("OCR_BACKEND", "tesseract")
    ocr_result = extract_text_from_document(file_path, backend=backend)
    raw_text = ocr_result['text']
    
    # Clean text
    cleaned = clean_ocr_text(raw_text)
    
    # Extract entities
    entities = extract_entities(cleaned['cleaned_text'])
    
    return {
        'id': f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'name': file_name,
        'text': cleaned['cleaned_text'],
        'entities': entities.get('entities', {}),
        'ocr_confidence': ocr_result.get('confidence', 0),
        'processed_at': datetime.now().isoformat()
    }


def display_document_results(doc: dict):
    """Display results for a processed document."""
    st.markdown('<div class="section-header">Extracted Information</div>', unsafe_allow_html=True)
    
    # Entity cards
    entities = doc.get('entities', {})
    
    cols = st.columns(3)
    
    entity_styles = {
        'PERSON': 'entity-person',
        'LOCATION': 'entity-location',
        'MONEY': 'entity-money',
        'DATE': 'entity-date',
        'AREA': 'entity-area',
        'PROPERTY_ID': 'entity-property'
    }
    
    col_idx = 0
    for entity_type, entity_list in entities.items():
        if entity_list:
            with cols[col_idx % 3]:
                style_class = entity_styles.get(entity_type, 'entity-property')
                st.markdown(f"**{entity_type.replace('_', ' ')}**")
                
                for e in entity_list[:5]:
                    text = e.get('text', e.get('canonical', str(e))) if isinstance(e, dict) else str(e)
                    st.markdown(f'<span class="entity-tag {style_class}">{text}</span>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            col_idx += 1
    
    # Extracted text
    with st.expander("📝 View Extracted Text"):
        st.text_area("Full Text", doc.get('text', ''), height=300, disabled=True)


def render_search():
    """Render the search page."""
    st.markdown('<div class="section-header">Search Documents</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Find property documents using natural language</p>', unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "Search",
        placeholder="e.g., 'Sale deeds in Indiranagar above 1 crore'",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_mode = st.radio(
            "Search Mode",
            ["Hybrid", "Semantic", "Keyword"],
            horizontal=True
        )
    
    if st.button("Search", key="search_btn"):
        if query:
            with st.spinner("Searching..."):
                results = search_documents(query, search_mode.lower())
                st.session_state.search_results = results
    
    # Display results
    if st.session_state.search_results:
        st.markdown("---")
        st.markdown(f"**Found {len(st.session_state.search_results)} results**")
        
        for result in st.session_state.search_results:
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{result.get('title', 'Document')}</div>
                    <div class="result-preview">{result.get('preview', '')[:200]}...</div>
                    <div style="margin-top: 0.5rem;">
                        <span class="result-score">Score: {result.get('score', 0):.2f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)


def search_documents(query: str, mode: str) -> list:
    """Search documents."""
    # Mock search results for now
    if not st.session_state.documents:
        return []
    
    results = []
    for doc in st.session_state.documents:
        if query.lower() in doc.get('text', '').lower() or query.lower() in doc.get('name', '').lower():
            results.append({
                'id': doc['id'],
                'title': doc['name'],
                'preview': doc.get('text', '')[:200],
                'score': 0.95,
                'entities': doc.get('entities', {})
            })
    
    return results


def render_documents():
    """Render the documents list page."""
    st.markdown('<div class="section-header">Documents</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">All processed property documents</p>', unsafe_allow_html=True)
    
    if not st.session_state.documents:
        st.info("No documents uploaded yet. Go to Upload to add documents.")
        return
    
    for doc in st.session_state.documents:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">📄 {doc['name']}</div>
                        <div class="result-preview">{doc.get('text', '')[:150]}...</div>
                        <div style="margin-top: 0.5rem; color: #888; font-size: 0.85rem;">
                            Processed: {doc.get('processed_at', 'N/A')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("View", key=f"view_{doc['id']}"):
                    st.session_state.current_doc = doc
                    st.session_state.page = 'reports'
                    st.rerun()


def render_reports():
    """Render the reports page."""
    st.markdown('<div class="section-header">Property Report</div>', unsafe_allow_html=True)
    
    doc = st.session_state.current_doc
    
    if not doc:
        if st.session_state.documents:
            doc = st.session_state.documents[-1]
            st.session_state.current_doc = doc
        else:
            st.info("No document selected. Upload a document first.")
            return
    
    st.markdown(f'<p class="section-subheader">Report for: {doc["name"]}</p>', unsafe_allow_html=True)
    
    # Generate report
    if st.button("Generate AI Report", key="gen_report"):
        with st.spinner("Generating report with AI..."):
            report = generate_report_for_doc(doc)
            st.session_state.current_report = report
    
    # Display report sections
    entities = doc.get('entities', {})
    
    # Property Summary Card
    st.markdown("""
        <div class="report-section">
            <div class="report-title">📋 Property Summary</div>
    """, unsafe_allow_html=True)
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Parties Involved
    persons = entities.get('PERSON', [])
    if persons:
        st.markdown("""
            <div class="report-section">
                <div class="report-title">👥 Parties Involved</div>
        """, unsafe_allow_html=True)
        
        for p in persons:
            text = p.get('text', str(p)) if isinstance(p, dict) else str(p)
            st.markdown(f"• {text}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Property Details
    prop_ids = entities.get('PROPERTY_ID', [])
    if prop_ids:
        st.markdown("""
            <div class="report-section">
                <div class="report-title">🏠 Property Details</div>
        """, unsafe_allow_html=True)
        
        for p in prop_ids:
            text = p.get('text', str(p)) if isinstance(p, dict) else str(p)
            st.markdown(f"• **ID:** {text}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download button
    st.markdown("---")
    report_text = f"""
PROPERTY TRANSACTION REPORT
===========================

Document: {doc['name']}
Processed: {doc.get('processed_at', 'N/A')}

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
        mime="text/plain"
    )


def generate_report_for_doc(doc: dict) -> dict:
    """Generate a report for a document."""
    try:
        from reports.report_generator import generate_report
        return generate_report(doc.get('text', ''), doc.get('entities', {}))
    except ImportError:
        return {"summary": "Report generated", "sections": {}}


# Main App
def main():
    render_top_nav()
    
    page = st.session_state.page
    
    if page == 'home':
        render_home()
    elif page == 'upload':
        render_upload()
    elif page == 'search':
        render_search()
    elif page == 'documents':
        render_documents()
    elif page == 'reports':
        render_reports()


if __name__ == "__main__":
    main()
