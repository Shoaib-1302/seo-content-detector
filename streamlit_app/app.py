"""
SEO Content Quality & Duplicate Detector - Streamlit App
Main application file
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import utility modules
from utils.parser import scrape_and_parse_url, parse_html_content
from utils.features import extract_features, compute_similarity
from utils.scorer import predict_quality, load_model
from utils.advanced_nlp import (
    extract_sentiment, 
    extract_entities, 
    extract_topics,
    generate_word_cloud
)

# Page configuration
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quality-high {
        color: #28a745;
        font-weight: bold;
    }
    .quality-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .quality-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load model and data
@st.cache_resource
def initialize_app():
    """Load model and existing dataset."""
    try:
        model_data = load_model('streamlit_app/models/quality_model.pkl')
        features_df = pd.read_csv('data/features.csv')
        extracted_df = pd.read_csv('data/extracted_content.csv')

        # Parse embeddings safely
        if 'embedding' in features_df.columns:
            def safe_eval(x):
                try:
                    if pd.isna(x) or str(x).lower() in ["", "nan", "none"]:
                        return []
                    return eval(x)
                except Exception:
                    return []
            features_df['embedding'] = features_df['embedding'].apply(safe_eval)
        else:
            features_df['embedding'] = [[] for _ in range(len(features_df))]

        return model_data, features_df, extracted_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Main app
def main():
    st.markdown('<p class="main-header">🔍 SEO Content Quality & Duplicate Detector</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio("Select Page", [
            "🏠 Home",
            "🔎 Analyze URL",
            "📈 Dashboard",
            "🔄 Duplicate Detection",
            "📊 Visualizations"
        ])
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This tool analyzes web content for:
        - SEO quality scoring
        - Duplicate content detection
        - Readability metrics
        - Sentiment analysis
        - Entity extraction
        - Topic modeling
        """)
    
    # Load data
    model_data, features_df, extracted_df = initialize_app()
    
    if model_data is None:
        st.error("Failed to load required data. Please ensure all files are present.")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔎 Analyze URL":
        show_analyze_page(model_data, features_df)
    elif page == "📈 Dashboard":
        show_dashboard_page(features_df, extracted_df)
    elif page == "🔄 Duplicate Detection":
        show_duplicates_page(features_df)
    elif page == "📊 Visualizations":
        show_visualizations_page(features_df)

def show_home_page():
    """Display home page with overview."""
    st.header("Welcome to SEO Content Analyzer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - Real-time URL analysis
        - Quality classification
        - Duplicate detection
        - Advanced NLP insights
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Metrics
        - Word count & readability
        - Flesch Reading Ease
        - Sentiment scores
        - Entity recognition
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 Quick Start
        1. Go to "Analyze URL"
        2. Enter any webpage URL
        3. Get instant insights
        4. View visualizations
        """)
    
    st.markdown("---")
    
    st.subheader("📈 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 1️⃣ Scrape\nExtract HTML content from the URL")
    with col2:
        st.markdown("#### 2️⃣ Parse\nExtract clean text and metadata")
    with col3:
        st.markdown("#### 3️⃣ Analyze\nCompute features & ML predictions")
    with col4:
        st.markdown("#### 4️⃣ Report\nDisplay insights & recommendations")

def show_analyze_page(model_data, features_df):
    """Display URL analysis page."""
    st.header("🔎 Analyze URL")
    
    url_input = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com/article",
        help="Enter a complete URL including http:// or https://"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🚀 Analyze", type="primary")
    with col2:
        advanced_nlp = st.checkbox("Include Advanced NLP Analysis", value=True)
    
    if analyze_button and url_input:
        with st.spinner("Analyzing URL... This may take a few seconds."):
            result = analyze_url_complete(url_input, model_data, features_df, advanced_nlp)
            
            if 'error' in result:
                st.error(f"❌ {result['error']}")
            else:
                st.session_state.analyzed_data = result
                display_analysis_results(result, advanced_nlp)

def analyze_url_complete(url, model_data, features_df, include_advanced=True):
    """Complete URL analysis with all features."""
    try:
        html_content, parsed = scrape_and_parse_url(url)
        if not parsed['body_text']:
            return {'error': 'No content extracted from URL'}
        
        features = extract_features(parsed['body_text'], model_data['model_objects'])
        quality_result = predict_quality(features, model_data)
        similarity_results = compute_similarity(
            features['embedding'], features_df, threshold=0.70
        )
        
        result = {
            'url': url,
            'title': parsed['title'],
            'word_count': features['word_count'],
            'sentence_count': features['sentence_count'],
            'readability': features['flesch_reading_ease'],
            'quality_label': quality_result['quality_label'],
            'quality_probabilities': quality_result['probabilities'],
            'is_thin': features['word_count'] < 500,
            'top_keywords': features['top_keywords'],
            'similar_to': similarity_results
        }
        
        if include_advanced:
            result['sentiment'] = extract_sentiment(parsed['body_text'])
            result['entities'] = extract_entities(parsed['body_text'])
            result['topics'] = extract_topics([parsed['body_text']])
            result['word_cloud_data'] = generate_word_cloud(parsed['body_text'])
        
        return result
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}

def display_analysis_results(result, include_advanced):
    """Display analysis results with visualizations."""
    st.success("✅ Analysis Complete!")
    st.subheader(result.get('title', 'Untitled'))
    st.caption(result['url'])
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        quality_class = f"quality-{result['quality_label'].lower()}"
        st.markdown("**Quality Score**")
        st.markdown(f'<p class="{quality_class}">{result["quality_label"]}</p>', unsafe_allow_html=True)
    with col2:
        st.metric("Word Count", result['word_count'])
    with col3:
        st.metric("Readability", f"{result['readability']:.1f}")
    with col4:
        status = "❌ Yes" if result['is_thin'] else "✅ No"
        st.markdown("**Thin Content**")
        st.markdown(status)

def show_dashboard_page(features_df, extracted_df):
    """Display dashboard with dataset statistics."""
    st.header("📈 Content Quality Dashboard")
    if features_df.empty:
        st.warning("No data available to display.")
        return

    features_df['is_thin'] = features_df['word_count'] < 500
    st.metric("Total Pages", len(features_df))
    st.metric("Thin Content", features_df['is_thin'].sum())

def show_duplicates_page(features_df):
    """Display duplicate detection page."""
    st.header("🔄 Duplicate Content Detection")
    try:
        duplicates_df = pd.read_csv('data/duplicates.csv')
        if duplicates_df.empty:
            st.info("No duplicates found.")
            return
        st.dataframe(duplicates_df)
    except FileNotFoundError:
        st.warning("No duplicates.csv found. Upload it to /data folder.")

def show_visualizations_page(features_df):
    """Display advanced visualizations."""
    st.header("📊 Advanced Visualizations")

    if 'embedding' not in features_df.columns:
        st.warning("No embedding column found in features.csv.")
        return

    def safe_eval(x):
        try:
            if pd.isna(x) or str(x).lower() in ["", "nan", "none"]:
                return []
            return eval(x)
        except Exception:
            return []

    embeddings_list = features_df['embedding'].apply(safe_eval).tolist()
    embeddings_list = [e for e in embeddings_list if len(e) > 0]

    if not embeddings_list:
        st.warning("No valid embeddings found in the dataset.")
        return

    embeddings = np.array(embeddings_list)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)

    n_display = min(20, len(similarity_matrix))
    sim_subset = similarity_matrix[:n_display, :n_display]

    fig = px.imshow(
        sim_subset,
        title=f"Cosine Similarity Matrix (First {n_display} Pages)",
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
