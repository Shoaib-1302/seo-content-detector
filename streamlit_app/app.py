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
        
        # Parse embeddings
        features_df['embedding'] = features_df['embedding'].apply(eval)
        
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
            "🔄 Duplicate Detection"
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
        st.markdown("""
        #### 1️⃣ Scrape
        Extract HTML content from the URL
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Parse
        Extract clean text and metadata
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Analyze
        Compute features & ML predictions
        """)
    
    with col4:
        st.markdown("""
        #### 4️⃣ Report
        Display insights & recommendations
        """)

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
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        quality_class = f"quality-{result['quality_label'].lower()}"
        st.markdown(f"**Quality Score**")
        st.markdown(f'<p class="{quality_class}">{result["quality_label"]}</p>', 
                    unsafe_allow_html=True)
    with col2:
        st.metric("Word Count", result['word_count'])
    with col3:
        st.metric("Readability", f"{result['readability']:.1f}")
    with col4:
        status = "❌ Yes" if result['is_thin'] else "✅ No"
        st.markdown(f"**Thin Content**")
        st.markdown(status)
    
    st.subheader("🎯 Quality Confidence")
    probs = result['quality_probabilities']
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs.keys()),
            y=list(probs.values()),
            marker_color=['#dc3545', '#ffc107', '#28a745']
        )
    ])
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Quality Level",
        yaxis_title="Probability",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔑 Top Keywords")
    keywords = result['top_keywords'].split('|') if result['top_keywords'] else []
    if keywords:
        cols = st.columns(len(keywords))
        for i, keyword in enumerate(keywords):
            with cols[i]:
                st.markdown(f"**`{keyword}`**")
    
    if result['similar_to']:
        st.subheader("🔄 Similar Content")
        similar_df = pd.DataFrame(result['similar_to'])
        similar_df['similarity'] = similar_df['similarity'].apply(lambda x: f"{x:.2%}")
        st.dataframe(similar_df, use_container_width=True)
    else:
        st.info("No similar content found in the database.")
    
    if include_advanced and 'sentiment' in result:
        st.markdown("---")
        st.subheader("🧠 Advanced NLP Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 😊 Sentiment Analysis")
            sentiment = result['sentiment']
            st.metric("Polarity", f"{sentiment['polarity']:.2f}")
            st.metric("Subjectivity", f"{sentiment['subjectivity']:.2f}")
        with col2:
            st.markdown("#### 📍 Named Entities")
            entities = result['entities']
            if entities:
                entity_df = pd.DataFrame(entities)
                st.dataframe(entity_df, use_container_width=True)
            else:
                st.info("No named entities detected")

def show_dashboard_page(features_df, extracted_df):
    """Display dashboard with dataset statistics."""
    st.header("📈 Content Quality Dashboard")
    
    quality_labels = []
    for idx, row in features_df.iterrows():
        wc = row['word_count']
        fre = row['flesch_reading_ease']
        if wc > 1500 and 50 <= fre <= 70:
            quality_labels.append('High')
        elif wc < 500 or fre < 30:
            quality_labels.append('Low')
        else:
            quality_labels.append('Medium')
    
    features_df['quality'] = quality_labels
    features_df['is_thin'] = features_df['word_count'] < 500
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", len(features_df))
    with col2:
        st.metric("High Quality", (features_df['quality'] == 'High').sum())
    with col3:
        st.metric("Thin Content", features_df['is_thin'].sum())
    with col4:
        st.metric("Avg Readability", f"{features_df['flesch_reading_ease'].mean():.1f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quality Distribution")
        quality_counts = features_df['quality'].value_counts()
        fig = px.pie(values=quality_counts.values, names=quality_counts.index,
                     color=quality_counts.index,
                     color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Word Count Distribution")
        fig = px.histogram(features_df, x='word_count', nbins=20)
        fig.add_vline(x=500, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Readability vs Word Count")
    fig = px.scatter(features_df, x='word_count', y='flesch_reading_ease',
                     color='quality', size='sentence_count',
                     color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Dataset Overview")
    display_df = features_df[['url', 'word_count', 'sentence_count', 
                              'flesch_reading_ease', 'quality', 'is_thin']].copy()
    display_df.columns = ['URL', 'Words', 'Sentences', 'Readability', 'Quality', 'Thin']
    st.dataframe(display_df, use_container_width=True)

def show_duplicates_page(features_df):
    """Display duplicate detection page."""
    st.header("🔄 Duplicate Content Detection")
    try:
        duplicates_df = pd.read_csv('data/duplicates.csv')
        if len(duplicates_df) > 0:
            st.success(f"Found {len(duplicates_df)} duplicate pairs")
            threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.05)
            filtered_df = duplicates_df[duplicates_df['similarity'] >= threshold]
            st.subheader(f"📊 Duplicate Pairs (Similarity ≥ {threshold:.0%})")
            display_df = filtered_df.copy()
            display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No duplicate content found in the dataset.")
    except FileNotFoundError:
        st.warning("No duplicates file found. Run the main pipeline first.")

if __name__ == "__main__":
    main()
