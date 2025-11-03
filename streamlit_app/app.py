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
        model_data = load_model('models/quality_model.pkl')
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
        # Scrape and parse
        html_content, parsed = scrape_and_parse_url(url)
        if not parsed['body_text']:
            return {'error': 'No content extracted from URL'}
        
        # Extract features
        features = extract_features(parsed['body_text'], model_data['model_objects'])
        
        # Predict quality
        quality_result = predict_quality(features, model_data)
        
        # Compute similarity
        similarity_results = compute_similarity(
            features['embedding'],
            features_df,
            threshold=0.70
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
        
        # Advanced NLP
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
    
    # Title and URL
    st.subheader(result.get('title', 'Untitled'))
    st.caption(result['url'])
    
    st.markdown("---")
    
    # Key Metrics
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
    
    # Quality Probabilities
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
    
    # Keywords
    st.subheader("🔑 Top Keywords")
    keywords = result['top_keywords'].split('|') if result['top_keywords'] else []
    if keywords:
        cols = st.columns(len(keywords))
        for i, keyword in enumerate(keywords):
            with cols[i]:
                st.markdown(f"**`{keyword}`**")
    
    # Similar Content
    if result['similar_to']:
        st.subheader("🔄 Similar Content")
        similar_df = pd.DataFrame(result['similar_to'])
        similar_df['similarity'] = similar_df['similarity'].apply(lambda x: f"{x:.2%}")
        st.dataframe(similar_df, use_container_width=True)
    else:
        st.info("No similar content found in the database.")
    
    # Advanced NLP Results
    if include_advanced and 'sentiment' in result:
        st.markdown("---")
        st.subheader("🧠 Advanced NLP Analysis")
        
        # Sentiment
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 😊 Sentiment Analysis")
            sentiment = result['sentiment']
            st.metric("Polarity", f"{sentiment['polarity']:.2f}", 
                     help="Range: -1 (negative) to +1 (positive)")
            st.metric("Subjectivity", f"{sentiment['subjectivity']:.2f}",
                     help="Range: 0 (objective) to 1 (subjective)")
        
        with col2:
            st.markdown("#### 📍 Named Entities")
            entities = result['entities']
            if entities:
                entity_df = pd.DataFrame(entities)
                st.dataframe(entity_df, use_container_width=True)
            else:
                st.info("No named entities detected")
        
        # Topics
        if result.get('topics'):
            st.markdown("#### 📚 Topic Distribution")
            topics_df = pd.DataFrame(result['topics'])
            fig = px.bar(topics_df, x='topic_id', y='weight', 
                        title="Dominant Topics")
            st.plotly_chart(fig, use_container_width=True)
        
        # Word Cloud
        if result.get('word_cloud_data'):
            st.markdown("#### ☁️ Word Cloud")
            st.image(result['word_cloud_data'])

def show_dashboard_page(features_df, extracted_df):
    """Display dashboard with dataset statistics."""
    st.header("📈 Content Quality Dashboard")
    
    # Load quality labels
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
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", len(features_df))
    with col2:
        high_quality = (features_df['quality'] == 'High').sum()
        st.metric("High Quality", high_quality)
    with col3:
        thin_pages = features_df['is_thin'].sum()
        st.metric("Thin Content", thin_pages)
    with col4:
        avg_readability = features_df['flesch_reading_ease'].mean()
        st.metric("Avg Readability", f"{avg_readability:.1f}")
    
    st.markdown("---")
    
    # Quality distribution
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
        fig = px.histogram(features_df, x='word_count', nbins=20,
                          title="Distribution of Word Counts")
        fig.add_vline(x=500, line_dash="dash", line_color="red", 
                     annotation_text="Thin Content Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Readability vs Word Count
    st.subheader("📊 Readability vs Word Count")
    fig = px.scatter(features_df, x='word_count', y='flesch_reading_ease',
                    color='quality', size='sentence_count',
                    color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'},
                    title="Content Quality Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Dataset Overview")
    display_df = features_df[['url', 'word_count', 'sentence_count', 
                               'flesch_reading_ease', 'quality', 'is_thin']].copy()
    display_df.columns = ['URL', 'Words', 'Sentences', 'Readability', 'Quality', 'Thin']
    st.dataframe(display_df, use_container_width=True)

def show_duplicates_page(features_df):
    """Display duplicate detection page."""
    st.header("🔄 Duplicate Content Detection")
    
    # Load or compute duplicates
    try:
        duplicates_df = pd.read_csv('data/duplicates.csv')
        
        if len(duplicates_df) > 0:
            st.success(f"Found {len(duplicates_df)} duplicate pairs")
            
            # Similarity threshold selector
            threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.05)
            filtered_df = duplicates_df[duplicates_df['similarity'] >= threshold]
            
            st.subheader(f"📊 Duplicate Pairs (Similarity ≥ {threshold:.0%})")
            
            # Format for display
            display_df = filtered_df.copy()
            display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True)
            
            # Similarity distribution
            st.subheader("📈 Similarity Distribution")
            fig = px.histogram(duplicates_df, x='similarity', nbins=20,
                             title="Distribution of Similarity Scores")
            fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                         annotation_text="Current Threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No duplicate content found in the dataset.")
            
    except FileNotFoundError:
        st.warning("No duplicates file found. Run the main pipeline first.")

def show_visualizations_page(features_df):
    """Display advanced visualizations."""
    st.header("📊 Advanced Visualizations")
    
    # Parse embeddings
    embeddings = np.array(features_df['embedding'].apply(eval).tolist())
    
    # Similarity heatmap
    st.subheader("🔥 Content Similarity Heatmap")
    
    with st.spinner("Computing similarity matrix..."):
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
    
    # Limit to first 20 for visualization
    n_display = min(20, len(similarity_matrix))
    sim_subset = similarity_matrix[:n_display, :n_display]
    
    fig = px.imshow(sim_subset, 
                    title=f"Cosine Similarity Matrix (First {n_display} Pages)",
                    color_continuous_scale='RdYlGn',
                    aspect='auto')
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (mock data for visualization)
    st.subheader("⚡ Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': ['word_count', 'flesch_reading_ease', 'sentence_count'],
        'importance': [0.45, 0.32, 0.23]
    })
    
    fig = px.bar(feature_importance, x='importance', y='feature', 
                orientation='h', title="Model Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlations")
    corr_features = features_df[['word_count', 'sentence_count', 'flesch_reading_ease']]
    corr_matrix = corr_features.corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plots
    st.subheader("📦 Feature Distributions by Quality")
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
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Word Count", "Readability"))
    
    for quality in ['High', 'Medium', 'Low']:
        data = features_df[features_df['quality'] == quality]
        fig.add_trace(
            go.Box(y=data['word_count'], name=quality, marker_color={
                'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'
            }[quality]),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=data['flesch_reading_ease'], name=quality, showlegend=False,
                  marker_color={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}[quality]),
            row=1, col=2
        )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()