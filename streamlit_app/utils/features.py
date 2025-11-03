"""
Feature Engineering and Extraction Utilities
"""

import re
import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Cache model loading
_model_cache = {}

def get_sentence_transformer():
    """Get or load sentence transformer model."""
    if 'sentence_transformer' not in _model_cache:
        _model_cache['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
    return _model_cache['sentence_transformer']

def clean_text(text):
    """
    Clean text by lowercasing and removing extra whitespace.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_sentences(text):
    """
    Count sentences in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of sentences
    """
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def calculate_readability(text):
    """
    Calculate Flesch Reading Ease score.
    
    Args:
        text (str): Input text
        
    Returns:
        float: Flesch Reading Ease score
    """
    try:
        if not text.strip():
            return 0
        score = textstat.flesch_reading_ease(text)
        return score
    except:
        return 0

def extract_top_keywords(text, n=5):
    """
    Extract top N keywords using TF-IDF.
    
    Args:
        text (str): Input text
        n (int): Number of keywords to extract
        
    Returns:
        str: Pipe-separated keywords
    """
    try:
        if not text.strip():
            return ""
        
        # Clean text
        cleaned = clean_text(text)
        
        # Use TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=n,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform([cleaned])
        feature_names = vectorizer.get_feature_names_out()
        
        return "|".join(feature_names)
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return ""

def generate_embedding(text, max_length=512):
    """
    Generate embedding vector for text using sentence transformer.
    
    Args:
        text (str): Input text
        max_length (int): Maximum text length to consider
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    try:
        model = get_sentence_transformer()
        # Truncate text for efficiency
        text_truncated = text[:max_length] if len(text) > max_length else text
        embedding = model.encode(text_truncated)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        # Return zero vector with default size
        return np.zeros(384)

def extract_features(text, model_objects=None):
    """
    Extract all features from text.
    
    Args:
        text (str): Input text
        model_objects (dict): Optional pre-loaded models
        
    Returns:
        dict: Feature dictionary
    """
    # Basic metrics
    word_count = len(text.split())
    sentence_count = count_sentences(text)
    
    # Readability
    flesch_score = calculate_readability(text)
    
    # Keywords
    keywords = extract_top_keywords(text)
    
    # Embedding
    embedding = generate_embedding(text)
    
    features = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': flesch_score,
        'top_keywords': keywords,
        'embedding': embedding
    }
    
    return features

def compute_similarity(embedding, features_df, threshold=0.70, top_k=5):
    """
    Compute similarity between a single embedding and dataset.
    
    Args:
        embedding (numpy.ndarray): Query embedding
        features_df (pandas.DataFrame): DataFrame with embeddings
        threshold (float): Similarity threshold
        top_k (int): Number of top results to return
        
    Returns:
        list: List of similar documents
    """
    try:
        # Convert stored embeddings to array
        embeddings_array = np.array(features_df['embedding'].tolist())
        
        # Compute similarities
        similarities = cosine_similarity([embedding], embeddings_array)[0]
        
        # Find similar documents above threshold
        similar_indices = np.where(similarities >= threshold)[0]
        
        # Sort by similarity
        similar_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]]
        
        # Limit to top_k
        similar_indices = similar_indices[:top_k]
        
        results = []
        for idx in similar_indices:
            results.append({
                'url': features_df.iloc[idx]['url'],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    except Exception as e:
        print(f"Error computing similarity: {str(e)}")
        return []

def calculate_content_metrics(text):
    """
    Calculate additional content metrics.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'avg_word_length': 0,
        'avg_sentence_length': 0,
        'unique_words': 0,
        'lexical_diversity': 0
    }
    
    try:
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if words:
            metrics['avg_word_length'] = sum(len(w) for w in words) / len(words)
            metrics['unique_words'] = len(set(words))
            metrics['lexical_diversity'] = len(set(words)) / len(words)
        
        if sentences:
            metrics['avg_sentence_length'] = len(words) / len(sentences)
    
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
    
    return metrics