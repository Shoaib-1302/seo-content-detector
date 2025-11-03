"""
Advanced NLP Analysis Utilities
Sentiment Analysis, Named Entity Recognition, Topic Modeling
"""

import re
import numpy as np
from collections import Counter
from textblob import TextBlob
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Cache models
_nlp_model_cache = {}

def get_spacy_model():
    """Load and cache spaCy model."""
    if 'spacy' not in _nlp_model_cache:
        try:
            _nlp_model_cache['spacy'] = spacy.load('en_core_web_sm')
        except:
            # If model not found, download it
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            _nlp_model_cache['spacy'] = spacy.load('en_core_web_sm')
    return _nlp_model_cache['spacy']

def extract_sentiment(text):
    """
    Extract sentiment using TextBlob.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment scores (polarity, subjectivity)
    """
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        return {
            'polarity': sentiment.polarity,  # -1 to 1 (negative to positive)
            'subjectivity': sentiment.subjectivity,  # 0 to 1 (objective to subjective)
            'label': get_sentiment_label(sentiment.polarity)
        }
    except Exception as e:
        print(f"Error extracting sentiment: {str(e)}")
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'label': 'Neutral'
        }

def get_sentiment_label(polarity):
    """Convert polarity score to label."""
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def extract_entities(text, max_length=1000000):
    """
    Extract named entities using spaCy.
    
    Args:
        text (str): Input text
        max_length (int): Maximum text length
        
    Returns:
        list: List of entity dictionaries
    """
    try:
        nlp = get_spacy_model()
        
        # Limit text length for processing
        if len(text) > max_length:
            text = text[:max_length]
        
        doc = nlp(text)
        
        entities = []
        entity_counts = Counter()
        
        for ent in doc.ents:
            entity_counts[(ent.text, ent.label_)] += 1
        
        # Get top entities
        for (entity_text, entity_label), count in entity_counts.most_common(10):
            entities.append({
                'text': entity_text,
                'label': entity_label,
                'count': count
            })
        
        return entities
    
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return []

def extract_topics(texts, n_topics=3, n_words=5):
    """
    Extract topics using LDA.
    
    Args:
        texts (list): List of text documents
        n_topics (int): Number of topics to extract
        n_words (int): Number of words per topic
        
    Returns:
        list: List of topic dictionaries
    """
    try:
        if len(texts) < 2:
            return []
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            min_df=1
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=min(n_topics, len(texts)),
            random_state=42
        )
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weight': float(topic.sum())
            })
        
        return topics
    
    except Exception as e:
        print(f"Error extracting topics: {str(e)}")
        return []

def generate_word_cloud(text, max_words=50, width=800, height=400):
    """
    Generate word cloud visualization.
    
    Args:
        text (str): Input text
        max_words (int): Maximum words in cloud
        width (int): Image width
        height (int): Image height
        
    Returns:
        str: Base64 encoded image or None
    """
    try:
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color='white',
            colormap='viridis',
            stopwords=set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
        ).generate(text)
        
        # Create image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        print(f"Error generating word cloud: {str(e)}")
        return None

def extract_key_phrases(text, n=10):
    """
    Extract key phrases using spaCy.
    
    Args:
        text (str): Input text
        n (int): Number of phrases to extract
        
    Returns:
        list: List of key phrases
    """
    try:
        nlp = get_spacy_model()
        doc = nlp(text[:100000])  # Limit length
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                noun_phrases.append(chunk.text.lower())
        
        # Count and return top phrases
        phrase_counts = Counter(noun_phrases)
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(n)]
        
        return top_phrases
    
    except Exception as e:
        print(f"Error extracting key phrases: {str(e)}")
        return []

def analyze_text_complexity(text):
    """
    Analyze text complexity metrics.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Complexity metrics
    """
    try:
        nlp = get_spacy_model()
        doc = nlp(text[:100000])
        
        # Calculate metrics
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct and not token.is_space]
        
        metrics = {
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': sum(len(token.text) for token in words) / len(words) if words else 0,
            'unique_words_ratio': len(set(token.text.lower() for token in words)) / len(words) if words else 0,
            'noun_ratio': sum(1 for token in words if token.pos_ == 'NOUN') / len(words) if words else 0,
            'verb_ratio': sum(1 for token in words if token.pos_ == 'VERB') / len(words) if words else 0,
            'adj_ratio': sum(1 for token in words if token.pos_ == 'ADJ') / len(words) if words else 0
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error analyzing complexity: {str(e)}")
        return {}

def extract_questions(text):
    """
    Extract questions from text.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of questions
    """
    # Simple regex pattern for questions
    question_pattern = r'[A-Z][^.!?]*\?'
    questions = re.findall(question_pattern, text)
    return questions[:10]  # Return top 10

def calculate_content_freshness_indicators(text):
    """
    Calculate indicators of content freshness/currency.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Freshness indicators
    """
    current_year = 2024
    
    # Look for years
    years = re.findall(r'\b(20\d{2})\b', text)
    years = [int(y) for y in years]
    
    indicators = {
        'contains_recent_year': any(y >= current_year - 1 for y in years) if years else False,
        'latest_year_mentioned': max(years) if years else None,
        'year_mentions': len(years),
        'has_dates': bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)),
        'has_updated_indicator': bool(re.search(r'\b(updated|revised|current|latest|new)\b', text.lower()))
    }
    
    return indicators