"""
Content Quality Scoring Utilities
"""

import pickle
import numpy as np
from pathlib import Path

def load_model(model_path):
    """
    Load trained model and label encoder.
    
    Args:
        model_path (str): Path to pickled model
        
    Returns:
        dict: Dictionary with model and label encoder
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return {
            'model': model_data['model'],
            'label_encoder': model_data['label_encoder'],
            'model_objects': model_data
        }
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_quality(features, model_data):
    """
    Predict content quality using trained model.
    
    Args:
        features (dict): Feature dictionary
        model_data (dict): Model and encoder
        
    Returns:
        dict: Prediction results with probabilities
    """
    try:
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        
        # Prepare feature vector
        feature_vector = np.array([[
            features['word_count'],
            features['sentence_count'],
            features['flesch_reading_ease']
        ]])
        
        # Predict
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Get label
        quality_label = label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {}
        for idx, label in enumerate(label_encoder.classes_):
            prob_dict[label] = float(probabilities[idx])
        
        return {
            'quality_label': quality_label,
            'probabilities': prob_dict,
            'prediction_encoded': int(prediction)
        }
    
    except Exception as e:
        print(f"Error predicting quality: {str(e)}")
        return {
            'quality_label': 'Unknown',
            'probabilities': {},
            'error': str(e)
        }

def assign_quality_label_rule_based(word_count, readability):
    """
    Assign quality label using rule-based approach (baseline).
    
    Args:
        word_count (int): Word count
        readability (float): Flesch Reading Ease score
        
    Returns:
        str: Quality label (High/Medium/Low)
    """
    if word_count > 1500 and 50 <= readability <= 70:
        return 'High'
    elif word_count < 500 or readability < 30:
        return 'Low'
    else:
        return 'Medium'

def get_quality_recommendations(features, quality_label):
    """
    Generate recommendations based on content quality.
    
    Args:
        features (dict): Feature dictionary
        quality_label (str): Predicted quality label
        
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    word_count = features['word_count']
    readability = features['flesch_reading_ease']
    
    # Word count recommendations
    if word_count < 500:
        recommendations.append("⚠️ Content is too thin. Aim for at least 500 words for better SEO.")
    elif word_count < 1000:
        recommendations.append("💡 Consider expanding content to 1000+ words for better rankings.")
    elif word_count > 3000:
        recommendations.append("📝 Very comprehensive content! Ensure it stays focused and scannable.")
    
    # Readability recommendations
    if readability < 30:
        recommendations.append("⚠️ Content is very difficult to read. Simplify sentences and vocabulary.")
    elif readability < 50:
        recommendations.append("💡 Readability could be improved. Use shorter sentences and simpler words.")
    elif readability > 80:
        recommendations.append("💡 Content might be too simple. Consider adding more depth.")
    
    # Quality-specific recommendations
    if quality_label == 'Low':
        recommendations.append("🔴 Low quality detected. Focus on content length and readability.")
    elif quality_label == 'Medium':
        recommendations.append("🟡 Medium quality. Small improvements can make this high quality.")
    else:
        recommendations.append("🟢 High quality content! Maintain this standard.")
    
    # Keyword recommendations
    if not features.get('top_keywords'):
        recommendations.append("💡 No clear keywords detected. Ensure content has a clear topic focus.")
    
    return recommendations

def calculate_seo_score(features, quality_label):
    """
    Calculate an overall SEO score (0-100).
    
    Args:
        features (dict): Feature dictionary
        quality_label (str): Predicted quality label
        
    Returns:
        int: SEO score (0-100)
    """
    score = 0
    
    # Word count scoring (max 40 points)
    word_count = features['word_count']
    if word_count >= 1500:
        score += 40
    elif word_count >= 1000:
        score += 30
    elif word_count >= 500:
        score += 20
    else:
        score += 10
    
    # Readability scoring (max 30 points)
    readability = features['flesch_reading_ease']
    if 50 <= readability <= 70:
        score += 30
    elif 40 <= readability <= 80:
        score += 20
    else:
        score += 10
    
    # Quality label bonus (max 30 points)
    if quality_label == 'High':
        score += 30
    elif quality_label == 'Medium':
        score += 20
    else:
        score += 10
    
    return min(score, 100)

def get_quality_color(quality_label):
    """
    Get color code for quality label.
    
    Args:
        quality_label (str): Quality label
        
    Returns:
        str: Hex color code
    """
    colors = {
        'High': '#28a745',
        'Medium': '#ffc107',
        'Low': '#dc3545'
    }
    return colors.get(quality_label, '#6c757d')