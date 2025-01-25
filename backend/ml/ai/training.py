import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.utils import resample
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.parse import urlparse

from .dataset import generate_dataset, extract_url_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def balance_dataset(X, y):
    """Balance dataset using upsampling for minority class"""
    X_df = pd.DataFrame(X)
    X_df['target'] = y

    # Separate majority and minority classes
    majority = X_df[X_df.target == 0]
    minority = X_df[X_df.target == 1]

    # Upsample minority class
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=42)

    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([majority, minority_upsampled])

    # Separate features and target
    y_balanced = df_balanced.target
    X_balanced = df_balanced.drop('target', axis=1)

    return X_balanced.values, y_balanced.values

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model using multiple metrics with enhanced reporting"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate additional metrics
    true_positives = sum((y_true == 1) & (y_pred == 1))
    false_positives = sum((y_true == 0) & (y_pred == 1))
    false_negatives = sum((y_true == 1) & (y_pred == 0))

    # Calculate detection rate and false positive rate
    detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    false_positive_rate = false_positives / (false_positives + sum(y_true == 0)) if (false_positives + sum(y_true == 0)) > 0 else 0

    logging.info(f"\n{model_name} Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Detection Rate: {detection_rate:.4f}")
    logging.info(f"False Positive Rate: {false_positive_rate:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate
    }

def calculate_age_based_risk(predictions, features, age_group):
    """
    Calculate risk level and score based on age group and URL features
    
    Args:
        predictions: Dictionary containing model predictions
        features: Dictionary of URL features
        age_group: Age group category ('kid', 'teen', 'adult')
    
    Returns:
        tuple: (risk_level, risk_score)
    """
    # Base weights for different models based on their reliability
    model_weights = {
        'knn': 0.3,
        'svm': 0.4,
        'nb': 0.3
    }

    # Age-specific risk modifiers
    age_modifiers = {
        'kid': 1.3,    # Increase risk score for kids
        'teen': 1.1,   # Slightly increase risk for teens
        'adult': 1.0   # No modification for adults
    }

    # Calculate weighted average probability
    weighted_prob = sum(
        predictions[model]['probability'] * model_weights[model]
        for model in predictions.keys()
    )

    # Apply age-specific modifier
    risk_score = weighted_prob * age_modifiers.get(age_group, 1.0)

    # Determine risk level based on score
    if risk_score > 0.8:
        risk_level = 'high'
    elif risk_score > 0.5:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    return risk_level, min(risk_score, 1.0)  # Cap risk score at 1.0

def train_models():
    """Train the ensemble of models for URL classification"""
    try:
        # Generate and preprocess dataset
        logging.info("Generating dataset...")
        dataset = generate_dataset()
        
        # Extract features and labels
        feature_columns = [col for col in dataset.columns if col not in ['url', 'is_blocked', 'category']]
        X = dataset[feature_columns].values
        y = dataset['is_blocked'].values
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Balance training data
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models with hyperparameters
        models = {
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'svm': SVC(kernel='rbf', probability=True, C=1.0),
            'nb': GaussianNB()
        }
        
        # Train and evaluate each model
        trained_models = {}
        for name, model in models.items():
            logging.info(f"\nTraining {name.upper()} model...")
            model.fit(X_train_scaled, y_train_balanced)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            metrics = evaluate_model(y_test, y_pred, name.upper())
            trained_models[name] = model
            
        # Save models and preprocessing objects
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in trained_models.items():
            joblib.dump(model, os.path.join(save_dir, f'{name}_model.pkl'))
        
        # Save scaler and feature columns
        joblib.dump(scaler, os.path.join(save_dir, 'url_scaler.pkl'))
        joblib.dump(feature_columns, os.path.join(save_dir, 'feature_cols.pkl'))
        
        logging.info("\nModel training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def predict_url(url, threshold=0.65, models_dir=None, age_group='kid'):
    """
    Make prediction for a single URL using ensemble of models
    Args:
        url: URL to analyze
        threshold: Classification threshold (default: 0.65)
        models_dir: Directory containing trained models
        age_group: Age group for risk assessment
    Returns:
        tuple: (is_unsafe, probability, risk_score)
    """
    try:
        # Load models and scaler
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')

        try:
            knn = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
            svm = joblib.load(os.path.join(models_dir, 'svm_model.pkl'))
            nb = joblib.load(os.path.join(models_dir, 'nb_model.pkl'))
            scaler = joblib.load(os.path.join(models_dir, 'url_scaler.pkl'))
            feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        except FileNotFoundError as e:
            logging.error(f"Could not load models from {models_dir}. Error: {e}")
            # Initialize empty predictions dictionary to avoid division by zero
            predictions = {}
            return False, 0.0, 0.0

        # Extract features
        features = extract_url_features(url)
        if not features:
            raise ValueError("Failed to extract features from URL")

        # Ensure features are a list of values
        if isinstance(features, dict):
            features = list(features.values())

        # Ensure features array matches expected features
        if len(features) != len(feature_cols):
            logging.error(f"Mismatch in feature length: Expected {len(feature_cols)} features, got {len(features)}")
            logging.error(f"Feature columns: {feature_cols}")
            logging.error(f"Extracted features: {features}")
            raise ValueError(f"Expected {len(feature_cols)} features, got {len(features)}")

        # Scale features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Get predictions and probabilities
        predictions = {}

        # KNN
        knn_prob = knn.predict_proba(features_scaled)[0][1]
        predictions['knn'] = {
            'prediction': knn_prob > threshold,
            'probability': knn_prob
        }

        # SVM
        svm_prob = svm.predict_proba(features_scaled)[0][1]
        predictions['svm'] = {
            'prediction': svm_prob > threshold,
            'probability': svm_prob
        }

        # Naive Bayes
        nb_prob = nb.predict_proba(features_scaled)[0][1]
        predictions['nb'] = {
            'prediction': nb_prob > threshold,
            'probability': nb_prob
        }

        # Calculate base risk from model predictions
        base_features = extract_url_features(url) if isinstance(url, str) else dict(zip(feature_cols, features))
        
        # Trust factors (reduce risk score)
        trust_score = 1.0
        if base_features.get('has_https', 0) == 1:
            trust_score *= 0.7  # Significant trust for HTTPS
        
        # Check for trusted domains
        domain = urlparse(url).netloc.lower()
        trusted_domains = {'github.com', 'python.org', 'wikipedia.org'}
        if any(td in domain for td in trusted_domains):
            trust_score *= 0.5  # High trust for known good domains
            
        # Risk factors (increase risk score)
        risk_multiplier = 1.0
        if base_features.get('is_ip_address', 0) == 1:
            risk_multiplier *= 2.0  # Major increase for IP-based URLs
        if base_features.get('suspicious_word_count', 0) > 2:
            risk_multiplier *= 1.5  # Increase for multiple suspicious words
        if base_features.get('suspicious_tld', 0) == 1:
            risk_multiplier *= 1.8  # Increase for suspicious TLDs
            
        # Calculate age-specific thresholds
        age_thresholds = {
            'kid': 0.5,    # More strict for kids
            'teen': 0.6,   # Moderate for teens
            'adult': 0.7   # More lenient for adults
        }
        effective_threshold = age_thresholds.get(age_group, threshold)
        
        # Get base risk score from models
        risk_level, risk_score = calculate_age_based_risk(
            predictions,
            base_features,
            age_group
        )
        
        # Apply trust and risk modifiers
        final_risk_score = (risk_score * risk_multiplier * trust_score)
        
        # Ensure score stays in [0,1] range
        final_risk_score = max(0.0, min(1.0, final_risk_score))

        # Update risk level based on final score
        if final_risk_score > 0.8:
            risk_level = 'high'
        elif final_risk_score > 0.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Enhanced result with age-specific risk assessment
        result = {
            'is_unsafe': bool(final_risk_score > effective_threshold),
            'risk_score': final_risk_score,
            'risk_level': risk_level,
            'age_group': age_group,
            'model_predictions': predictions
        }

        # Visualize the predictions
        plt.figure(figsize=(8, 6))
        plt.bar(['Risk Score'], [final_risk_score])
        plt.title('Risk Score for URL')
        plt.ylim(0, 1)
        plt.ylabel('Risk Score')
        plt.show()

        # Determine category based on features and predictions
        domain = urlparse(url).netloc.lower()
        
        # Convert URL to lowercase for comparison
        url_lower = url.lower()
        
        # Special case for localhost URLs
        if domain.startswith('localhost'):
            # Check path for category hints
            if '/dashboard' in url_lower:
                determined_category = 'Dashboard'
            else:
                determined_category = 'Local'
            result['category'] = determined_category
            result['risk_level'] = 'Low'  # Local URLs are generally safe
            return result['is_unsafe'], result['risk_score'], result


        # Initialize category mapping with more comprehensive patterns
        categories = {
            'education': ['edu', 'school', 'university', 'learn', 'course', 'tutorial', 'study'],
            'entertainment': ['youtube.com', 'netflix.com', 'games', 'music', 'movie', 'video', 'stream'],
            'social': ['facebook.com', 'twitter.com', 'instagram.com', 'social', '/notifications', '/feed', '/posts'],
            'news': ['news', 'bbc.com', 'cnn.com', 'reuters.com', 'article', 'blog'],
            'shopping': ['amazon.com', 'ebay.com', 'shop', 'store', 'cart', 'checkout', 'product'],
            'adult': ['adult', 'xxx', 'porn', 'nsfw'],
            'gambling': ['casino', 'bet', 'poker', 'gambling', 'lottery'],
            'malware': ['malware', 'virus', 'hack', 'trojan', 'worm'],
            'professional': ['linkedin', 'indeed', 'glassdoor', 'jobs', 'career', 'resume']
        }

        # Special case for LinkedIn
        if 'linkedin.com' in domain:
            result['category'] = 'Professional'
            result['risk_level'] = 'Low'  # LinkedIn is a trusted platform
            return result['is_unsafe'], result['risk_score'], result
        
        # Determine category with confidence threshold
        determined_category = 'Unknown'
        url_lower = url.lower()
        max_confidence = 0.0
        best_category = 'Unknown'
        
        # Calculate confidence scores for each category
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(keyword in url_lower for keyword in keywords)
            category_scores[category] = score
            if score > max_confidence:
                max_confidence = score
                best_category = category.capitalize()
        
        # Only assign category if confidence is above threshold
        confidence_threshold = 2  # At least 2 strong matches
        if max_confidence >= confidence_threshold:
            determined_category = best_category
        else:
            # For low confidence, use domain-based fallback
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                tld = domain_parts[-1]
                if tld in ['edu', 'gov']:
                    determined_category = 'Education' if tld == 'edu' else 'Government'
                elif tld in ['org', 'net']:
                    determined_category = 'Organization'
                else:
                    determined_category = 'General'
        
        result['category'] = determined_category
        
        # Adjust risk level for unknown categories
        if determined_category == 'Unknown':
            result['risk_level'] = 'Medium'  # Default to medium risk for unknown
            result['risk_score'] = min(result['risk_score'] * 1.2, 1.0)  # Slightly increase risk
        
        return result['is_unsafe'], result['risk_score'], result

    except Exception as e:
        logging.error(f"Error during ensemble prediction: {e}")
        raise

if __name__ == "__main__":
    train_models()
