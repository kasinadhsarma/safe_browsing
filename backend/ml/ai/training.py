import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler

# Import required functions and variables
from dataset import generate_dataset, FIXED_FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom feature extractor for URL features"""
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100)
        self.svd = TruncatedSVD(n_components=20)

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.svd.fit(self.tfidf.transform(X))
        return self

    def transform(self, X):
        tfidf_features = self.tfidf.transform(X)
        svd_features = self.svd.transform(tfidf_features)
        return svd_features

def balance_dataset(X, y):
    """Balance dataset using SMOTE"""
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

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

def objective(trial, X_train, y_train):
    """Optuna objective function for hyperparameter tuning"""
    # Define hyperparameters to tune
    model_name = trial.suggest_categorical('model', ['knn', 'svm', 'nb'])
    
    if model_name == 'knn':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    
    elif model_name == 'svm':
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    
    elif model_name == 'nb':
        var_smoothing = trial.suggest_float('var_smoothing', 1e-11, 1e-6, log=True)
        model = GaussianNB(var_smoothing=var_smoothing)
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    return scores.mean()

def train_models():
    """Train the ensemble of models for URL classification with advanced techniques"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Generate and preprocess dataset
        logging.info("Generating dataset...")
        dataset = generate_dataset()  # Now defined or imported

        if dataset.empty:
            logging.error("Dataset is empty. Check the input CSV files.")
            return

        X = dataset.drop('is_blocked', axis=1).values
        y = dataset['is_blocked'].values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Balance training data using SMOTE
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)

        # Use Optuna for hyperparameter tuning
        logging.info("Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train_balanced), n_trials=50)

        # Get the best hyperparameters
        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")

        # Train the best model
        if best_params['model'] == 'knn':
            model = KNeighborsClassifier(
                n_neighbors=best_params['n_neighbors'],
                weights=best_params['weights']
            )
        elif best_params['model'] == 'svm':
            model = SVC(
                C=best_params['C'],
                kernel=best_params['kernel'],
                gamma=best_params['gamma'],
                probability=True
            )
        elif best_params['model'] == 'nb':
            model = GaussianNB(var_smoothing=best_params['var_smoothing'])

        logging.info(f"Training best model: {best_params['model']}")
        model.fit(X_train_scaled, y_train_balanced)

        # Evaluate the best model
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, f"Best Model ({best_params['model']})")
        joblib.dump(model, os.path.join(save_dir, 'best_model.pkl'))

        # Save models and preprocessing objects
        joblib.dump(scaler, os.path.join(save_dir, 'url_scaler.pkl'))
        joblib.dump(FIXED_FEATURE_COLUMNS, os.path.join(save_dir, 'feature_cols.pkl'))

        logging.info("\nModel training completed successfully!")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def predict_url(url):
    """
    Predict URL safety information using the trained model
    
    Args:
        url (str): The URL to classify
        
    Returns:
        dict: Dictionary containing:
            - is_blocked (bool): Whether the URL should be blocked
            - confidence (float): Prediction confidence score
            - risk_features (dict): Dictionary of risk features like:
                - kid_unsafe_score
                - teen_unsafe_score
                - suspicious_word_count
                - has_malicious_content
            - category (str): Predicted category (e.g., 'safe', 'adult', 'malware', etc.)
    """
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
        
        # Load the saved model and preprocessing objects
        model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'url_scaler.pkl'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
        
        # Extract features using the same logic as in dataset.py
        from .dataset import extract_url_features
        features_dict = extract_url_features(url)
        
        # Convert features dict to list in the same order as feature_cols
        features = [features_dict[col] for col in feature_cols]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction with probability
        is_blocked = bool(model.predict(features_scaled)[0])
        
        # Get prediction probability
        prob = model.predict_proba(features_scaled)[0]
        confidence = float(prob[1] if is_blocked else prob[0])
        
        # Extract relevant risk features
        risk_features = {
            'kid_unsafe_score': float(features_dict['kid_unsafe_score']),
            'teen_unsafe_score': float(features_dict['teen_unsafe_score']),
            'suspicious_word_count': float(features_dict['suspicious_word_count']),
            'has_malicious_content': bool(features_dict['has_suspicious_words'])
        }
        
        # Determine category based on features
        category = 'safe'
        if is_blocked:
            if features_dict['kid_unsafe_score'] > 0.7:
                category = 'adult'
            elif features_dict['has_executable'] > 0:
                category = 'malware'
            elif features_dict['suspicious_word_count'] > 5:
                category = 'inappropriate'
            elif features_dict['has_suspicious_params'] > 0:
                category = 'suspicious'
            else:
                category = 'blocked'
        
        return {
            'is_blocked': is_blocked,
            'confidence': confidence,
            'risk_features': risk_features,
            'category': category
        }
        
    except Exception as e:
        logging.error(f"Error during URL prediction: {e}")
        raise

if __name__ == "__main__":
    train_models()
