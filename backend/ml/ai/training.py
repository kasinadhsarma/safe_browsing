import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
import joblib
from urllib.parse import urlparse
from .dataset import load_url_data, extract_url_features, process_urls_parallel
from .metrics import MLMetrics
from datetime import datetime
from .metrics import SessionLocal

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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

    # Combine majority and upsampled minority
    balanced_df = pd.concat([majority, minority_upsampled])
    X_balanced = balanced_df.drop('target', axis=1)
    y_balanced = balanced_df['target'].values
    return X_balanced, y_balanced

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model and plot metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics

def calculate_age_based_risk(predictions, features, age_group='kid'):
    """Calculate risk level and score based on age group and features"""
    # Base weights for different models based on their performance
    model_weights = {
        'knn': 0.3,
        'svm': 0.4,
        'nb': 0.3
    }

    # Calculate weighted average of model predictions
    weighted_score = sum(pred['probability'] * model_weights[model]
                        for model, pred in predictions.items())

    # Age-specific risk modifiers
    age_risk_multipliers = {
        'kid': 1.5,    # More strict for kids
        'teen': 1.2,   # Moderately strict for teens
        'adult': 1.0   # Base level for adults
    }

    # Apply age-specific risk multiplier
    risk_score = min(1.0, weighted_score * age_risk_multipliers.get(age_group, 1.0))

    # Determine risk level
    if risk_score > 0.8:
        risk_level = 'high'
    elif risk_score > 0.5:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    return risk_level, risk_score

def predict_url(url, threshold=0.65, models_dir=None, age_group='kid'):
    """Make prediction for a single URL using ensemble of models"""
    # Add automatic adult content blocking
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    # List of known adult domains
    adult_domains = {
        'brazzers.com', 'pornhub.com', 'xvideos.com', 'xnxx.com',
        'youporn.com', 'redtube.com', 'xhamster.com'
    }

    # Block adult content for kids automatically
    if any(adult in domain for adult in adult_domains):
        if age_group == 'kid':
            return True, 1.0, 'high'
        elif age_group == 'teen':
            return True, 0.8, 'medium'
        else:  # adult
            return True, 0.6, 'low'

    # Handle internal URLs and trusted domains
    if parsed_url.scheme in ['chrome', 'about', 'file']:
        return False, 0.1, 'low'

    trusted_domains = {
        'youtube.com', 'linkedin.com', 'github.com',
        'python.org', 'wikipedia.org', 'google.com'
    }
    if any(td in domain for td in trusted_domains):
        return False, 0.3, 'low'

    try:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'latest')

        # Load models and scaler
        scaler = joblib.load(os.path.join(models_dir, 'url_scaler.pkl'))
        knn_model = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
        svm_model = joblib.load(os.path.join(models_dir, 'svm_model.pkl'))
        nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl'))
        feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))

        # Extract and prepare features
        features = extract_url_features(url)
        if isinstance(features, dict):
            features = list(features.values())

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Get individual model predictions
        knn_prob = knn_model.predict_proba(features_scaled)[0][1]
        svm_prob = svm_model.predict_proba(features_scaled)[0][1]
        nb_prob = nb_model.predict_proba(features_scaled)[0][1]

        # Calculate weighted average
        weighted_score = (knn_prob * 0.3) + (svm_prob * 0.4) + (nb_prob * 0.3)

        # Calculate age-based risk
        age_risk_level, age_risk_score = calculate_age_based_risk(
            predictions={
                'knn': {'probability': knn_prob},
                'svm': {'probability': svm_prob},
                'nb': {'probability': nb_prob}
            },
            features=features,
            age_group=age_group
        )

        is_unsafe = weighted_score >= threshold
        risk_level = age_risk_level
        risk_score = age_risk_score

        return is_unsafe, risk_score, risk_level

    except Exception as e:
        logging.error(f"Error in predict_url: {e}")
        return False, 0.5, 'medium'

def train_models(save_dir: str):
    """Train ML models and save them to the specified directory."""
    logging.info("Loading dataset...")
    X, y = load_url_data()

    # Check if the dataset is empty
    if X.empty or y.empty:
        logging.error("The dataset is empty. Cannot train models.")
        return

    logging.info(f"Dataset loaded with {len(X)} samples and {len(y)} labels.")

    # If 'url' is needed for any reason, ensure it's present
    if 'url' not in X.columns:
        logging.error("'url' column is missing from the feature set.")
        return

    urls = X['url'].tolist()  # Replace 'data' with 'X'

    # Extract features for each URL
    logging.info("Extracting features for URLs...")
    features = process_urls_parallel(urls)
    X = pd.DataFrame(features)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Initialize models
    knn = KNeighborsClassifier()
    svm = SVC(probability=True)
    nb = GaussianNB()

    # Train KNN
    logging.info("Training KNN model...")
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    logging.info(f"KNN Accuracy: {knn_acc:.4f}")

    # Train SVM
    logging.info("Training SVM model...")
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    logging.info(f"SVM Accuracy: {svm_acc:.4f}")

    # Train Naive Bayes
    logging.info("Training Naive Bayes model...")
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    logging.info(f"Naive Bayes Accuracy: {nb_acc:.4f}")

    # Save models and scaler
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(knn, os.path.join(save_dir, 'knn_model.pkl'))
    joblib.dump(svm, os.path.join(save_dir, 'svm_model.pkl'))
    joblib.dump(nb, os.path.join(save_dir, 'naive_bayes_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'url_scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(save_dir, 'feature_cols.pkl'))

    logging.info(f"Models and scaler saved to: {save_dir}")

    # Evaluate models and store metrics
    metrics = MLMetrics(
        id=os.urandom(16).hex(),
        timestamp=datetime.utcnow(),
        model_name="KNN",
        accuracy=knn_acc,
        precision=precision_score(y_test, knn_pred),
        recall=recall_score(y_test, knn_pred),
        f1_score=f1_score(y_test, knn_pred),
        training_data_size=len(X_train)
    )
    db = SessionLocal()
    try:
        db.add(metrics)
        db.commit()
        logging.info("KNN metrics saved to database.")
    except Exception as e:
        logging.error(f"Error saving KNN metrics: {e}")
    finally:
        db.close()

    # Repeat metrics storage for SVM and Naive Bayes as needed
    # ...existing code...

# Ensure the main execution calls train_models with the correct directory
if __name__ == "__main__":
    models_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'latest')
    train_models(save_dir=models_save_dir)
