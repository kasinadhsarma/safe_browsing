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

def train_models():
    """Train and evaluate multiple models for URL classification"""
    try:
        # Generate and prepare dataset
        logging.info("Generating dataset...")
        df = generate_dataset()
        if df.empty:
            raise ValueError("Empty dataset generated")

        # Ensure all required columns exist in the DataFrame
        feature_cols = [
            # Basic features
            'length', 'num_dots', 'num_digits', 'num_special', 'entropy',
            'token_count', 'avg_token_length', 'max_token_length', 'min_token_length',
            # Domain features
            'domain_length', 'has_subdomain', 'has_www', 'domain_entropy',
            'is_ip_address', 'domain_digit_ratio', 'domain_special_ratio',
            'domain_uppercase_ratio',
            # Path features
            'path_length', 'num_directories', 'path_entropy', 'has_double_slash',
            'directory_length_mean', 'directory_length_max', 'directory_length_min',
            'path_special_ratio',
            # Query features
            'num_params', 'query_length', 'has_suspicious_params',
            'param_entropy', 'param_special_ratio',
            # Security features
            'has_https', 'has_port', 'suspicious_tld', 'has_fragment',
            'has_redirect', 'has_obfuscation',
            # Content indicators
            'has_suspicious_words', 'suspicious_word_count', 'suspicious_word_ratio',
            'has_executable', 'has_archive',
            # Age-specific features
            'kid_unsafe_words', 'teen_unsafe_words', 'kid_unsafe_ratio',
            'teen_unsafe_ratio', 'kid_unsafe_score', 'teen_unsafe_score'
        ]

        # Initialize missing columns with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[feature_cols].values
        y = df['is_blocked'].values

        # Balance dataset
        X_balanced, y_balanced = balance_dataset(X, y)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create models directory with versioning
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Create versioned directory
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = os.path.join(models_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'feature_columns': feature_cols,
            'dataset_size': len(df),
            'positive_samples': sum(y),
            'negative_samples': len(y) - sum(y)
        }

        # Save scaler and feature columns
        joblib.dump(scaler, os.path.join(version_dir, 'url_scaler.pkl'))
        joblib.dump(feature_cols, os.path.join(version_dir, 'feature_cols.pkl'))
        joblib.dump(metadata, os.path.join(version_dir, 'metadata.pkl'))

        # Create symlink to latest version
        latest_link = os.path.join(models_dir, 'latest')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(version_dir, latest_link)

        models = {}

        # 1. K-Nearest Neighbors with hyperparameter tuning
        logging.info("\nTraining KNN Model with Grid Search...")
        knn_params = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'cosine'],
            'p': [1, 2]
        }

        # Use F1 score for optimization
        f1_scorer = make_scorer(f1_score)
        knn = GridSearchCV(
            KNeighborsClassifier(),
            knn_params,
            scoring=f1_scorer,
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
            verbose=1
        )
        knn.fit(X_train_scaled, y_train)

        logging.info(f"Best KNN parameters: {knn.best_params_}")
        knn_pred = knn.predict(X_test_scaled)
        models['knn'] = {
            'model': knn.best_estimator_,
            'metrics': evaluate_model(y_test, knn_pred, "KNN"),
            'best_params': knn.best_params_
        }
        joblib.dump(knn.best_estimator_, os.path.join(version_dir, 'knn_model.pkl'))

        # 2. Support Vector Machine with hyperparameter tuning
        logging.info("\nTraining SVM Model with Grid Search...")
        svm_params = {
            'C': [0.1, 1, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear'],
            'class_weight': [{0: 1.0, 1: 2.0}, 'balanced']
        }

        svm = GridSearchCV(
            SVC(probability=True),
            svm_params,
            scoring=f1_scorer,
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
            verbose=1
        )
        svm.fit(X_train_scaled, y_train)

        logging.info(f"Best SVM parameters: {svm.best_params_}")
        svm_pred = svm.predict(X_test_scaled)
        models['svm'] = {
            'model': svm.best_estimator_,
            'metrics': evaluate_model(y_test, svm_pred, "SVM"),
            'best_params': svm.best_params_
        }
        joblib.dump(svm.best_estimator_, os.path.join(version_dir, 'svm_model.pkl'))

        # 3. Naive Bayes with hyperparameter tuning
        logging.info("\nTraining Naive Bayes Model with Grid Search...")
        nb_params = {
            'var_smoothing': [1e-9, 1e-7, 1e-5],
            'priors': [None, [0.6, 0.4], [0.5, 0.5]]
        }

        nb = GridSearchCV(
            GaussianNB(),
            nb_params,
            scoring=f1_scorer,
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
            verbose=1
        )

        # Pre-process data for Naive Bayes
        epsilon = 1e-10
        X_train_scaled_nb = X_train_scaled + epsilon
        nb.fit(X_train_scaled_nb, y_train)

        logging.info(f"Best Naive Bayes parameters: {nb.best_params_}")
        nb_pred = nb.predict(X_test_scaled)
        models['nb'] = {
            'model': nb.best_estimator_,
            'metrics': evaluate_model(y_test, nb_pred, "Naive Bayes"),
            'best_params': nb.best_params_
        }
        joblib.dump(nb.best_estimator_, os.path.join(version_dir, 'nb_model.pkl'))

        # Generate evaluation report
        generate_evaluation_report(models, version_dir)

        return models

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

def generate_evaluation_report(models, output_dir):
    """Generate comprehensive evaluation report for all models"""
    report = {
        'models': {},
        'summary': {
            'best_model': None,
            'best_f1': 0,
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'improvement': None
        }
    }

    # Compare with previous version if exists
    models_dir = Path(output_dir).parent
    previous_versions = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name != 'latest'])

    if len(previous_versions) > 1:
        # Get second-to-last version
        prev_version = previous_versions[-2]
        prev_report_path = prev_version / 'evaluation_report.json'

        if prev_report_path.exists():
            with open(prev_report_path) as f:
                prev_report = json.load(f)

            # Calculate improvement metrics
            current_best_f1 = max(m['metrics']['f1'] for m in models.values())
            prev_best_f1 = prev_report['summary']['best_f1']

            report['summary']['improvement'] = {
                'previous_best_f1': prev_best_f1,
                'current_best_f1': current_best_f1,
                'f1_delta': current_best_f1 - prev_best_f1,
                'percent_improvement': ((current_best_f1 - prev_best_f1) / prev_best_f1) * 100
            }

    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Generate metrics and plots for each model
    for model_name, model_data in models.items():
        metrics = model_data['metrics']
        report['models'][model_name] = {
            'metrics': metrics,
            'best_params': model_data['best_params']
        }

        # Update summary
        if metrics['f1'] > report['summary']['best_f1']:
            report['summary']['best_model'] = model_name
            report['summary']['best_f1'] = metrics['f1']

        # Generate metrics plot
        plt.figure(figsize=(10, 6))
        plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'],
                [metrics['accuracy'], metrics['precision'],
                 metrics['recall'], metrics['f1']])
        plt.title(f'{model_name} Performance Metrics')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(plots_dir, f'{model_name}_metrics.png'))
        plt.close()

    # Save report
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logging.info(f"\nEvaluation report saved to: {report_path}")

    # Generate comparison plot if improvement data exists
    if report['summary']['improvement']:
        plt.figure(figsize=(10, 6))
        plt.bar(['Previous', 'Current'],
                [report['summary']['improvement']['previous_best_f1'],
                 report['summary']['improvement']['current_best_f1']])
        plt.title('Model Performance Improvement')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'))
        plt.close()

def calculate_age_based_risk(predictions, features, age_group='kid'):
    """
    Calculate risk level based on age group and features
    Args:
        predictions: Dict of model predictions
        features: URL features
        age_group: 'kid' or 'teen'
    Returns:
        tuple: (risk_level, risk_score)
    """
    # Base risk from model predictions
    ensemble_score = (
        0.4 * predictions['svm']['probability'] +
        0.35 * predictions['knn']['probability'] +
        0.25 * predictions['nb']['probability']
    )

    # Age-specific risk factors
    age_multipliers = {
        'kid': {
            'kid_unsafe_words': 2.0,
            'suspicious_tld': 1.5,
            'has_suspicious_words': 1.5,
            'has_executable': 2.0
        },
        'teen': {
            'teen_unsafe_words': 1.5,
            'suspicious_tld': 1.2,
            'has_suspicious_words': 1.2,
            'has_executable': 1.5
        }
    }

    multiplier = age_multipliers[age_group]
    risk_score = ensemble_score

    # Apply age-specific risk multipliers
    for feature, weight in multiplier.items():
        if feature in features and features[feature] > 0:
            risk_score *= weight

    # Cap risk score at 1.0
    risk_score = min(risk_score, 1.0)

    # Determine risk level
    if risk_score > 0.8:
        risk_level = "High"
    elif risk_score > 0.5:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return risk_level, risk_score

def ensemble_predict(url_features, threshold=0.65, models_dir='models/latest', age_group='kid'):
    """
    Make predictions using ensemble of models with adjustable threshold
    Returns:
        tuple: (is_unsafe, probability, predictions)
        - is_unsafe: bool indicating if URL is classified as unsafe
        - probability: float indicating probability of URL being unsafe
        - predictions: dict containing individual model predictions and probabilities
    """
    try:
        # Load models and scaler
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, models_dir)

        knn = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
        svm = joblib.load(os.path.join(models_dir, 'svm_model.pkl'))
        nb = joblib.load(os.path.join(models_dir, 'nb_model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'url_scaler.pkl'))
        feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))

        # Convert url_features to list if it's a string
        if isinstance(url_features, str):
            url_features = list(extract_url_features(url_features).values())

        # Ensure features array matches expected features
        if len(url_features) != len(feature_cols):
            logging.error(f"Mismatch in feature length: Expected {len(feature_cols)} features, got {len(url_features)}")
            logging.error(f"Feature columns: {feature_cols}")
            logging.error(f"Extracted features: {url_features}")
            raise ValueError(f"Expected {len(feature_cols)} features, got {len(url_features)}")

        # Scale features
        features_scaled = scaler.transform([url_features])
        logging.info(f"Scaled features: {features_scaled}")

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

        # Dynamic threshold based on age group
        age_thresholds = {
            'kid': 0.6,  # More strict for kids
            'teen': 0.7,  # Moderate for teens
            'adult': 0.8  # More lenient for adults
        }
        effective_threshold = age_thresholds.get(age_group, threshold)

        # Enhanced ensemble voting with feature-based boosting
        base_features = extract_url_features(url_features) if isinstance(url_features, str) else dict(zip(feature_cols, url_features))

        # Calculate risk with boosted weighting
        risk_level, risk_score = calculate_age_based_risk(
            predictions,
            base_features,
            age_group
        )

        # Apply additional safety checks
        if base_features.get('is_ip_address', 0) == 1:
            risk_score *= 1.2  # Increase risk for IP-based URLs
        if base_features.get('has_https', 0) == 0:
            risk_score *= 1.1  # Increase risk for non-HTTPS
        if base_features.get('suspicious_word_count', 0) > 2:
            risk_score *= 1.3  # Significant increase for multiple suspicious words

        # Enhanced result with age-specific risk assessment
        result = {
            'is_unsafe': bool(risk_score > threshold),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'age_group': age_group,
            'model_predictions': predictions
        }

        # Visualize the predictions
        plt.figure(figsize=(8, 6))
        plt.bar(['Risk Score'], [risk_score])
        plt.title('Risk Score for URL')
        plt.ylim(0, 1)
        plt.ylabel('Risk Score')
        plt.show()

        return result['is_unsafe'], result['risk_score'], result

    except Exception as e:
        logging.error(f"Error during ensemble prediction: {e}")
        raise

if __name__ == "__main__":
    train_models()