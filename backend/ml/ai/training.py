import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from .dataset import load_url_data, process_urls_parallel, FIXED_FEATURE_COLUMNS, extract_url_features
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def balance_dataset(X, y):
    """Balance dataset using SMOTE."""
    smote = SMOTE(random_state=42, n_jobs=-1)  # Use all CPU cores
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate rates
    true_positives = sum((y_true == 1) & (y_pred == 1))
    false_positives = sum((y_true == 0) & (y_pred == 1))
    false_negatives = sum((y_true == 1) & (y_pred == 0))
    
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

def predict_url(url):
    """Predict URL safety using trained models."""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
        
        # Load models and preprocessing objects
        scaler = joblib.load(os.path.join(model_dir, 'url_scaler.pkl'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
        
        # Load individual models
        models = {
            'knn': joblib.load(os.path.join(model_dir, 'knn_model.pkl')),
            'svm': joblib.load(os.path.join(model_dir, 'svm_model.pkl')),
            'nb': joblib.load(os.path.join(model_dir, 'nb_model.pkl'))
        }
        
        # Extract features and ensure feature names are preserved
        features_dict = extract_url_features(url)
        # Create DataFrame with feature names to maintain column order
        features_df = pd.DataFrame([features_dict], columns=feature_cols)
        # Scale features and maintain DataFrame structure
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=feature_cols
        )
        
        # Get predictions from each model
        predictions = {}
        for name, model in models.items():
            # Get predictions maintaining feature names
            proba = model.predict_proba(features_scaled.values)[0]
            predictions[name] = {
                'is_unsafe': bool(model.predict(features_scaled.values)[0]),
                'probability': float(proba[1])  # Probability of unsafe class
            }
        
        # Weight predictions based on model reliability
        weights = {'knn': 0.25, 'svm': 0.45, 'nb': 0.30}
        weighted_probability = sum(
            pred['probability'] * weights[name] 
            for name, pred in predictions.items()
        )
        
        # Overall safety assessment
        is_unsafe = weighted_probability > 0.45  # Lower threshold for better detection
        
        return predictions, is_unsafe, weighted_probability
        
    except Exception as e:
        logging.error(f"Error during URL prediction: {e}")
        raise

def train_models():
    """Train models with parallel feature extraction."""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Load URLs
        logging.info("Loading dataset...")
        dataset = load_url_data()
        if dataset.empty:
            logging.error("Dataset is empty. Check input files.")
            return

        # Extract features in parallel 
        logging.info("Extracting features using parallel processing...")
        features = process_urls_parallel(dataset['url'], max_workers=50)
        
        # Convert features to DataFrame
        X = pd.DataFrame(features, columns=FIXED_FEATURE_COLUMNS)
        y = dataset['is_blocked'].values

        # Remove rows with all zero features
        valid_rows = (X != 0).any(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]

        logging.info(f"Successfully processed {len(X)} URLs out of {len(dataset)}")

        # Split and balance dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

        # Scale features with named columns
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_balanced),
            columns=X_train_balanced.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        # Initialize models with multicore support where possible
        models = {
            'knn': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean',
                n_jobs=-1  # Use all cores
            ),
            'svm': SVC(
                C=0.8,
                kernel='rbf',
                gamma='scale',
                probability=True,
                class_weight='balanced',
                max_iter=1000
            ),
            'nb': GaussianNB(
                var_smoothing=1e-8,
                priors=None
            )
        }

        # Train and evaluate each model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in models.items():
            logging.info(f"Training {name} model...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=cv, scoring='f1', n_jobs=-1)
            logging.info(f"Cross-validation F1 scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train final model
            model.fit(X_train_scaled, y_train_balanced)
            y_pred = model.predict(X_test_scaled)
            evaluate_model(y_test, y_pred, name.upper())
            joblib.dump(model, os.path.join(save_dir, f'{name}_model.pkl'))

        # Create ensemble
        voting_clf = VotingClassifier(
            estimators=[
                ('knn', models['knn']),
                ('svm', models['svm']), 
                ('nb', models['nb'])
            ],
            voting='soft',
            weights=[0.25, 0.45, 0.30]  # Weight by performance
        )
        
        logging.info("Training ensemble model...")
        voting_clf.fit(X_train_scaled, y_train_balanced)
        y_pred_ensemble = voting_clf.predict(X_test_scaled)
        evaluate_model(y_test, y_pred_ensemble, "ENSEMBLE")
        joblib.dump(voting_clf, os.path.join(save_dir, 'ensemble_model.pkl'))

        # Save preprocessing objects with feature names
        scaler.feature_names_in_ = X_train_balanced.columns
        joblib.dump(scaler, os.path.join(save_dir, 'url_scaler.pkl'))
        joblib.dump(list(X_train_balanced.columns), os.path.join(save_dir, 'feature_cols.pkl'))

        logging.info("\nModel training completed successfully!")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    train_models()
