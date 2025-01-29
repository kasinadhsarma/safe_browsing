import os
from fastapi import FastAPI, HTTPException, Form, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import time
import json
import logging
import re
import traceback
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, select,
    Boolean, Integer, JSON, Float, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, ConfigDict
import tempfile
from ml.ai.training import train_models, predict_url, calculate_age_based_risk
from ml.ai.dataset import extract_url_features
import uvicorn
import socket
import whois
import pickle
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from urllib.parse import urlparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
# Suppress only the InsecureRequestWarning from urllib3
urllib3.disable_warnings(InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safebrowsing.log'),
        logging.StreamHandler()
    ]
)

# Database setup
DATABASE_PATH = "./safebrowsing.db"
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)  # Remove existing database to avoid schema conflicts

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create database directory with proper permissions
db_dir = os.path.dirname(DATABASE_PATH)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, mode=0o777)

# Create empty database file with write permissions
with open(DATABASE_PATH, 'w') as f:
    pass
os.chmod(DATABASE_PATH, 0o666)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global ML model and preprocessing objects
knn_model = None
svm_model = None
naive_bayes_model = None
best_model = None
url_scaler = None
feature_cols = None

# Database Models
class Activity(Base):
    __tablename__ = "activities"
    id = Column(String, primary_key=True)
    url = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)  # blocked, allowed, override, warning
    category = Column(String)
    risk_level = Column(String)
    ml_scores = Column(JSON, nullable=True)  # Store detailed ML predictions
    age_group = Column(String)  # kid, teen, adult
    block_reason = Column(String, nullable=True)  # Reason for blocking
    trust_factors = Column(JSON, nullable=True)  # Store trust factors (HTTPS, domain age, etc.)
    risk_factors = Column(JSON, nullable=True)  # Store risk factors (suspicious words, etc.)

class Setting(Base):
    __tablename__ = "settings"
    id = Column(String, primary_key=True)
    key = Column(String, unique=True)
    value = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow)

class MLMetrics(Base):
    __tablename__ = "ml_metrics"
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_data_size = Column(Integer)

# Create all tables
try:
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created successfully")
except Exception as e:
    logging.error(f"Error creating database tables: {e}")
    raise

# Pydantic models
class DashboardStats(BaseModel):
    total_sites: int
    blocked_sites: int
    recent_activities: List[Dict[str, Any]]
    ml_metrics: Dict[str, Dict[str, float]]
    risk_distribution: Dict[str, int]

# FastAPI App
app = FastAPI(
    title="Safe Browsing API",
    description="API for Safe Browsing extension with ML-powered content filtering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifecycle Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global knn_model, svm_model, naive_bayes_model, best_model, url_scaler, feature_cols

    try:
        # Load model and preprocessing objects
        logging.info("Loading URL classification models...")
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'latest')
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            logging.info(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            # Train models if they don't exist
            trained_models, scaler, _ = train_models(save_dir=model_dir)
            logging.info("Models trained successfully")

        # Map model files to variables
        model_files = {
            'knn_model.pkl': ('knn_model', KNeighborsClassifier),
            'svm_model.pkl': ('svm_model', SVC),
            'naive_bayes_model.pkl': ('naive_bayes_model', GaussianNB),
            'url_scaler.pkl': ('url_scaler', StandardScaler),
            'feature_cols.pkl': ('feature_cols', list)
        }

        # Load each model with detailed logging
        for filename, (var_name, expected_type) in model_files.items():
            filepath = os.path.join(model_dir, filename)
            logging.info(f"Loading {filename} from {filepath}")
            
            if not os.path.exists(filepath):
                logging.error(f"Model file not found: {filepath}")
                continue
                
            try:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                    if not isinstance(model, expected_type):
                        logging.error(f"Invalid model type for {filename}")
                        continue
                    globals()[var_name] = model
                    logging.info(f"Successfully loaded {filename}")
            except Exception as e:
                logging.error(f"Error loading {filename}: {str(e)}")
                continue

        # Verify models loaded correctly
        required_models = ['knn_model', 'svm_model', 'naive_bayes_model', 'url_scaler', 'feature_cols']
        missing_models = [model for model in required_models if globals().get(model) is None]
        
        if missing_models:
            logging.error(f"Missing required models: {missing_models}")
            raise ValueError("Not all required models were loaded successfully")
        
        logging.info("All URL classification models loaded successfully")
        yield

    except Exception as e:
        logging.error(f"Startup error: {e}")
        raise
    finally:
        logging.info("Shutting down")

app.lifespan = lifespan

# Utility Functions
def load_model(path: str):
    """Load a model from a pickle file"""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is None:
                model.feature_names_in_ = feature_cols if 'feature_cols' in globals() else []
            return model
    except Exception as e:
        logging.error(f"Error loading model {path}: {e}")
        return None

def get_domain_age(domain: str) -> Optional[int]:
    """Get the age of a domain in days"""
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if (creation_date):
            age = (datetime.now() - creation_date).days
            return age
    except whois.parser.PywhoisError as e:
        logging.warning(f"WHOIS lookup failed for {domain}: {e}")
    except Exception as e:
        logging.warning(f"Could not get domain age for {domain}: {e}")
    return None

def calculate_trust_factors(url: str, features: dict) -> dict:
    """Calculate trust factors for a URL"""
    trust_factors = {
        'has_https': features.get('has_https', 0),
        'domain_age': None,
        'trusted_domain': False
    }

    try:
        domain = urlparse(url).netloc.lower()
        trust_factors['domain_age'] = get_domain_age(domain)

        trusted_domains = {'google.com','youtube.com' ,'chrome://new-tab-page/','github.com', 'python.org', 'wikipedia.org'}
        trust_factors['trusted_domain'] = any(td in domain for td in trusted_domains)

    except Exception as e:
        logging.error(f"Error calculating trust factors: {e}")

    return trust_factors

def calculate_risk_factors(features: dict) -> dict:
    """Calculate risk factors for a URL"""
    return {
        'is_ip_address': features.get('is_ip_address', 0),
        'suspicious_word_count': features.get('suspicious_word_count', 0),
        'suspicious_tld': features.get('suspicious_tld', 0),
    }

# API Endpoints
@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    db = SessionLocal()
    try:
        # Get recent activities
        activities = db.query(Activity)\
            .order_by(Activity.timestamp.desc())\
            .limit(10)\
            .all()

        # Calculate stats
        total = db.query(Activity).count()
        blocked = db.query(Activity)\
            .filter(Activity.action == "blocked")\
            .count()

        # Get risk distribution
        risk_dist = {}
        for risk in ["low", "medium", "high"]:
            count = db.query(Activity)\
                .filter(Activity.risk_level.ilike(risk))\
                .count()
            risk_dist[risk.capitalize()] = count

        # Get ML metrics
        ml_metrics = {}
        latest_metrics = db.query(MLMetrics)\
            .order_by(MLMetrics.timestamp.desc())\
            .first()

        if latest_metrics:
            accuracy = latest_metrics.accuracy if latest_metrics.accuracy is not None else 0.0
            precision = latest_metrics.precision if latest_metrics.precision is not None else 0.0
            recall = latest_metrics.recall if latest_metrics.recall is not None else 0.0
            f1 = latest_metrics.f1_score if latest_metrics.f1_score is not None else 0.0
            ml_metrics['model'] = {
                'accuracy': accuracy * 100,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100
            }

        # Include recent activities with individual model scores
        activities_with_scores = []
        for activity in activities:
            scores = activity.ml_scores or {}
            individual_scores = scores.get('individual_models', {})
            
            activities_with_scores.append({
                "url": activity.url,
                "timestamp": activity.timestamp.isoformat(),
                "action": activity.action,
                "category": activity.category,
                "risk_level": activity.risk_level,
                "ml_scores": {
                    **scores,
                    "KNN": individual_scores.get("KNN", 0.0),
                    "SVM": individual_scores.get("SVM", 0.0),
                    "NB": individual_scores.get("NB", 0.0)
                },
                "age_group": activity.age_group,
                "block_reason": activity.block_reason,
                "trust_factors": activity.trust_factors,
                "risk_factors": activity.risk_factors
            })

        return {
            "total_sites": total,
            "blocked_sites": blocked,
            "recent_activities": activities_with_scores,
            "ml_metrics": ml_metrics,
            "risk_distribution": risk_dist
        }

    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/api/check-url")
async def check_url(url: str = Form(...), age_group: str = Form("kid")):
    """Check if a URL should be blocked based on ML predictions and age group"""
    db = SessionLocal()
    try:
        # Add adult content check
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        adult_domains = {
            'brazzers.com', 'pornhub.com', 'xvideos.com', 'xnxx.com',
            'youporn.com', 'redtube.com', 'xhamster.com'
        }
        
        if any(adult in domain for adult in adult_domains):
            ml_scores = {
                "kid": 1.0,
                "teen": 0.8,
                "adult": 0.6,
                "individual_models": {"KNN": 100.0, "SVM": 100.0, "NB": 100.0}
            } if age_group == 'kid' else {
                "kid": 1.0,
                "teen": 0.8,
                "adult": 0.6,
                "individual_models": {"KNN": 80.0, "SVM": 80.0, "NB": 80.0}
            }
            activity = Activity(
                id=os.urandom(16).hex(),
                url=url.strip(),
                action="blocked",
                category="Adult Content",
                risk_level="high",
                ml_scores=ml_scores,
                age_group=age_group,
                block_reason="Adult domain restricted",
                trust_factors={"has_https": 1, "trusted_domain": False},
                risk_factors={"is_adult_content": 1, "suspicious_word_count": 1}
            )
            db.add(activity)
            db.commit()
            return {
                "blocked": True,
                "risk_level": "high",
                "category": "Adult Content",
                "risk_score": 1.0 if age_group == 'kid' else 0.8,
                "trust_factors": activity.trust_factors,
                "risk_factors": activity.risk_factors,
                "ml_scores": ml_scores,
                "block_reason": "Adult content restricted"
            }

        # Handle internal URLs and schemes
        parsed_url = urlparse(url)
        if parsed_url.scheme in ['chrome', 'about', 'file']:
            return {
                "blocked": False,
                "risk_level": "low",
                "category": "Internal",
                "risk_score": 0.1,  # Never return 0.0
                "trust_factors": {"has_https": 1, "trusted_domain": True},
                "risk_factors": {"is_ip_address": 0, "suspicious_word_count": 0},
                "ml_scores": {"kid": 0.1, "teen": 0.1, "adult": 0.1}
            }

        # Handle trusted domains
        trusted_domains = {
            'youtube.com', 'linkedin.com', 'github.com',
            'python.org', 'wikipedia.org', 'google.com'
        }
        domain = parsed_url.netloc.lower()
        if any(td in domain for td in trusted_domains):
            return {
                "blocked": False,
                "risk_level": "low",
                "category": "Trusted",
                "risk_score": 0.3,
                "trust_factors": {"has_https": 1, "trusted_domain": True},
                "risk_factors": {"is_ip_address": 0, "suspicious_word_count": 0},
                "ml_scores": {"kid": 0.3, "teen": 0.2, "adult": 0.1}
            }

        # Extract features
        features_dict = extract_url_features(url) or {}
        features_df = pd.DataFrame([features_dict], columns=feature_cols)

        if not features_dict:
            # Handle internal URLs
            if url.startswith('http://localhost:') or url.startswith('https://localhost:'):
                return {
                    "blocked": False,
                    "risk_level": "low",
                    "category": "Internal",
                    "risk_score": 0.0,
                    "trust_factors": {"has_https": 1, "trusted_domain": True},
                    "risk_factors": {"is_ip_address": 0, "suspicious_word_count": 0},
                    "ml_scores": {"kid": 0.0, "teen": 0.0, "adult": 0.0}
                }

        # Store individual model predictions
        individual_scores = {}
        
        try:
            # Get features and make predictions
            features = extract_url_features(url)
            if isinstance(features, dict):
                features_df = pd.DataFrame([features], columns=feature_cols)

            features_scaled = url_scaler.transform(features_df)
                
            # Get individual model predictions
            if knn_model is not None:
                individual_scores['KNN'] = float(knn_model.predict_proba(features_scaled)[0][1])
            if svm_model is not None:
                individual_scores['SVM'] = float(svm_model.predict_proba(features_scaled)[0][1])
            if naive_bayes_model is not None:
                individual_scores['NB'] = float(naive_bayes_model.predict_proba(features_scaled)[0][1])
        except Exception as e:
            logging.error(f"Error getting individual model predictions: {e}")
            individual_scores = {'KNN': 0.5, 'SVM': 0.5, 'NB': 0.5}

        # Make final prediction
        is_unsafe, risk_score, risk_level = predict_url(
            url=url,
            age_group=age_group,
            models_dir=os.path.join(os.path.dirname(__file__), 'models', 'latest')
        )

        # Ensure risk_score is not None
        risk_score = risk_score if risk_score is not None else 0.0

        # Ensure individual_scores are valid
        precision = float(np.mean(list(individual_scores.values()))) if individual_scores else 0.0
        precision = precision if precision is not None else 0.0

        # Store metrics in database
        metrics = MLMetrics(
            id=os.urandom(16).hex(),
            timestamp=datetime.utcnow(),
            model_name="ensemble",
            accuracy=float(risk_score) if risk_score is not None else 0.0,
            precision=precision,
            recall=float(risk_score) if risk_score is not None else 0.0,
            f1_score=float(risk_score) if risk_score is not None else 0.0,
            training_data_size=1000  # Update with actual training size
        )
        db.add(metrics)
        db.commit()

        # Calculate trust and risk factors
        trust_factors = calculate_trust_factors(url, features_dict)
        risk_factors = calculate_risk_factors(features_dict)

        # Calculate age-specific scores
        predictions = {
            'knn': {'probability': risk_score},
            'svm': {'probability': risk_score},
            'nb': {'probability': risk_score}
        }

        age_risk_level, age_risk_score = calculate_age_based_risk(
            predictions=predictions,
            features=features,
            age_group=age_group
        )

        # Only block if risk level is high
        should_block = risk_level == "high"

        # Prepare response
        result = {
            "blocked": should_block,
            "risk_level": risk_level,
            "risk_score": age_risk_score,
            "category": "Unsafe" if should_block else "Safe",
            "trust_factors": trust_factors,
            "risk_factors": risk_factors,
            "ml_scores": {
                "kid": min(risk_score * 1.5, 1.0),
                "teen": min(risk_score * 1.2, 1.0),
                "adult": risk_score,
                "individual_models": {
                    "KNN": individual_scores.get('KNN', 0.0) * 100,
                    "SVM": individual_scores.get('SVM', 0.0) * 100,
                    "NB": individual_scores.get('NB', 0.0) * 100
                }
            },
            "block_reason": "Model flagged high risk" if should_block else None
        }

        # Ensure ML scores are never 0.0
        result["ml_scores"] = {
            age: max(0.1, min(score, 1.0))
            for age, score in result["ml_scores"].items()
        }

        # Log activity
        activity = Activity(
            id=os.urandom(16).hex(),
            url=url.strip(),
            action="blocked" if should_block else "allowed",
            category=result["category"],
            risk_level=risk_level,
            ml_scores=result["ml_scores"],
            age_group=age_group,
            block_reason=result["block_reason"],
            trust_factors=trust_factors,
            risk_factors=risk_factors
        )
        db.add(activity)
        db.commit()

        # Update activity timestamp format
        activity.timestamp = datetime.utcnow()
        
        return result

    except Exception as e:
        logging.error(f"Error in check_url: {e}")
        return {
            "blocked": False,
            "risk_level": "medium",
            "category": "Error",
            "risk_score": 0.5,
            "trust_factors": {},
            "risk_factors": {},
            "ml_scores": {
                "kid": 0.5, "teen": 0.5, "adult": 0.5,
                "individual_models": {"KNN": 50.0, "SVM": 50.0, "NB": 50.0}
            },
            "block_reason": "Error during analysis"
        }
    finally:
        db.close()

@app.post("/api/activity")
async def log_activity(request: Request):
    form_data = await request.form()
    url = form_data.get("url", "")
    action = form_data.get("action", "")
    category = form_data.get("category", "Unknown")
    risk_level = form_data.get("risk_level", "Unknown")
    ml_scores = form_data.get("ml_scores", "{}")
    block_reason = form_data.get("block_reason", None)
    age_group = form_data.get("age_group", "kid")
    trust_factors = form_data.get("trust_factors", "{}")
    risk_factors = form_data.get("risk_factors", "{}")

    db = SessionLocal()
    try:
        # Parse and validate scores and factors
        try:
            ml_scores_dict = json.loads(ml_scores)
            valid_scores = {k: v for k, v in ml_scores_dict.items()
                          if isinstance(v, (int, float)) and v >= 0}
        except json.JSONDecodeError:
            valid_scores = {}

        try:
            trust_factors_dict = json.loads(trust_factors)
        except json.JSONDecodeError:
            trust_factors_dict = {}

        try:
            risk_factors_dict = json.loads(risk_factors)
        except json.JSONDecodeError:
            risk_factors_dict = {}

        activity = Activity(
            id=os.urandom(16).hex(),
            url=url.strip(),
            action=action.strip(),
            category=category.strip() or "Unknown",
            risk_level=risk_level.strip() or "Unknown",
            ml_scores=valid_scores,
            age_group=age_group.strip(),
            block_reason=block_reason.strip() if block_reason else None,
            trust_factors=trust_factors_dict,
            risk_factors=risk_factors_dict
        )
        db.add(activity)
        db.commit()
        db.refresh(activity)
        return {
            "url": activity.url,
            "timestamp": activity.timestamp,
            "action": activity.action,
            "category": activity.category,
            "risk_level": activity.risk_level,
            "ml_scores": activity.ml_scores,
            "age_group": activity.age_group,
            "block_reason": activity.block_reason,
            "trust_factors": activity.trust_factors,
            "risk_factors": activity.risk_factors
        }
    except Exception as e:
        logging.error(f"Error logging activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/activities")
async def get_activities(limit: int = 100, offset: int = 0):
    """Get paginated list of activities"""
    db = SessionLocal()
    try:
        activities = db.query(Activity)\
            .order_by(Activity.timestamp.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()

        return [
            {
                "url": activity.url,
                "timestamp": activity.timestamp.isoformat(),
                "action": activity.action,
                "category": activity.category,
                "risk_level": activity.risk_level,
                "ml_scores": "N/A" if not activity.ml_scores else activity.ml_scores,
                "age_group": activity.age_group,
                "block_reason": activity.block_reason,
                "trust_factors": activity.trust_factors,
                "risk_factors": activity.risk_factors,
                "time": activity.timestamp.strftime("%b %d, %Y, %I:%M %p")
            }
            for activity in activities
        ]
    except Exception as e:
        logging.error(f"Error getting activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts"""
    db = SessionLocal()
    try:
        # Get activities with high risk or blocked status from last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        alerts = db.query(Activity)\
            .filter(
                (Activity.risk_level == "high") |
                (Activity.action == "blocked")
            )\
            .filter(Activity.timestamp > yesterday)\
            .order_by(Activity.timestamp.desc())\
            .all()

        return [
            {
                "url": alert.url,
                "timestamp": alert.timestamp.isoformat(),
                "action": alert.action,
                "category": alert.category,
                "risk_level": alert.risk_level,
                "ml_scores": alert.ml_scores,
                "age_group": alert.age_group,
                "block_reason": alert.block_reason,
                "trust_factors": alert.trust_factors,
                "risk_factors": alert.risk_factors
            }
            for alert in alerts
        ]
    except Exception as e:
        logging.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Start the server
if __name__ == "__main__":
    port = 8000
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))  # Check if port is free
        logging.info(f"Starting server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except socket.error:
        logging.error(f"Port {port} is already in use. Please free the port and try again.")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"Server startup error: {str(e)}")
        raise

def get_page_content(url):
    """Fetch content with minimal processing."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 Chrome/91.0.4472.124'}
        session = requests.Session()
        response = session.get(
            url,
            headers=headers,
            timeout=(2, 5),  # Aggressive timeouts
            verify=False,  # Skip SSL verification
            allow_redirects=False
        )
        
        if not response.ok:
            return ""

        text = BeautifulSoup(response.text, 'html.parser').get_text()
        return text[:5000]  # Limit content size
    except Exception as e:
        logging.error(f"Error loading image from URL {url}: {e}")
        return ""
