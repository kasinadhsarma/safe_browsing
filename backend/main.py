"""
Safe Browsing for Kids Under Parental Supervision Using Machine Learning

ABSTRACT:
Since birth, 21st century children have access to various websites through their devices. However, not all internet content is child-friendly, and children may encounter violent or inappropriate images that can negatively impact their development. Many websites contain ads that may display unsuitable content. This system provides parental controls using three supervised machine learning techniques (K-Nearest Neighbor, Support Vector Machine, and Naive Bayes Classifier) for URL classification, combined with deep learning for image detection to block inappropriate content.

Key Features:
1. Parental control settings for different age groups
2. Real-time URL classification using ensemble ML models
3. Image content detection using deep learning
4. Detailed activity logging and reporting
5. Customizable risk thresholds for different age groups
"""

from fastapi import FastAPI, HTTPException, Form, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
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
from ml.ai.training import train_models, predict_url
from ml.ai.dataset import extract_url_features
import uvicorn
import socket
import whois
import pickle

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
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global ML model and preprocessing objects
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
    global best_model, url_scaler, feature_cols

    try:
        # Load model and preprocessing objects
        logging.info("Loading URL classification model...")
        model_dir = 'backend/ml/ai/models/latest'
        best_model = load_model(os.path.join(model_dir, 'best_model.pkl'))
        url_scaler = load_model(os.path.join(model_dir, 'url_scaler.pkl'))
        feature_cols = load_model(os.path.join(model_dir, 'feature_cols.pkl'))

        if best_model and url_scaler and feature_cols:
            logging.info("URL classification model loaded successfully")
        else:
            logging.warning("URL model not loaded")

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
    with open(path, 'rb') as file:
        return pickle.load(file)

def determine_risk_level(predictions: dict, age_group: str) -> Tuple[str, bool, str]:
    """Determine risk level and blocking decision based on predictions and age group"""
    try:
        # Get age-specific risk scores
        kid_unsafe = predictions.get("ml_scores", {}).get("kid_unsafe", 0.0)
        teen_unsafe = predictions.get("ml_scores", {}).get("teen_unsafe", 0.0)
        suspicious = predictions.get("ml_scores", {}).get("suspicious", 0.0)

        # Calculate risk score based on age group
        if age_group == "kid":
            risk_score = max(kid_unsafe, suspicious * 0.5)
            high_threshold = 0.7
            medium_threshold = 0.4
        elif age_group == "teen":
            risk_score = max(teen_unsafe, suspicious * 0.3)
            high_threshold = 0.8
            medium_threshold = 0.6
        else:  # adult
            risk_score = max(teen_unsafe * 0.5, suspicious * 0.2)
            high_threshold = 0.9
            medium_threshold = 0.7

        # Determine risk level and blocking decision
        if risk_score >= high_threshold:
            return "High", True, f"High risk content detected (score: {risk_score:.2f})"
        elif risk_score >= medium_threshold:
            should_block = age_group == "kid"  # Block medium risk content only for kids
            return "Medium", should_block, f"Medium risk content (score: {risk_score:.2f})"
        else:
            return "Low", False, ""

    except Exception as e:
        logging.error(f"Error in determine_risk_level: {e}")
        return "Unknown", True, "Error determining risk level"  # Fail safe by blocking

def get_domain_age(domain: str) -> Optional[int]:
    """Get the age of a domain in days"""
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            age = (datetime.now() - creation_date).days
            return age
    except Exception as e:
        logging.warning(f"Could not get domain age for {domain}: {e}")
    return None

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
        for risk in ["Low", "Medium", "High"]:
            count = db.query(Activity)\
                .filter(Activity.risk_level == risk)\
                .count()
            risk_dist[risk] = count

        # Get ML metrics (if model has been trained)
        ml_metrics = {}
        if best_model:
            # Get metrics from MLMetrics table
            latest_metrics = db.query(MLMetrics)\
                .order_by(MLMetrics.timestamp.desc())\
                .first()
            
            if latest_metrics:
                ml_metrics['model'] = {
                    'accuracy': latest_metrics.accuracy,
                    'precision': latest_metrics.precision,
                    'recall': latest_metrics.recall,
                    'f1_score': latest_metrics.f1_score
                }

        return {
            "total_sites": total,
            "blocked_sites": blocked,
            "recent_activities": [
                {
                    "url": a.url,
                    "timestamp": a.timestamp.isoformat(),
                    "action": a.action,
                    "category": a.category,
                    "risk_level": a.risk_level,
                    "ml_scores": a.ml_scores,
                    "age_group": a.age_group,
                    "block_reason": a.block_reason
                } for a in activities
            ],
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
    try:
        # Extract features
        features = extract_url_features(url)
        if not features:
            # Handle internal URLs
            if url.startswith('http://localhost:') or url.startswith('https://localhost:'):
                result = {
                    "blocked": False,
                    "risk_level": "Low",
                    "category": "Internal",
                    "probability": 0.0,
                    "predictions": {
                        "risk_features": {
                            "kid_unsafe_score": 0.0,
                            "teen_unsafe_score": 0.0,
                            "suspicious_word_count": 0.0
                        }
                    },
                    "ml_scores": {
                        "kid_unsafe": 0.0,
                        "teen_unsafe": 0.0,
                        "suspicious": 0.0
                    }
                }
            else:
                result = {
                    "blocked": False,
                    "risk_level": "Unknown",
                    "category": "Unknown",
                    "probability": 0.0,
                    "predictions": {
                        "risk_features": {
                            "kid_unsafe_score": 0.0,
                            "teen_unsafe_score": 0.0,
                            "suspicious_word_count": 0.0
                        }
                    },
                    "ml_scores": {
                        "kid_unsafe": 0.0,
                        "teen_unsafe": 0.0,
                        "suspicious": 0.0
                    }
                }
        else:
            # Make prediction with enhanced URL analysis
            try:
                prediction = predict_url(url)
                
                result = {
                    "blocked": prediction['is_blocked'],
                    "probability": prediction['confidence'],
                    "risk_level": "High" if prediction['risk_features']['kid_unsafe_score'] > 0.7 else 
                                "Medium" if prediction['risk_features']['kid_unsafe_score'] > 0.4 else "Low",
                    "category": prediction['category'],
                    "url": url,
                    "predictions": {
                        "risk_features": prediction['risk_features']
                    },
                    "ml_scores": {
                        "kid_unsafe": prediction['risk_features']['kid_unsafe_score'],
                        "teen_unsafe": prediction['risk_features']['teen_unsafe_score'],
                        "suspicious": float(prediction['risk_features']['suspicious_word_count'])
                    }
                }
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                result = {
                    "blocked": False,
                    "risk_level": "Unknown",
                    "probability": 0.0,
                    "predictions": {
                        "risk_features": {
                            "kid_unsafe_score": 0.0,
                            "teen_unsafe_score": 0.0,
                            "suspicious_word_count": 0.0
                        }
                    },
                    "category": "Unknown",
                    "ml_scores": {
                        "kid_unsafe": 0.0,
                        "teen_unsafe": 0.0,
                        "suspicious": 0.0
                    }
                }

        # Log activity
        db = SessionLocal()
        try:
            # Determine risk level and blocking decision
            risk_level, should_block, block_reason = determine_risk_level(result, age_group)

            # Update result with new risk assessment
            result["blocked"] = should_block
            result["risk_level"] = risk_level
            result["block_reason"] = block_reason if should_block else ""

            activity = Activity(
                id=os.urandom(16).hex(),
                url=url.strip(),  # Remove any whitespace
                action="blocked" if should_block else "allowed",
                category=result.get("category", "Internal"),
                risk_level=risk_level,
                ml_scores={k: v for k, v in result.get("ml_scores", {}).items()
                          if isinstance(v, (int, float)) and v >= 0},  # Filter valid scores
                age_group=age_group.strip(),
                block_reason=block_reason if should_block else None
            )
            db.add(activity)
            db.commit()
        except Exception as db_error:
            logging.error(f"Database error: {db_error}")
            db.rollback()
        finally:
            db.close()

        return {
            **result,
            "ml_scores": result.get("ml_scores", {
                "kid_unsafe": 0.0,
                "teen_unsafe": 0.0,
                "suspicious": 0.0
            })
        }

    except Exception as e:
        logging.error(f"Error checking URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

    """Log browsing activity"""
    db = SessionLocal()
    try:
        # Parse and validate ML scores
        try:
            ml_scores_dict = json.loads(ml_scores)
            valid_scores = {k: v for k, v in ml_scores_dict.items()
                          if isinstance(v, (int, float)) and v >= 0}
        except json.JSONDecodeError:
            valid_scores = {}

        activity = Activity(
            id=os.urandom(16).hex(),
            url=url.strip(),
            action=action.strip(),
            category=category.strip() or "Unknown",
            risk_level=risk_level.strip() or "Unknown",
            ml_scores=valid_scores,
            age_group=age_group.strip(),
            block_reason=block_reason.strip() if block_reason else None
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
            "block_reason": activity.block_reason
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
                "ml_scores": activity.ml_scores,
                "age_group": activity.age_group,
                "block_reason": activity.block_reason
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
                (Activity.risk_level == "High") |
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
                "block_reason": alert.block_reason
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
