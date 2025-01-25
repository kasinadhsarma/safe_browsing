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
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.ai.training import train_models, predict_url
from ml.ai.dataset import extract_url_features, generate_dataset
from ml.ai.image_classification import ImageClassifier
import os
import logging
import re
import json
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, select,
    Boolean, Integer, JSON, Float, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, ConfigDict
import tempfile

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

# Global ML models
url_model: Optional[Dict] = None
image_model: Optional[ImageClassifier] = None

# Database Models
class Activity(Base):
    __tablename__ = "activities"
    id = Column(String, primary_key=True)
    url = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)  # blocked, allowed, override
    category = Column(String)
    risk_level = Column(String)
    ml_scores = Column(JSON, nullable=True)  # Store detailed ML predictions

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
    global url_model, image_model
    
    try:
        # Train models
        logging.info("Training URL classification models...")
        url_model = train_models()
        if url_model:
            logging.info("URL classification models loaded successfully")
        else:
            logging.warning("URL models not loaded")
            
        yield
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        raise
    finally:
        logging.info("Shutting down")

app.lifespan = lifespan

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

        # Get ML metrics (if any models have been trained)
        ml_metrics = {}
        if url_model:
            for name, data in url_model.items():
                if 'metrics' in data:
                    ml_metrics[name] = data['metrics']

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
                    "ml_scores": a.ml_scores
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

class URLCheckRequest(BaseModel):
    url: str
    age_group: str = "kid"  # Default to most restrictive setting

@app.post("/api/check-url")
async def check_url(url: str = Form(...), age_group: str = Form("kid")):
    """Check if a URL should be blocked"""
    try:
        # Extract features
        features = extract_url_features(url)
        if not features:
            result = {
                "blocked": False,
                "risk_level": "Unknown",
                "category": "Unknown",
                "probability": 0.0,
                "predictions": {}
            }
        else:
            # Make prediction with age-based risk assessment
            if url_model:
                is_blocked, risk_score, predictions = predict_url(
                    url,
                    threshold=0.7 if age_group == "kid" else 0.8,
                    models_dir='ml/ai/models/',
                    age_group=age_group
                )
                
                result = {
                    "blocked": is_blocked,
                    "risk_level": predictions["risk_level"],
                    "risk_score": float(risk_score),
                    "age_group": age_group,
                    "category": "Malicious" if is_blocked else "Safe",
                    "predictions": predictions["model_predictions"],
                    "unsafe_content": {
                        "kid_unsafe": predictions.get("kid_unsafe_words", 0),
                        "teen_unsafe": predictions.get("teen_unsafe_words", 0)
                    },
                    "security_flags": {
                        "suspicious_tld": features.get("suspicious_tld", 0),
                        "is_ip_address": features.get("is_ip_address", 0),
                        "has_https": features.get("has_https", 1)
                    }
                }
            else:
                result = {
                    "blocked": False,
                    "risk_level": "Unknown",
                    "probability": 0.0,
                    "predictions": {},
                    "category": "Unknown",
                    "ml_scores": {
                        "knn": 0.0,
                        "svm": 0.0,
                        "nb": 0.0
                    }
                }

        # Log activity
        db = SessionLocal()
        try:
            activity = Activity(
                id=os.urandom(16).hex(),
                url=url,
                action="blocked" if result["blocked"] else "allowed",
                category=result["category"],
                risk_level=result["risk_level"],
                ml_scores=result.get("ml_scores", {})
            )
            db.add(activity)
            db.commit()
        except Exception as db_error:
            logging.error(f"Database error: {db_error}")
            db.rollback()
        finally:
            db.close()

        return result

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
    """Log browsing activity"""
    db = SessionLocal()
    try:
        activity = Activity(
            id=os.urandom(16).hex(),
            url=url,
            action=action,
            category=category or "Unknown",
            risk_level=risk_level or "Unknown",
            ml_scores=json.loads(ml_scores)
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
            "ml_scores": activity.ml_scores
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
                "ml_scores": activity.ml_scores
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
                "ml_scores": alert.ml_scores
            }
            for alert in alerts
        ]
    except Exception as e:
        logging.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return False
            except socket.error:
                return True

    try:
        logging.info("Initializing server...")
        
        # Try ports from 8000 to 8010
        port = 8000
        while port < 8010:
            if not is_port_in_use(port):
                logging.info(f"Starting server on port {port}")
                uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
                break
            logging.info(f"Port {port} is in use, trying next port")
            port += 1
        else:
            logging.error("No available ports found between 8000")
    except Exception as e:
        logging.error(f"Server startup error: {str(e)}")
        raise
