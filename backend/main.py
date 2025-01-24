from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from ml.ai.training import URLClassifier, train
from ml.ai.dataset import URLDataset
import os
import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database setup
DATABASE_URL = "sqlite:///./safebrowsing.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global ML model
ml_model: Optional[URLClassifier] = None

# Database Models
class Activity(Base):
    __tablename__ = "activities"
    id = Column(String, primary_key=True)
    url = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)
    category = Column(String, nullable=True)
    risk_level = Column(String, nullable=True)

class ErrorLog(Base):
    __tablename__ = "error_logs"
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    error = Column(String)
    stack = Column(Text, nullable=True)
    url = Column(String, nullable=True)

# Pydantic models
class ActivityCreate(BaseModel):
    url: str
    action: str
    category: Optional[str] = None
    timestamp: Optional[datetime] = None

class ActivityResponse(BaseModel):
    url: str
    timestamp: datetime
    action: str
    category: Optional[str] = None
    risk_level: Optional[str] = None

class DashboardStats(BaseModel):
    total_sites: int
    blocked_sites: int
    allowed_sites: int
    visited_sites: int
    recent_activities: List[ActivityResponse]
    daily_stats: Dict[str, int]

class ErrorLogCreate(BaseModel):
    error: str
    stack: Optional[str] = None
    timestamp: Optional[datetime] = None
    url: Optional[str] = None

def get_domain_category(url: str) -> str:
    """Helper function to categorize URLs."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        # Video platforms
        if any(x in domain for x in ['youtube.com', 'vimeo.com', 'netflix.com']):
            return 'Video'
        
        # Social media
        if any(x in domain for x in ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com']):
            return 'Social Media'
        
        # Search engines
        if any(x in domain for x in ['google.com', 'bing.com', 'yahoo.com']):
            return 'Search'
        
        # Educational
        if any(x in domain for x in ['coursera.org', 'udemy.com', 'edx.org']) or '.edu' in domain:
            return 'Educational'
        
        # News
        if any(x in domain for x in ['news', 'cnn.com', 'bbc.com', 'nytimes.com']):
            return 'News'

        return 'General'
    except:
        return 'Unknown'

def check_url_patterns(url: str) -> Dict[str, Any]:
    """Check URL using pattern matching."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()

        # Known safe domains
        safe_domains = {
            'youtube.com', 'www.youtube.com',
            'google.com', 'www.google.com',
            'facebook.com', 'www.facebook.com',
            'twitter.com', 'www.twitter.com',
            'netflix.com', 'www.netflix.com',
            'disney.com', 'www.disney.com'
        }

        if hostname in safe_domains:
            return {
                "blocked": False,
                "category": "Safe",
                "risk_level": "Low",
                "method": "pattern"
            }

        # Check for inappropriate content
        risk_patterns = {
            "adult": r"(adult|xxx|porn|sex|nude|escort|dating|cam|strip|playboy|nsfw)",
            "gambling": r"(gambling|casino|bet|poker|lottery|blackjack|roulette|slot)",
            "violence": r"(gore|violence|death|torture|weapon|drug|cartel)",
            "malware": r"(crack|warez|keygen|hack|torrent|pirate|stolen)",
            "phishing": r"(phish|scam|fake|fraud|spam)"
        }

        full_url = (hostname + path + query).lower()
        for category, pattern in risk_patterns.items():
            if re.search(pattern, full_url, re.IGNORECASE):
                return {
                    "blocked": True,
                    "category": category.capitalize(),
                    "risk_level": "High",
                    "method": "pattern"
                }

        return None

    except Exception as e:
        logging.error(f"Error in pattern matching: {e}")
        return None

def check_url_ml(url: str) -> Dict[str, Any]:
    """Check URL using ML model."""
    try:
        if not ml_model:
            logging.warning("ML model not loaded, skipping ML check")
            return None

        # Extract features
        dataset = URLDataset([url], [0])
        features = dataset[0]['features']

        # Make prediction
        with torch.no_grad():
            prediction = ml_model(features.unsqueeze(0))
            probability = prediction.item()

        return {
            "blocked": probability > 0.8,
            "category": "ML Detected" if probability > 0.8 else get_domain_category(url),
            "risk_level": "High" if probability > 0.8 else "Low",
            "probability": float(probability),
            "method": "ml"
        }

    except Exception as e:
        logging.error(f"Error in ML check: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan function to manage the FastAPI app lifecycle."""
    global ml_model
    model_path = 'ml/ai/models/url_classifier_final.pth'

    # Create database tables
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created")

    # Load ML model if available
    try:
        if os.path.exists(model_path):
            ml_model = URLClassifier()
            ml_model.load_state_dict(torch.load(model_path, weights_only=True))
            ml_model.eval()
            logging.info("ML model loaded successfully")
        else:
            logging.warning("ML model not found, running without ML capabilities")
    except Exception as e:
        logging.error(f"Error loading ML model: {e}")
        ml_model = None

    yield

    # Cleanup
    logging.info("Shutting down the app")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.post("/api/log-error")
async def log_error(error: ErrorLogCreate) -> Dict[str, str]:
    """Endpoint to log errors from the extension."""
    try:
        db = SessionLocal()
        error_log = ErrorLog(
            id=os.urandom(16).hex(),
            error=error.error,
            stack=error.stack,
            timestamp=error.timestamp or datetime.utcnow(),
            url=error.url
        )
        db.add(error_log)
        db.commit()
        db.close()
        
        logging.error(f"Extension error: {error.error}")
        if error.stack:
            logging.error(f"Stack trace: {error.stack}")
            
        return {"status": "logged"}
    except Exception as e:
        logging.error(f"Error logging extension error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-url")
async def check_url(url: str = Form(...)) -> Dict[str, Any]:
    """Endpoint to check if a URL should be blocked."""
    try:
        # First try pattern matching
        pattern_result = check_url_patterns(url)
        if pattern_result is not None:
            logging.info(f"URL {url} checked via pattern matching")
            
            # Store the activity
            db = SessionLocal()
            activity = Activity(
                id=os.urandom(16).hex(),
                url=url,
                action="blocked" if pattern_result["blocked"] else "allowed",
                category=pattern_result["category"],
                risk_level=pattern_result["risk_level"]
            )
            db.add(activity)
            db.commit()
            db.close()
            
            return pattern_result

        # If pattern matching is inconclusive, try ML
        ml_result = check_url_ml(url)
        if ml_result is not None:
            logging.info(f"URL {url} checked via ML")
            
            # Store the activity
            db = SessionLocal()
            activity = Activity(
                id=os.urandom(16).hex(),
                url=url,
                action="blocked" if ml_result["blocked"] else "allowed",
                category=ml_result["category"],
                risk_level=ml_result["risk_level"]
            )
            db.add(activity)
            db.commit()
            db.close()
            
            return ml_result

        # If both methods fail, return safe default
        return {
            "blocked": False,
            "category": get_domain_category(url),
            "risk_level": "Low",
            "method": "default"
        }

    except Exception as e:
        logging.error(f"Error checking URL: {e}")
        return {
            "blocked": False,
            "category": "Error",
            "risk_level": "Unknown",
            "method": "error"
        }

@app.get("/api/dashboard/stats")
async def get_dashboard_stats() -> DashboardStats:
    """Endpoint to get dashboard statistics."""
    try:
        db = SessionLocal()
        now = datetime.utcnow()

        # Get counts excluding checking and error actions
        activities = db.query(Activity).all()
        total_sites = len({activity.url for activity in activities})  # Unique URLs
        blocked_sites = len({activity.url for activity in activities if activity.action == "blocked"})
        allowed_sites = len({activity.url for activity in activities if activity.action == "allowed"})
        visited_sites = len({activity.url for activity in activities if activity.action == "visited"})

        # Get recent activities, excluding checking actions and duplicate URLs within a minute
        recent = []
        seen_urls = set()
        for activity in db.query(Activity).filter(Activity.action != "checking").order_by(Activity.timestamp.desc()).all():
            # Create a key combining URL and minute timestamp to prevent duplicates within the same minute
            key = (activity.url, activity.timestamp.replace(second=0, microsecond=0))
            if key not in seen_urls:
                recent.append(activity)
                seen_urls.add(key)
            if len(recent) >= 10:
                break

        # Calculate daily stats
        daily_stats = {}
        for day_offset in range(7):
            day = now - timedelta(days=day_offset)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            count = db.query(Activity).filter(
                Activity.timestamp >= day_start,
                Activity.timestamp < day_end
            ).count()
            daily_stats[day_start.strftime("%Y-%m-%d")] = count

        db.close()
        return DashboardStats(
            total_sites=total_sites,
            blocked_sites=blocked_sites,
            allowed_sites=allowed_sites,
            visited_sites=visited_sites,
            recent_activities=[
                ActivityResponse(
                    url=a.url,
                    timestamp=a.timestamp,
                    action=a.action,
                    category=a.category,
                    risk_level=a.risk_level
                ) for a in recent
            ],
            daily_stats=daily_stats,
            alerts=[
                {
                    "id": os.urandom(16).hex(),
                    "message": f"High risk activity detected: {activity.url}",
                    "priority": "high",
                    "timestamp": activity.timestamp.isoformat()
                } for activity in recent if activity.risk_level and activity.risk_level.lower() == 'high'
            ] if any(a.risk_level and a.risk_level.lower() == 'high' for a in recent) else []
        )
    except Exception as e:
        logging.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/activity")
async def record_activity(activity: ActivityCreate) -> ActivityResponse:
    """Endpoint to record browsing activity."""
    try:
        db = SessionLocal()
        db_activity = Activity(
            id=os.urandom(16).hex(),
            url=activity.url,
            timestamp=activity.timestamp or datetime.utcnow(),
            action=activity.action,
            category=activity.category or get_domain_category(activity.url)
        )
        db.add(db_activity)
        db.commit()
        db.refresh(db_activity)
        db.close()

        return ActivityResponse(
            url=db_activity.url,
            timestamp=db_activity.timestamp,
            action=db_activity.action,
            category=db_activity.category,
            risk_level=db_activity.risk_level
        )
    except Exception as e:
        logging.error(f"Error recording activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_alerts() -> List[ActivityResponse]:
    """Endpoint to get alerts (blocked and high-risk activities)."""
    try:
        db = SessionLocal()
        activities = db.query(Activity).filter(
            (Activity.action == "blocked") |
            (Activity.risk_level.ilike("high"))
        ).order_by(Activity.timestamp.desc()).all()

        db.close()
        return [
            ActivityResponse(
                url=activity.url,
                timestamp=activity.timestamp,
                action=activity.action,
                category=activity.category,
                risk_level=activity.risk_level
            ) for activity in activities
        ]
    except Exception as e:
        logging.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/activities")
async def get_recent_activities() -> List[ActivityResponse]:
    """Endpoint to get recent activities."""
    try:
        db = SessionLocal()
        activities = db.query(Activity).order_by(Activity.timestamp.desc()).all()

        db.close()
        return [
            ActivityResponse(
                url=activity.url,
                timestamp=activity.timestamp,
                action=activity.action,
                category=activity.category,
                risk_level=activity.risk_level
            ) for activity in activities
        ]
    except Exception as e:
        logging.error(f"Error getting recent activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
async def retrain() -> Dict[str, str]:
    """Endpoint to retrain the ML model."""
    global ml_model
    try:
        logging.info("Starting model retraining...")
        ml_model = train()
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        logging.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
