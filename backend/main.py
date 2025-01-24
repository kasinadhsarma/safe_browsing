from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from ml.ai.training import URLClassifier, train
from ml.ai.dataset import URLDataset
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

# Importing the required models from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database setup
DATABASE_URL = "sqlite:///./safebrowsing.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Model
class Activity(Base):
    __tablename__ = "activities"

    id = Column(String, primary_key=True)
    url = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)  # 'blocked', 'allowed', 'visited', 'checking'
    category = Column(String, nullable=True)
    risk_level = Column(String, nullable=True)
    probability = Column(String, nullable=True)

# Pydantic models for request/response
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

model: Optional[URLClassifier] = None
nb_model: Optional[MultinomialNB] = None
svm_model: Optional[SVC] = None
knn_model: Optional[KNeighborsClassifier] = None
vectorizer: Optional[Any] = None

def get_domain_category(url: str) -> str:
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        if 'chat.openai.com' in domain: return 'AI Chat'
        if 'youtube.com' in domain: return 'Video'
        if 'google.com' in domain: return 'Search'
        if any(x in domain for x in ['facebook.com', 'twitter.com', 'instagram.com']):
            return 'Social Media'
        return 'General'
    except:
        return 'Unknown'

def classify_text(text: str) -> Dict[str, Any]:
    """
    Classify text using the loaded models.
    """
    global nb_model, svm_model, knn_model, vectorizer
    if not all([nb_model, svm_model, knn_model, vectorizer]):
        raise HTTPException(status_code=500, detail="Text classification models not loaded")

    try:
        text_vectorized = vectorizer.transform([text])
        nb_prediction = nb_model.predict(text_vectorized)[0]
        svm_prediction = svm_model.predict(text_vectorized)[0]
        knn_prediction = knn_model.predict(text_vectorized)[0]

        return {
            "nb_prediction": int(nb_prediction),
            "svm_prediction": int(svm_prediction),
            "knn_prediction": int(knn_prediction)
        }
    except Exception as e:
        logging.error(f"Error classifying text: {e}")
        raise HTTPException(status_code=500, detail=f"Error classifying text: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan function to manage the FastAPI app lifecycle.
    """
    global model, nb_model, svm_model, knn_model, vectorizer
    model_path = 'ml/ai/models/url_classifier_final.pth'
    nb_model_path = 'text_classification_model_nb.joblib'
    svm_model_path = 'text_classification_model_svm.joblib'
    knn_model_path = 'text_classification_model_knn.joblib'
    vectorizer_path = 'text_vectorizer.joblib'

    # Create database tables
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created.")

    # Ensure the models directory exists
    if not os.path.exists('ml/ai/models'):
        os.makedirs('ml/ai/models')
        logging.info("Created models directory.")

    try:
        model = URLClassifier()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        logging.info("Loaded pre-trained URL classification model.")
    except Exception as e:
        logging.warning(f"Failed to load pre-trained URL classification model: {e}. Training a new model...")
        model = train()

    try:
        nb_model = joblib.load(nb_model_path)
        svm_model = joblib.load(svm_model_path)
        knn_model = joblib.load(knn_model_path)
        vectorizer = joblib.load(vectorizer_path)
        logging.info("Loaded pre-trained text classification models.")
    except Exception as e:
        logging.error(f"Failed to load pre-trained text classification models: {e}")

    yield
    logging.info("Shutting down the app.")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://*",  # Allow Chrome extensions
        "http://localhost:3000",  # Allow local development
        "http://localhost:8000"   # Allow local API
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    end_time = datetime.utcnow()

    logging.info(
        f"[{request.method}] {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {(end_time - start_time).total_seconds():.3f}s"
    )
    return response

@app.get("/api/dashboard/stats")
async def get_dashboard_stats() -> DashboardStats:
    """
    Endpoint to get dashboard statistics.
    """
    try:
        db = SessionLocal()

        # Calculate time ranges
        now = datetime.utcnow()
        past_24h = now - timedelta(days=1)
        past_7d = now - timedelta(days=7)

        # Get counts
        total_sites = db.query(Activity).count()
        blocked_sites = db.query(Activity).filter(Activity.action == "blocked").count()
        allowed_sites = db.query(Activity).filter(Activity.action == "allowed").count()
        visited_sites = db.query(Activity).filter(Activity.action == "visited").count()

        # Get recent activities
        recent = db.query(Activity).order_by(Activity.timestamp.desc()).limit(10).all()

        # Calculate daily stats for the past 7 days
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
            daily_stats=daily_stats
        )
    except Exception as e:
        logging.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-url")
async def check_url(url: str = Form(...)) -> Dict[str, Any]:
    """
    Endpoint to check if a URL should be blocked.
    """
    global model
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Extract features from the URL
        dataset = URLDataset([url], [0])
        features = dataset[0]['features']

        # Make a prediction
        with torch.no_grad():
            prediction = model(features.unsqueeze(0))
            probability = prediction.item()

        # Determine the risk level
        risk_level = "High" if probability > 0.8 else "Medium" if probability > 0.5 else "Low"
        category = get_domain_category(url)

        # Classify the text content of the URL
        text_classification = classify_text(url)

        # Store the check activity
        db = SessionLocal()
        activity = Activity(
            id=os.urandom(16).hex(),
            url=url,
            action="blocked" if probability > 0.5 else "allowed",
            category=category,
            risk_level=risk_level,
            probability=str(probability)
        )
        db.add(activity)
        db.commit()
        db.close()

        return {
            "blocked": probability > 0.5,
            "probability": float(probability),
            "risk_level": risk_level,
            "category": category,
            "url": url,
            "text_classification": text_classification
        }
    except Exception as e:
        db = SessionLocal()
        # Log the error but still record the activity
        activity = Activity(
            id=os.urandom(16).hex(),
            url=url,
            action="error",
            category="Error",
            risk_level="Unknown"
        )
        db.add(activity)
        db.commit()
        db.close()

        logging.error(f"Error checking URL: {url}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {e}")

@app.post("/api/activity")
async def record_activity(activity: ActivityCreate) -> ActivityResponse:
    """
    Endpoint to record browsing activity.
    """
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

@app.post("/api/retrain")
async def retrain() -> Dict[str, str]:
    """
    Endpoint to retrain the model.
    """
    global model
    try:
        logging.info("Retraining the model...")
        model = train()
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Error during retraining: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
