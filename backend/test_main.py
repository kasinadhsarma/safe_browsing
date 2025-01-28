import pytest
import httpx
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from main import app, SessionLocal, Activity, Setting, MLMetrics, Base, engine

@pytest.fixture
def db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
async def client():
    app.dependency_overrides = {}  # Clear any overrides
    async with httpx.AsyncClient(base_url="http://test") as client:
        app.mount = lambda path, app: None  # Disable real HTTP transport
        app.middleware = []  # Clear middleware to avoid external calls
        client.app = app
        yield client

@pytest.mark.asyncio
async def test_get_stats(db, client):
    response = await client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_sites" in data
    assert "blocked_sites" in data
    assert "recent_activities" in data
    assert "ml_metrics" in data
    assert "risk_distribution" in data

@pytest.mark.asyncio
async def test_check_url(db, client):
    # Test safe URLs
    safe_urls = [
        "https://www.youtube.com/",
        "https://mail.google.com/mail/u/0/#inbox"
    ]
    for url in safe_urls:
        response = await client.post("/api/check-url", data={"url": url, "age_group": "kid"})
        assert response.status_code == 200
        data = response.json()
        assert data["blocked"] == False
        assert data["risk_level"] == "Low"
        assert data["category"] == "Internal" or data["category"] == "Unknown"
        assert "ml_scores" in data
        ml_scores = data["ml_scores"]
        assert ml_scores["knn"] == 0.0
        assert ml_scores["svm"] == 0.0
        assert ml_scores["nb"] == 0.0

    # Test unsafe URL
    response = await client.post("/api/check-url", data={"url": "https://pornhub.com", "age_group": "kid"})
    assert response.status_code == 200
    data = response.json()
    assert "blocked" in data
    assert "risk_level" in data
    assert "category" in data
    assert "probability" in data
    assert "predictions" in data
    assert "ml_scores" in data

@pytest.mark.asyncio
async def test_log_activity(db, client):
    # Test logging safe URL
    safe_url_data = {
        "url": "https://www.youtube.com/",
        "action": "allowed",
        "category": "Unknown",
        "risk_level": "Low",
        "ml_scores": '{"knn": 0.0, "svm": 0.0, "nb": 0.0}',
        "block_reason": None,
        "age_group": "kid"
    }
    response = await client.post("/api/activity", data=safe_url_data)
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == safe_url_data["url"]
    assert data["action"] == "allowed"
    assert data["risk_level"] == "Low"
    assert data["ml_scores"] == {"knn": 0.0, "svm": 0.0, "nb": 0.0}

    # Test logging unsafe URL
    unsafe_url_data = {
        "url": "https://vette-porno.nl",
        "action": "blocked",
        "category": "Malware",
        "risk_level": "High",
        "ml_scores": '{"knn": 0.9, "svm": 0.8, "nb": 0.7}',
        "block_reason": "High risk content detected",
        "age_group": "kid"
    }
    response = await client.post("/api/activity", data=unsafe_url_data)
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == unsafe_url_data["url"]
    assert data["action"] == "blocked"
    assert data["risk_level"] == "High"
    assert "timestamp" in data
    assert data["category"] == unsafe_url_data["category"]
    assert data["ml_scores"] == {"knn": 0.9, "svm": 0.8, "nb": 0.7}
    assert data["age_group"] == "kid"
    assert data["block_reason"] == "High risk content detected"

@pytest.mark.asyncio
async def test_get_activities(db, client):
    response = await client.get("/api/activities")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_get_alerts(db, client):
    response = await client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
