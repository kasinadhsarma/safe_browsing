from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import torch
from ml.ai.training import URLClassifier, train
from ml.ai.dataset import URLDataset
import os
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model: Optional[URLClassifier] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = 'ml/ai/models/url_classifier_final.pth'
    
    if not os.path.exists('ml/ai/models'):
        os.makedirs('ml/ai/models')
        logging.info("Created models directory.")
    
    try:
        model = URLClassifier()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        logging.info("Loaded pre-trained model.")
    except Exception as e:
        logging.warning(f"Failed to load pre-trained model: {e}. Training a new model...")
        model = train()
    
    yield
    
    logging.info("Shutting down the app.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/check-url")
async def check_url(url: str) -> Dict[str, Any]:
    global model
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        dataset = URLDataset([url], [0])
        features = dataset[0]['features']
        
        with torch.no_grad():
            prediction = model(features.unsqueeze(0))
            probability = prediction.item()
        
        risk_level = "High" if probability > 0.8 else "Medium" if probability > 0.5 else "Low"
        
        return {
            "blocked": probability > 0.5,
            "probability": float(probability),
            "risk_level": risk_level,
            "url": url
        }
    except Exception as e:
        logging.error(f"Error checking URL: {url}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {e}")

@app.post("/api/retrain")
async def retrain() -> Dict[str, str]:
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