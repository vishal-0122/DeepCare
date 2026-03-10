from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache
import os
import pickle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.5"))

@lru_cache(maxsize=1)
def get_model():
    """Load pickled model"""
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@lru_cache(maxsize=1)
def get_scaler():
    """Load pickled scaler"""
    try:
        logger.info(f"Loading scaler from {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            scaler_data = pickle.load(f)
        logger.info("Scaler loaded successfully")
        return scaler_data["scaler"], scaler_data["numerical_cols"]
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise

class PatientFeatures(BaseModel):
    age: float
    chronic_conditions: int
    avg_monthly_income: float
    num_visits: int
    total_spent: float
    time_in_waiting: float
    visit_duration: float
    satisfaction_score: float
    log_count: int
    critical_logs: int
    unique_events: int
    patient_segment: int

@app.get("/health")
def health():
    return {"model_path": MODEL_PATH, "status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict(data: PatientFeatures):
    try:
        model = get_model()
        scaler, numerical_cols = get_scaler()
        
        # Extract raw values in the order of numerical_cols
        raw_values = [
            data.age,
            data.chronic_conditions,
            data.avg_monthly_income,
            data.num_visits,
            data.total_spent,
            data.time_in_waiting,
            data.visit_duration,
            data.satisfaction_score,
            data.log_count,
            data.critical_logs,
            data.unique_events
        ]
        
        # Scale the values
        scaled_values = scaler.transform([raw_values])[0]
        
        # Append patient_segment (not scaled)
        features = np.append(scaled_values, data.patient_segment).reshape(1, -1)
        
        pred = model.predict(features)
        proba = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else float(pred[0])
        
        return {
            "prediction": int(pred[0]),
            "probability": float(proba),
            "threshold": RISK_THRESHOLD,
            "drop_off_risk": "high" if proba > RISK_THRESHOLD else "low"
        }
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))