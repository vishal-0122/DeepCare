import mlflow
import pandas as pd
import os
import pickle
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

def generate_predictions(pred_cfg, model_path=None, df=None):
    """
    Predict using a pickled model. Reads:
      - model_path from prediction.model_path (default: models/best_model.pkl)
      - feature_col from prediction.feature_col (default: features)
    """
    if df is None:
        raise ValueError("df must be provided")

    feature_col = pred_cfg.get("feature_col", "features")
    model_path = model_path or pred_cfg.get("model_path", "models/best_model.pkl")

    if feature_col not in df.columns:
        raise KeyError(f"Feature column '{feature_col}' not found in DataFrame")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info("Running predictions...")
    X = np.array([np.array(xi) for xi in df[feature_col]])
    preds = model.predict(X)

    logger.info(f"Predictions generated for {len(preds)} samples")
    return preds
