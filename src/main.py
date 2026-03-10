# driver for entire ml pipeline 
import sys
import pandas as pd

from src.config import Config
from src.logger import get_logger
from src import data_loader, preprocess, mining, balancing, train, predict


def run_pipeline(config_path: str = "configs/config.yaml"):
    config = Config(config_path)
    logger = get_logger(__name__)

    logger.info("Starting ML pipeline")

    # Load merged dataset (demographics + visits + logs)
    df = data_loader.load_data(config)

    # Create log-derived features and merge into dataset
    log_cfg = config.get("preprocessing", {}).get("hospital_logs", {})
    df_log_features = preprocess.create_log_features(df, log_cfg)
    df_ml_ready = df.merge(df_log_features, on="patient_id", how="left")

    # Encode categorical and scale numerical features
    df_ml_ready, scaler, numerical_cols = preprocess.encode_and_scale(df_ml_ready, config)
    
    # Save scaler for deployment
    import pickle
    import os
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "numerical_cols": numerical_cols}, f)
    logger.info("✅ Scaler saved to models/scaler.pkl")

    logger.info("Mining data...")
    df_mined = mining.apply_clustering(df_ml_ready, config.get("mining", {}))

    logger.info("Balancing dataset...")
    df_balanced = balancing.balance_data(df_mined, config.get("balancing", {}))

    logger.info("Training models with Bayesian optimization...")
    # Train with Optuna (no MLflow)
    results = train.train_models_no_mlflow(df_balanced, config_path=config_path)
    logger.info("Training finished.")

    logger.info("Generating predictions...")
    preds = predict.generate_predictions(
        config.get("prediction", {}),
        model_path=config.get("prediction", {}).get("model_path", "models/best_model.pkl"),
        df=df_balanced,
    )

    output_path = config.get("prediction", {}).get("output_path")
    if output_path:
        df_balanced["prediction"] = preds
        df_balanced.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    return {
        "predictions": preds,
        "model_name": results["model_name"],
        "score": results["score"],
        "balanced_data": df_balanced,
    }


def main(config_path: str = "configs/config.yaml"):
    """
    CLI entry point: wraps run_pipeline() with sys.exit handling.
    """
    try:
        result = run_pipeline(config_path)
        logger = get_logger(__name__)
        logger.info(f"✅ Pipeline completed successfully. Predictions: {len(result['predictions'])} samples")
    except Exception as e:
        logger = get_logger(__name__)
        logger.exception("Pipeline failed due to an error")
        sys.exit(1)


if __name__ == "__main__":
    main()
