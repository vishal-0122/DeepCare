# src/train.py
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import time
import yaml
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.logger import get_logger
from src.config import Config
from src.tune import tune_hyperparameters

logger = get_logger(__name__)

def load_config(config_path="configs/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _safe_auc(y_true, y_proba):
    """Compute AUC if possible; return None if cannot compute (e.g. multiclass or constant labels)."""
    try:
        if len(np.unique(y_true)) == 2:
            return float(roc_auc_score(y_true, y_proba))
    except Exception:
        pass
    return None


def _log_model(model, name, register_model=False, registry_name=None, X_sample=None):
    """
    Wrapper to log models safely:
      - If register_model=True and registry_name provided → try to register
      - Else → only log locally (no registry).
    """
    tracking_uri = mlflow.get_tracking_uri()
    # Infer signature if X_sample is provided
    signature = infer_signature(X_sample) if X_sample is not None else None
    if register_model and registry_name and not tracking_uri.startswith("file:"):
        mlflow.sklearn.log_model(
            model, name, registered_model_name=registry_name,
            input_example=X_sample, signature=signature
        )
    else:
        mlflow.sklearn.log_model(
            model, name,
            input_example=X_sample, signature=signature
        )

def train_models(
    df_ml_ready,
    config_path="configs/config.yaml",
    mlflow_uri=None,
    register_model=True,
):
    """
    Train several sklearn models and return the best model.
    """
    # Load full config once
    full_cfg = load_config(config_path)
    training_cfg = full_cfg.get("training", {})
    mlflow_cfg = full_cfg.get("mlflow", {})              # <-- read from ROOT, not training
    pred_cfg = full_cfg.get("prediction", {})

    experiment_name = training_cfg.get("experiment_name", "default_experiment")

    # Label/feature column resolution
    label_col = (
        training_cfg.get("label_col")
        or training_cfg.get("label_column")
        or training_cfg.get("label")
        or "label"
    )
    feature_col = training_cfg.get("feature_col", "features")

    logger.info(f"Starting training with experiment: {experiment_name}")
    logger.info(f"Using label_col='{label_col}', feature_col='{feature_col}'")

    if feature_col not in df_ml_ready.columns:
        raise KeyError(f"Feature column '{feature_col}' not found in DataFrame")

    if label_col not in df_ml_ready.columns:
        if "label" in df_ml_ready.columns:
            label_col = "label"
            logger.warning(
                "label_col not found in config; falling back to 'label' column"
            )
        else:
            raise KeyError(f"Label column '{label_col}' not found in DataFrame")

    X = list(df_ml_ready[feature_col])
    y = df_ml_ready[label_col].values

    # Split
    split_cfg = training_cfg.get("train_test_split", {})
    train_ratio = split_cfg.get("train_ratio", 0.8)
    test_ratio = split_cfg.get("test_ratio", 0.2)
    seed = split_cfg.get("seed", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_ratio, 
        test_size=test_ratio, 
        random_state=seed,
        stratify=y 
    )

    # >>> Move this after split so X_train exists
    X_sample = pd.DataFrame([X_train[0]])  # minimal 2D example for signature

    logger.info(
        f"Data split -> train: {train_ratio}, test: {test_ratio} (seed={seed})"
    )

    # ----------------- MLflow setup -----------------
    if mlflow_uri:
        if os.path.isabs(str(mlflow_uri)) or os.path.isdir(str(mlflow_uri)):
            mlflow.set_tracking_uri(f"file:{mlflow_uri}")
        else:
            mlflow.set_tracking_uri(str(mlflow_uri))
    elif mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])  # <-- now works
    else:
        mlflow.set_tracking_uri("file:./mlruns")
    # Make registry follow tracking (file: registry unsupported)
    mlflow.set_registry_uri(mlflow.get_tracking_uri())
    mlflow.set_experiment(experiment_name)
    # -------------------------------------------------

    best_model = None
    best_score = -np.inf
    results = []
    best = {"name": None, "score": -np.inf, "run_id": None}  # <-- track best run

    registry_name = mlflow_cfg.get("registry_model_name", "PatientDropOffModel")

    # ------- Logistic Regression -------
    lr_params = training_cfg.get("models", {}).get("logistic_regression", {})
    with mlflow.start_run(run_name="LogisticRegression") as r:
        start = time.time()
        try:
            lr = LogisticRegression(
                max_iter=lr_params.get("max_iter", 100),
                C=1.0 / (lr_params.get("reg_param", 1e-6) + 1e-9),
                penalty="elasticnet"
                if lr_params.get("elastic_net_param", 0.0) > 0
                else "l2",
                solver="saga",
                l1_ratio=(
                    lr_params.get("elastic_net_param")
                    if lr_params.get("elastic_net_param", 0.0) > 0
                    else None
                ),
            )
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            y_proba = (
                lr.predict_proba(X_test)[:, 1]
                if hasattr(lr, "predict_proba")
                else np.zeros(len(y_test))
            )
            end = time.time()

            auc = _safe_auc(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_params(lr_params or {})
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_metric("Accuracy", float(acc))
            if auc is not None:
                mlflow.log_metric("AUC", float(auc))
            mlflow.log_metric("TrainTime", round(end - start, 2))

            # Always log the model artifact
            _log_model(lr, "model", register_model=False, registry_name=registry_name, X_sample=X_sample)

            score = auc if auc is not None else acc
            results.append(
                ("LogisticRegression", auc, acc, round(end - start, 2))
            )
            if score > best_score:
                best_score, best_model = score, lr
                best["name"] = "LogisticRegression"
                best["score"] = score
                best["run_id"] = r.info.run_id         # <-- store winning run_id
        except Exception:
            logger.exception("Logistic Regression training failed.")

    # ------- Random Forest -------
    rf_params = training_cfg.get("models", {}).get("random_forest", {})
    with mlflow.start_run(run_name="RandomForest") as r:
        start = time.time()
        try:
            rf = RandomForestClassifier(
                n_estimators=rf_params.get("num_trees", 50),
                max_depth=rf_params.get("max_depth", None),
                random_state=rf_params.get("seed", 42),
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_proba = (
                rf.predict_proba(X_test)[:, 1]
                if hasattr(rf, "predict_proba")
                else np.zeros(len(y_test))
            )
            end = time.time()

            auc = _safe_auc(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_params(rf_params or {})
            mlflow.log_param("model", "RandomForest")
            mlflow.log_metric("Accuracy", float(acc))
            if auc is not None:
                mlflow.log_metric("AUC", float(auc))
            mlflow.log_metric("TrainTime", round(end - start, 2))

            # Always log the model artifact
            _log_model(rf, "model", register_model=False, registry_name=registry_name, X_sample=X_sample)

            score = auc if auc is not None else acc
            results.append(("RandomForest", auc, acc, round(end - start, 2)))
            if score > best_score:
                best_score, best_model = score, rf
                best["name"] = "RandomForest"
                best["score"] = score
                best["run_id"] = r.info.run_id
        except Exception:
            logger.exception("Random Forest training failed.")

    # ------- XGBoost -------
    xgb_params = training_cfg.get("models", {}).get("xgboost", {})
    with mlflow.start_run(run_name="XGBoost") as r:
        start = time.time()
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_params.get("n_estimators", 100),
                max_depth=xgb_params.get("max_depth", 3),
                learning_rate=xgb_params.get("learning_rate", 0.1),
                subsample=xgb_params.get("subsample", 1.0),
                colsample_bytree=xgb_params.get("colsample_bytree", 1.0),
                random_state=xgb_params.get("seed", 42),
                use_label_encoder=False,
                eval_metric="logloss",  # avoids warnings
            )
            xgb_model.fit(X_train, y_train)

            y_pred = xgb_model.predict(X_test)
            y_proba = (
                xgb_model.predict_proba(X_test)[:, 1]
                if hasattr(xgb_model, "predict_proba")
                else np.zeros(len(y_test))
            )

            end = time.time()

            auc = _safe_auc(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_params(xgb_params or {})
            mlflow.log_param("model", "XGBoost")
            mlflow.log_metric("Accuracy", float(acc))
            if auc is not None:
                mlflow.log_metric("AUC", float(auc))
            mlflow.log_metric("TrainTime", round(end - start, 2))

            # Always log the model artifact
            _log_model(xgb_model, "model", register_model=False, registry_name=registry_name, X_sample=X_sample)

            score = auc if auc is not None else acc
            results.append(("XGBoost", auc, acc, round(end - start, 2)))
            if score > best_score:
                best_score, best_model = score, xgb_model
                best["name"] = "XGBoost"
                best["score"] = score
                best["run_id"] = r.info.run_id
        except Exception:
            logger.exception("XGBoost training failed.")

    logger.info("Training finished.")

    # Debug logging to see what values we have
    logger.info(f"DEBUG: mlflow_cfg.get('log_models')={mlflow_cfg.get('log_models')}")
    logger.info(f"DEBUG: register_model={register_model}")
    logger.info(f"DEBUG: best={best}")
    logger.info(f"DEBUG: registry_name={registry_name}")

    # Register the best model by linking to the best run (so metrics appear in UI)
    # Changed condition: register if we have a best run_id, regardless of log_models setting
    if register_model and best.get("run_id"):
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        model_name = registry_name

        try:
            # Ensure the registered model exists
            try:
                client.get_registered_model(model_name)
                logger.info(f"Model '{model_name}' already exists in registry")
            except Exception:
                client.create_registered_model(model_name)
                logger.info(f"Created registered model '{model_name}'")

            source = f"runs:/{best['run_id']}/model"
            mv = client.create_model_version(
                name=model_name,
                source=source,
                run_id=best["run_id"]
            )
            logger.info(f"Created model version {mv.version} from run {best['run_id']}")
            
            # Promote to stage
            stage = pred_cfg.get("model_stage", "Production")
            if stage:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=stage,
                    archive_existing_versions=True
                )
            logger.info(f"✅ Registered {model_name} v{mv.version} from run {best['run_id']} (best={best['name']}, score={best['score']:.4f}) → stage={stage}")
        except Exception as e:
            logger.exception(f"Failed to register model: {e}")
    else:
        logger.warning(f"Model registration skipped: register_model={register_model}, best_run_id={best.get('run_id')}")

    return best_model

def train_models_no_mlflow(df_ml_ready, config_path="configs/config.yaml", **kwargs):
    """Train models with Bayesian optimization (Optuna), NO MLflow"""
    
    config = Config(config_path)
    training_cfg = config.get("training", {})
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Extract features and target
    X = np.array([np.array(xi) for xi in df_ml_ready['features']])
    y = df_ml_ready['drop_off'].values
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split
    test_size = training_cfg.get("train_test_split", {}).get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Bayesian optimization with Optuna
    n_trials = training_cfg.get("n_trials", 50)
    hp_results = tune_hyperparameters(X_train, y_train, n_trials=n_trials)
    
    # Train models with best parameters
    models = {}
    scores = {}
    
    logger.info("\n" + "="*60)
    logger.info("Training final models with tuned hyperparameters...")
    logger.info("="*60)
    
    # LR
    logger.info("\n[1/3] Training Logistic Regression...")
    lr = LogisticRegression(**hp_results['LogisticRegression']['best_params'], max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    lr_acc = accuracy_score(y_test, lr_pred)
    models['LogisticRegression'] = lr
    scores['LogisticRegression'] = lr_auc
    logger.info(f"✅ LR - AUC: {lr_auc:.4f}, Accuracy: {lr_acc:.4f}")
    
    # RF
    logger.info("[2/3] Training Random Forest...")
    rf = RandomForestClassifier(**hp_results['RandomForest']['best_params'], random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    rf_acc = accuracy_score(y_test, rf_pred)
    models['RandomForest'] = rf
    scores['RandomForest'] = rf_auc
    logger.info(f"✅ RF - AUC: {rf_auc:.4f}, Accuracy: {rf_acc:.4f}")
    
    # XGB
    logger.info("[3/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**hp_results['XGBoost']['best_params'], random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    xgb_acc = accuracy_score(y_test, xgb_pred)
    models['XGBoost'] = xgb_model
    scores['XGBoost'] = xgb_auc
    logger.info(f"✅ XGB - AUC: {xgb_auc:.4f}, Accuracy: {xgb_acc:.4f}")
    
    # Best model
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    best_score = scores[best_name]
    
    logger.info("\n" + "="*60)
    logger.info(f"🏆 WINNER: {best_name} with AUC = {best_score:.4f}")
    logger.info("="*60 + "\n")
    
    # Save best model
    model_path = "models/best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"✅ Model saved to {model_path}")
    
    # Save all models for comparison
    with open("models/all_models.pkl", 'wb') as f:
        pickle.dump(models, f)
    
    return {
        'model': best_model,
        'model_name': best_name,
        'score': best_score,
        'X_test': X_test,
        'y_test': y_test,
        'all_models': models,
        'all_scores': scores,
        'hp_results': hp_results
    }

