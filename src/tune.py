import optuna
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

def objective_lr(trial, X_train, y_train):
    C = trial.suggest_float('C', 0.001, 100, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    
    lr = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    score = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

def objective_rf(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                               min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

def objective_xgb(trial, X_train, y_train):
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    xgb_model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                 n_estimators=n_estimators, colsample_bytree=colsample_bytree,
                                 subsample=subsample, random_state=42, n_jobs=-1)
    score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

def tune_hyperparameters(X_train, y_train, n_trials=50):
    logger.info(f"Starting Bayesian hyperparameter tuning ({n_trials} trials per model)...")
    results = {}
    
    # LR
    logger.info("Tuning Logistic Regression...")
    study_lr = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study_lr.optimize(lambda trial: objective_lr(trial, X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    results['LogisticRegression'] = {'best_params': study_lr.best_params, 'best_score': study_lr.best_value}
    logger.info(f"  ✅ LR Best Score: {study_lr.best_value:.4f}")
    
    # RF
    logger.info("Tuning Random Forest...")
    study_rf = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    results['RandomForest'] = {'best_params': study_rf.best_params, 'best_score': study_rf.best_value}
    logger.info(f"  ✅ RF Best Score: {study_rf.best_value:.4f}")
    
    # XGB
    logger.info("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    results['XGBoost'] = {'best_params': study_xgb.best_params, 'best_score': study_xgb.best_value}
    logger.info(f"  ✅ XGB Best Score: {study_xgb.best_value:.4f}")
    
    return results