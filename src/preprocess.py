import pandas as pd
import numpy as np
from src.logger import get_logger
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = get_logger(__name__)

# for patient_demographics.csv 
def preprocess_demographics(df_demo: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    demo_cfg = cfg

    logger.info("Preprocessing patient demographics data...")

    # fillna for numerical 
    if "fillna" in demo_cfg and "avg_monthly_income" in demo_cfg["fillna"]:
        if demo_cfg["fillna"]["avg_monthly_income"] == "mean":
            income_mean = df_demo["avg_monthly_income"].mean()
            df_demo["avg_monthly_income"].fillna(income_mean, inplace=True)
            logger.info(f"Filled missing avg_monthly_income with mean={income_mean:.2f}")

    # fillna for categorical
    if "categorical_features" in demo_cfg:
        for c in demo_cfg["categorical_features"]:
            fill_val = demo_cfg["fillna"].get("categorical", "Unknown")
            df_demo[c] = df_demo[c].fillna(fill_val)
            logger.info(f"Filled missing values in {c} with {fill_val}")

    # normalize gender
    if demo_cfg.get("normalize_gender", False) and "gender" in df_demo.columns:
        df_demo["gender"] = df_demo["gender"].replace(
            {"M": "Male", "Male": "Male", "F": "Female", "Female": "Female"}
        ).fillna("Other")
        logger.info("Normalized gender values")

    # cap chronic conditions
    if "cap_chronic_conditions" in demo_cfg and "chronic_conditions" in df_demo.columns:
        cap_val = demo_cfg["cap_chronic_conditions"]
        df_demo["chronic_conditions"] = np.where(
            df_demo["chronic_conditions"] > cap_val,
            cap_val,
            df_demo["chronic_conditions"]
        )
        logger.info(f"Capped chronic_conditions at {cap_val}")

    return df_demo


# for patient_visits.csv
def preprocess_visits(df_visits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    visit_cfg = cfg
    logger.info("Preprocessing patient visits data...")

    # fillna
    df_visits.fillna(visit_cfg.get("fillna", {}), inplace=True)

    # cast columns
    if "cast_columns" in visit_cfg:
        for col_name, dtype in visit_cfg["cast_columns"].items():
            df_visits[col_name] = df_visits[col_name].astype(dtype, errors="ignore")
            logger.info(f"Casted {col_name} to {dtype}")

    return df_visits


# for hospital_logs.xml
def preprocess_logs(df_logs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    log_cfg = cfg
    logger.info("Preprocessing hospital logs data...")

    # cast columns
    if "cast_columns" in log_cfg:
        for col_name, dtype in log_cfg["cast_columns"].items():
            df_logs[col_name] = df_logs[col_name].astype(dtype, errors="ignore")
            logger.info(f"Casted {col_name} to {dtype}")

    # parse timestamp
    if "timestamp_format" in log_cfg and "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], format=log_cfg["timestamp_format"], errors="coerce")
        logger.info(f"Parsed timestamp column with format {log_cfg['timestamp_format']}")

    return df_logs

# feature engineering
def create_log_features(df_logs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    log_cfg = cfg
    logger.info("Creating log-based features...")
    # Compute each feature independently to avoid named-aggregation issues
    # log_count
    if "patient_id" not in df_logs.columns:
        raise KeyError("'patient_id' column not found in logs DataFrame")
    log_count = (
        df_logs.groupby("patient_id")["patient_id"].count().rename("log_count")
    )

    # critical_logs
    if "log_type" in df_logs.columns:
        critical_logs = (
            df_logs.groupby("patient_id")["log_type"].apply(lambda x: (x == "critical").sum()).rename("critical_logs")
        )
    else:
        logger.info("Column 'log_type' not found; setting critical_logs to 0")
        critical_logs = log_count.copy()
        critical_logs[:] = 0
        critical_logs.rename("critical_logs", inplace=True)

    # unique_events
    if "event" in df_logs.columns:
        unique_events = (
            df_logs.groupby("patient_id")["event"].nunique().rename("unique_events")
        )
    else:
        logger.info("Column 'event' not found; setting unique_events to 0")
        unique_events = log_count.copy()
        unique_events[:] = 0
        unique_events.rename("unique_events", inplace=True)

    # Combine into a single DataFrame
    df_log_features = (
        pd.concat([log_count, critical_logs, unique_events], axis=1).reset_index()
    )

    # latest department by patient (handle suffixed columns from merges)
    dept_col = None
    for cand in ["department", "department_x", "department_y"]:
        if cand in df_logs.columns:
            dept_col = cand
            break

    if "timestamp" in df_logs.columns and dept_col:
        latest_logs = df_logs.sort_values("timestamp").drop_duplicates("patient_id", keep="last")
        latest_logs = latest_logs[["patient_id", dept_col]].rename(columns={dept_col: "most_recent_department"})
        df_log_features = df_log_features.merge(latest_logs, on="patient_id", how="left")
    else:
        logger.info("'timestamp' or department column not found; skipping most_recent_department feature")
        df_log_features["most_recent_department"] = None

    # fillna
    df_log_features.fillna(log_cfg.get("fillna", {}), inplace=True)
    logger.info("Log features created successfully")

    return df_log_features


# merge final dataset
def merge_datasets(df_demo, df_visits, df_log_features, cfg):
    logger.info("Merging demographics, visits, and log features into final dataset...")     

    df_patient = pd.merge(df_demo, df_visits, on="patient_id", how="inner")
    df_final = pd.merge(df_patient, df_log_features, on="patient_id", how="left")

    # Use .get() to access nested config
    fillna_dict = cfg.get("preprocessing").get("hospital_logs").get("fillna", {})
    df_final.fillna(fillna_dict, inplace=True)

    return df_final


# encoding and scaling - UPDATED TO MATCH NOTEBOOK
def encode_and_scale(df_final: pd.DataFrame, cfg: dict):
    fe_cfg = cfg.get("preprocessing").get("feature_engineering")
    logger.info("Starting feature encoding and scaling...")

    # Features from your notebook (matching the exact structure)
    categorical_cols = [
        "gender", "insurance", "marital_status", "education_level",
        "employment_status", "language_preference", "visit_type",
        "appointment_day", "department", "most_recent_department",
        "has_mobile_app"
    ]
    
    numerical_cols = [
        "age", "chronic_conditions", "avg_monthly_income", "num_visits",
        "total_spent", "time_in_waiting", "visit_duration",
        "satisfaction_score", "log_count", "critical_logs", "unique_events"
    ]
    
    X = df_final.copy()
    target_col = fe_cfg.get("target", "drop_off")

    # 1. ENCODE CATEGORICAL FEATURES (exactly as in notebook)
    available_categorical = [col for col in categorical_cols if col in X.columns]
    
    if available_categorical:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(X[available_categorical])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(available_categorical), 
            index=X.index
        )
        X = pd.concat([X.drop(columns=available_categorical), encoded_df], axis=1)
        logger.info(f"Encoded categorical columns: {available_categorical}")

    # 2. SCALE NUMERICAL FEATURES (exactly as in notebook)
    available_numerical = [col for col in numerical_cols if col in X.columns]
    
    scaler = None
    if available_numerical:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X[available_numerical])
        scaled_df = pd.DataFrame(
            scaled,
            columns=[f"{c}_scaled" for c in available_numerical],
            index=X.index
        )
        X = pd.concat([X.drop(columns=available_numerical), scaled_df], axis=1)
        logger.info(f"Scaled numerical columns: {available_numerical}")

    # 3. CREATE FINAL DATASET
    if target_col in X.columns:
        final_df = X[[target_col] + [c for c in X.columns if c != target_col]]
    else:
        final_df = X

    feature_names = [c for c in final_df.columns if c != target_col]

    logger.info(f"Feature engineering completed. Final feature count: {len(feature_names)}")
    print("Model expects these features:", feature_names)
    
    return final_df, scaler, available_numerical