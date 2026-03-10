import pandas as pd
import os
from src.logger import get_logger

# Centralized logger
logger = get_logger(__name__)

def _resolve_path(entry):
    if isinstance(entry, dict):
        return entry.get("path")
    return entry

def load_demographics(cfg):
    """
    Load patient demographics CSV into Spark DataFrame.
    """
    path = _resolve_path(cfg)
    if not path or not os.path.exists(path):
        logger.error(f"Demographics file not found: {path}")
        raise FileNotFoundError(f"Demographics file not found at {path}")

    logger.info(f"Loading patient demographics from {path}")
    df_demographics = pd.read_csv(path)
    logger.info(f"✅ Loaded {len(df_demographics)} patient demographics records")
    return df_demographics


def load_visits(cfg):
    """
    Load patient visits CSV into Pandas DataFrame.
    """
    path = _resolve_path(cfg)
    if not path or not os.path.exists(path):
        logger.error(f"Visits file not found: {path}")
        raise FileNotFoundError(f"Visits file not found at {path}")

    logger.info(f"Loading patient visits from {path}")
    df_visits = pd.read_csv(path)
    logger.info(f"✅ Loaded {len(df_visits)} patient vists records")
    return df_visits

def load_logs(cfg):
    """
    Load hospital logs XML into Pandas DataFrame with schema adjustments.
    """
    path = _resolve_path(cfg)
    if not path or not os.path.exists(path):
        logger.error(f"Hospital logs XML not found: {path}")
        raise FileNotFoundError(f"Hospital logs file not found at {path}")

    logger.info(f" Loading hospital logs from {path}")

    df_logs = pd.read_xml(path, xpath=".//log")  # reads <log> entries

    # Minimal cleanup for schema consistency
    if "patient_id" in df_logs.columns:
        df_logs["patient_id"] = pd.to_numeric(df_logs["patient_id"], errors="coerce")

    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")

    logger.info(f"✅ Hospital logs loaded with {len(df_logs)} records")
    return df_logs

def load_data(cfg: dict):
    """
    Wrapper to load and merge demographics, visits, and logs into a single DataFrame.
    """
    df_demo = load_demographics(cfg.get("data", {}).get("patient_demographics"))
    df_visits = load_visits(cfg.get("data", {}).get("patient_visits"))
    df_logs = load_logs(cfg.get("data", {}).get("hospital_logs"))

    # Merge on patient_id (assuming common key)
    df = pd.merge(df_demo, df_visits, on="patient_id", how="left")
    df = pd.merge(df, df_logs, on="patient_id", how="left")

    logger.info(f"✅ Final merged dataset shape: {df.shape}")
    return df
