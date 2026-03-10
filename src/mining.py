import pandas as pd
from sklearn.cluster import KMeans
import time

from src.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def apply_clustering(df_preprocessed: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Applies KMeans clustering on the preprocessed dataframe (using pandas + sklearn).

    Args:
        df_preprocessed (pd.DataFrame): Preprocessed Pandas DataFrame
        cfg (dict): Configuration dictionary (from config.yaml)

    Returns:
        pd.DataFrame: DataFrame with clustering results
    """
    mining_cfg = cfg

    # Select features based on suffixes from config
    suffixes = mining_cfg.get("features_suffix", [])
    clustering_features = [
        col for col in df_preprocessed.columns 
        if any(col.endswith(suffix) for suffix in suffixes)
    ]

    if not clustering_features:
        logger.error("No clustering features found. Please check config.")
        raise ValueError("No clustering features selected for clustering.")

    X = df_preprocessed[clustering_features].fillna(0).values

    # Fit KMeans (params from config)
    start = time.time()
    kmeans = KMeans(
        n_clusters=mining_cfg.get("k", 3),
        random_state=mining_cfg.get("seed", 42),
        n_init="auto"
    )

    try:
        df_preprocessed[mining_cfg.get("output_segment_col", "patient_segment")] = kmeans.fit_predict(X)
        end = time.time()
        logger.info(f"KMeans clustering completed in {round(end - start, 2)} seconds")
    except Exception as e:
        logger.error(f"KMeans training failed: {str(e)}")
        raise

    logger.info(f"Clustering results added as column '{mining_cfg.get('output_segment_col', 'patient_segment')}'")

    return df_preprocessed
