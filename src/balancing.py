import pandas as pd
import numpy as np
from sklearn.utils import resample
from src.logger import get_logger

logger = get_logger(__name__)


def balance_data(df_clustered: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Balance majority/minority classes in Pandas DataFrame 
    using undersampling or oversampling.
    """

    try:
        method = cfg["method"]
        majority_class = cfg["majority_class"]
        minority_class = cfg["minority_class"]

        seed = cfg.get("seed", 42)
        shuffle = cfg.get("shuffle", True)

        logger.info(f"Balancing started | Method: {method}, Seed: {seed}, Shuffle: {shuffle}")

        # Split majority/minority
        df_major = df_clustered[df_clustered["drop_off"] == majority_class]
        df_minor = df_clustered[df_clustered["drop_off"] == minority_class]

        logger.info(f"Majority class count: {len(df_major)} | Minority class count: {len(df_minor)}")

        # Apply balancing method
        if method == "undersample_majority":
            target_size = len(df_minor)
            df_major_sampled = resample(
                df_major,
                replace=False,
                n_samples=target_size,
                random_state=seed
            )
            df_balanced = pd.concat([df_major_sampled, df_minor], axis=0)

            logger.info("Applied undersampling on majority class.")

        elif method == "oversample_minority":
            target_size = len(df_major)
            df_minor_sampled = resample(
                df_minor,
                replace=True,
                n_samples=target_size,
                random_state=seed
            )
            df_balanced = pd.concat([df_major, df_minor_sampled], axis=0)

            logger.info("Applied oversampling on minority class.")

        else:
            logger.error(f"Unsupported balancing method: {method}")
            raise ValueError(f"Unsupported balancing method: {method}")

        # Shuffle if enabled
        if shuffle:
            df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
            logger.info("Shuffling applied on balanced dataset.")

        # Collect ML features (suffix-driven + cluster feature)
        suffixes = cfg.get("features_suffix", [])
        ml_features = [c for c in df_balanced.columns if any(c.endswith(sfx) for sfx in suffixes)]

        if cfg["include_cluster_feature"]:
            ml_features.append("patient_segment")

        logger.info(f"Selected ML features: {ml_features}")

        # Assemble features into numpy array
        output_col = cfg["output_feature_col"]
        df_balanced[output_col] = df_balanced[ml_features].values.tolist()

        logger.info("Balancing completed successfully.")

        return df_balanced

    except Exception as e:
        logger.error(f"Error in balance_data: {str(e)}")
        raise
