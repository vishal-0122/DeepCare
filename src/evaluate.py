import os
import yaml
import pandas as pd
from tabulate import tabulate
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_models(results, cfg):
    """
    Evaluate models based on config-driven metrics.

    Args:
        results (list): List of dicts (preferred) or tuples produced by training.
          Preferred dict format example:
            {"model": "Logistic Regression", "partitions": 4, "auc": 0.8123,
             "accuracy": 0.7654, "f1": 0.7012, "duration": 12.5}
        cfg (dict or str): Configuration dict (or path to config YAML).
    Returns:
        pandas.DataFrame: evaluation table
    """
    # allow passing path or dict
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    # prefer binary metrics for binary problems; user config defines these lists
    metrics = eval_cfg.get("metrics", {}).get("binary", [])
    output_format = eval_cfg.get("output_format", "table")
    save_results = eval_cfg.get("save_results", False)
    results_path = eval_cfg.get("results_path", "results/evaluation_metrics.csv")

    if not results:
        logger.warning("No results supplied to evaluate_models().")
        return None

    rows = []
    for res in results:
        # ----- dict style (preferred) -----
        if isinstance(res, dict):
            model = res.get("model")
            partitions = res.get("partitions", None)
            duration = res.get("duration", res.get("time", None))

            row = {"Model": model, "Partitions": partitions, "Time (s)": duration}

            # populate metrics in order of config; allow common alias names
            for m in metrics:
                key = m.lower()
                # handle known misspellings/aliases gracefully
                alias_map = {
                    "areadunderroc": "auc",
                    "areaunderroc": "auc",
                    "areaunderroc": "auc",
                    "auc": "auc",
                    "accuracy": "accuracy",
                    "f1": "f1",
                    "precision": "precision",
                    "recall": "recall",
                }
                mapped = alias_map.get(key, key)
                row[m] = res.get(mapped, None)
            rows.append(row)

        # ----- tuple style (legacy from some train scripts) -----
        else:
            # try several unpack shapes safely
            model = None; partitions = None; auc = None; acc = None; duration = None
            try:
                # common shape: (model, partitions, auc, acc, duration)
                model, partitions, auc, acc, duration = res
            except Exception:
                try:
                    # fallback: (model, auc, acc, duration)
                    model, auc, acc, duration = res
                except Exception:
                    # very fallback: put everything as string
                    model = str(res)

            row = {"Model": model, "Partitions": partitions, "Time (s)": duration}
            for m in metrics:
                mk = m.lower()
                if mk in ("areaunderroc", "areadunderroc", "auc"):
                    row[m] = auc
                elif mk == "accuracy":
                    row[m] = acc
                else:
                    row[m] = None
            rows.append(row)

    # build DataFrame
    df = pd.DataFrame(rows)

    # ensure columns order: Model, Partitions, metrics..., Time (s)
    cols = ["Model", "Partitions"] + metrics + ["Time (s)"]
    # keep only those that exist in df
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # pretty-print table to logs
    if output_format == "table":
        table_str = tabulate(df, headers=cols, showindex=False, floatfmt=".4f", tablefmt="github")
        logger.info("\nModel Evaluation:\n%s", table_str)
    else:
        logger.info("Unsupported output_format in config: %s", output_format)

    # save results if required
    if save_results:
        out_dir = os.path.dirname(results_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info("Saved evaluation results to %s", results_path)

    return df


if __name__ == "__main__":
    # quick standalone test
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dummy_results = [
        {"model": "Logistic Regression", "partitions": 4, "auc": 0.8123, "accuracy": 0.7654, "f1": 0.7012, "duration": 12.5},
        {"model": "Random Forest", "partitions": 8, "auc": 0.8456, "accuracy": 0.7891, "f1": 0.7213, "duration": 20.3},
    ]
    evaluate_models(dummy_results, cfg)
