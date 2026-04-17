"""
Training Script for Phase 2: Demand Forecasting.

Loads engineered features, runs time-based CV to evaluate model performance,
trains a final production model on all available data, and saves it.
"""

import logging
from pathlib import Path

import click
import pandas as pd
import yaml

from src.models.forecaster import DemandForecaster
from src.utils.logger import get_logger

# Set up logging
log = get_logger("train_model")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", default="config/config.yaml", help="Path to config.yaml")
@click.option("--skip-cv", is_flag=True, help="Skip cross-validation, just fit final model")
@click.option("--sample", is_flag=True, help="Run on sampled fast dataset (if available)")
def main(config: str, skip_cv: bool, sample: bool):
    cfg = load_config(config)
    
    # 1. Paths
    processed_dir = Path(cfg["data"].get("processed_dir", "data/processed"))
    model_dir = Path(cfg["data"].get("model_dir", "data/models"))
    
    file_name = "sample_features.parquet" if sample else cfg["data"].get("output_file", "features.parquet")
    features_path = processed_dir / file_name

    if not features_path.exists():
        log.error(f"Features file not found at {features_path}. Run data_pipeline.py first.")
        return

    # 2. Load data
    log.info(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Convert object columns to category if needed for Tree Models
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    # 3. Initialize Forecaster
    forecaster = DemandForecaster(cfg)

    # 4. Cross Validation (Optional)
    if not skip_cv:
        log.info("--- Starting Time-Series Cross-Validation ---")
        cv_metrics = forecaster.train_cv(df)
        log.info(f"--- Full CV Results: {cv_metrics} ---")

    # 5. Fit final production model
    log.info("--- Training Final Production Model ---")
    forecaster.fit(df)

    # 6. Save model
    forecaster.save(str(model_dir))
    log.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
