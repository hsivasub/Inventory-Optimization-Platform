"""
CLI entry point for the M5 data processing pipeline.

Usage:
    python scripts/run_data_pipeline.py
    python scripts/run_data_pipeline.py --config config/config.yaml
    python scripts/run_data_pipeline.py --config config/config.yaml --stores CA_1 CA_2
    python scripts/run_data_pipeline.py --sample   # Use built-in sample data for testing

This script is intentionally thin — it wires together the loader and
feature engineer but contains no business logic itself. All parameters
live in config.yaml or can be overridden via CLI flags.
"""

import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
import yaml

from src.data.features import FeatureEngineer
from src.data.loader import M5DataLoader
from src.utils.logger import get_logger


@click.command()
@click.option(
    "--config",
    default="config/config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--stores",
    multiple=True,
    default=None,
    help="Store IDs to process (overrides config). E.g. --stores CA_1 --stores CA_2",
)
@click.option(
    "--items",
    multiple=True,
    default=None,
    help="Item IDs to process (overrides config).",
)
@click.option(
    "--sample",
    is_flag=True,
    default=False,
    help=(
        "Use the built-in synthetic sample dataset for quick testing. "
        "No M5 download required."
    ),
)
@click.option(
    "--output",
    default=None,
    help="Override output parquet path from config.",
)
def main(config: str, stores: tuple, items: tuple, sample: bool, output: str):
    """
    Run the M5 data ingestion and feature engineering pipeline.

    Outputs a parquet file ready for the forecasting and optimization modules.
    """
    log = get_logger("run_data_pipeline", config_path=config)
    log.info("=" * 60)
    log.info("Inventory Optimization Platform — Data Pipeline")
    log.info("=" * 60)
    log.info("Config: %s", config)

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Handle --sample flag: point config to the fixtures directory
    # ------------------------------------------------------------------
    if sample:
        log.info("Running in SAMPLE mode — using synthetic fixture data")
        config = _patch_config_for_sample(config)

    # ------------------------------------------------------------------
    # Apply CLI overrides onto config
    # ------------------------------------------------------------------
    if stores or items:
        _override_filters(config, list(stores) or None, list(items) or None)

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    try:
        loader = M5DataLoader(config_path=config)
        df_raw = loader.load()
    except FileNotFoundError as exc:
        log.error(
            "Raw data files not found.\n"
            "  → Download from: https://www.kaggle.com/c/m5-forecasting-accuracy/data\n"
            "  → Place in: data/raw/\n"
            "  → Or run with --sample flag for synthetic data.\n"
            "Error: %s",
            exc,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    fe = FeatureEngineer(config_path=config)
    df_features = fe.transform(df_raw)

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    cfg = _load_yaml(config)
    out_path = Path(output) if output else Path(cfg["data"]["processed_dir"]) / cfg["data"]["output_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Saving processed features → %s", out_path)
    df_features.to_parquet(out_path, index=False, engine="pyarrow")

    elapsed = time.perf_counter() - t_start
    log.info("-" * 60)
    log.info("Pipeline complete")
    log.info("  Output rows : %d", len(df_features))
    log.info("  Output cols : %d", len(df_features.columns))
    log.info("  Output path : %s", out_path)
    log.info("  Total time  : %.1fs", elapsed)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _patch_config_for_sample(config_path: str) -> str:
    """
    Write a temporary config that points to the sample fixtures.

    Returns path to the patched config (written to a temp location).
    """
    import tempfile

    cfg = _load_yaml(config_path)
    fixture_dir = str(Path(__file__).resolve().parent.parent / "tests" / "fixtures")
    cfg["data"]["raw_dir"] = fixture_dir
    cfg["data"]["sales_file"] = "sample_sales.csv"
    cfg["data"]["calendar_file"] = "sample_calendar.csv"
    cfg["data"]["prices_file"] = "sample_prices.csv"

    # Write to a temp file so we don't mutate the real config
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="sample_config_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _override_filters(config_path: str, stores: list, items: list) -> None:
    """Mutate the config file in-place to apply CLI store/item filters."""
    cfg = _load_yaml(config_path)
    if stores:
        cfg["data"]["stores_filter"] = stores
    if items:
        cfg["data"]["items_filter"] = items
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    main()
