"""
Feature Engineering for the Inventory Optimization Platform.

This module transforms raw merged M5 data into a model-ready feature matrix.
Every feature group has explicit business motivation — we don't engineer
features blindly; each one encodes a signal the forecaster can act on.

Feature groups:
┌───────────────────────┬───────────────────────────────────────────────────┐
│ Group                 │ Business Signal                                   │
├───────────────────────┼───────────────────────────────────────────────────┤
│ Calendar              │ Day-of-week patterns, monthly seasonality         │
│ Events                │ Holiday demand spikes                             │
│ SNAP                  │ Demand lift from gov't food assistance days        │
│ Lag features          │ Autocorrelation in demand (same day last week)    │
│ Rolling stats         │ Trend direction and demand volatility             │
│ Price features        │ Price elasticity & promotional sensitivity        │
└───────────────────────┴───────────────────────────────────────────────────┘

Lag / rolling feature computation:
- All lag/rolling features are computed within (store_id, item_id) groups
  so there is NO data leakage across store-item combinations.
- We sort by date before shifting to guarantee temporal order.
- NaN values introduced by shifting are intentionally kept; the forecasting
  model (XGBoost/LightGBM) handles them natively. We do NOT impute them
  to avoid introducing artificial signal at the start of each series.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

# Snap flag column name is state-specific in M5
_SNAP_COL_MAP = {
    "CA": "snap_CA",
    "TX": "snap_TX",
    "WI": "snap_WI",
}


class FeatureEngineer:
    """
    Transforms merged M5 data into a feature-rich DataFrame for modeling.

    Args:
        config_path: Path to config.yaml.

    Example::

        fe = FeatureEngineer("config/config.yaml")
        df_features = fe.transform(df_raw)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.cfg = self._load_config()
        self.feat_cfg = self.cfg.get("features", {})
        self.lag_days: list[int] = self.feat_cfg.get("lag_days", [7, 14, 28])
        self.rolling_windows: list[int] = self.feat_cfg.get("rolling_windows", [7, 28])
        self.min_group_rows: int = self.feat_cfg.get("min_group_rows", 30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps in sequence.

        Args:
            df: Output of M5DataLoader.load() — merged long-format DataFrame.

        Returns:
            pd.DataFrame: Feature-enriched DataFrame ready for train/test split.
        """
        t0 = time.perf_counter()
        log.info("Starting feature engineering | input rows=%d", len(df))

        # Ensure consistent sort order before any lag/group operations
        df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        df = self.add_calendar_features(df)
        df = self.add_event_features(df)
        df = self.add_snap_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_price_features(df)
        df = self._drop_sparse_groups(df)

        elapsed = time.perf_counter() - t0
        log.info(
            "Feature engineering complete | output rows=%d | cols=%d | elapsed=%.1fs",
            len(df), len(df.columns), elapsed,
        )
        return df

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from the date column.

        Why these features?
        - day_of_week: retail strongly follows weekly rhythms (Mon=slow, Fri=busy)
        - month: captures seasonal buying patterns (back-to-school, holidays)
        - week_of_year: finer seasonal resolution (53 buckets vs 12)
        - is_weekend: binary signal often more predictive than full day-of-week
        - quarter: coarse seasonality for promotion planning
        """
        log.debug("Adding calendar features")
        df = df.copy()

        df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")   # 0=Mon
        df["month"] = df["date"].dt.month.astype("int8")
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int8")
        df["day_of_month"] = df["date"].dt.day.astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
        df["quarter"] = df["date"].dt.quarter.astype("int8")
        df["year"] = df["date"].dt.year.astype("int16")

        return df

    # ------------------------------------------------------------------
    # Event features
    # ------------------------------------------------------------------

    def add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode holiday/event features from M5 calendar columns.

        M5 has up to two events per day (event_name_1, event_name_2).
        We create:
        - has_event: binary flag (any event today)
        - event_type_encoded: ordinal encoding of primary event type
          (Cultural=1, National=2, Religious=3, Sporting=4, none=0)

        We intentionally avoid one-hot encoding event names because there
        are 30+ unique events — a label-encoded numeric is sufficient for
        tree-based models and keeps memory usage low.
        """
        log.debug("Adding event features")
        df = df.copy()

        # Binary: any event today?
        df["has_event"] = (
            df["event_name_1"].notna() | df["event_name_2"].notna()
        ).astype("int8")

        # Ordinal encode primary event type
        event_type_order = {None: 0, "Cultural": 1, "National": 2,
                            "Religious": 3, "Sporting": 4}
        if "event_type_1" in df.columns:
            # Convert to plain object/str first so fillna(0) works without
            # the "new category" restriction that categorical dtype enforces.
            event_type_str = df["event_type_1"].astype(object)
            df["event_type_encoded"] = (
                event_type_str
                .map(event_type_order)
                .fillna(0)
                .astype("int8")
            )

        return df

    # ------------------------------------------------------------------
    # SNAP features
    # ------------------------------------------------------------------

    def add_snap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a unified SNAP flag based on the store's state.

        M5 provides snap_CA, snap_TX, snap_WI separately. We resolve which
        flag applies to each row using the store's state_id so the model
        gets a single boolean feature (snap_active).

        SNAP days show consistent demand lifts in M5 especially for food
        staples — this is one of the highest-signal features in the dataset.
        """
        log.debug("Adding SNAP features")
        df = df.copy()

        if "state_id" not in df.columns:
            log.warning("state_id column missing — skipping SNAP feature")
            df["snap_active"] = 0
            return df

        # Resolve per-row snap flag based on state
        state_ids = df["state_id"].astype(str)
        df["snap_active"] = 0
        for state, col in _SNAP_COL_MAP.items():
            if col in df.columns:
                mask = state_ids == state
                df.loc[mask, "snap_active"] = df.loc[mask, col].values
        df["snap_active"] = df["snap_active"].astype("int8")

        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features: sales N days ago for each (store, item).

        Lags encode autocorrelation — "how much did we sell exactly one week
        ago?" is highly predictive for weekly-seasonal retail demand.

        Critical implementation note:
          We shift WITHIN each (store, item) group to avoid leakage across
          products. A naive df['sales'].shift(7) would incorrectly take the
          last observation from a different product's time series.
        """
        log.debug("Adding lag features for windows: %s", self.lag_days)
        df = df.copy()

        group_sales = df.groupby(["store_id", "item_id"], observed=True)["sales"]

        for lag in self.lag_days:
            col_name = f"lag_{lag}"
            df[col_name] = group_sales.transform(lambda s: s.shift(lag))
            df[col_name] = df[col_name].astype("float32")
            log.debug("  Created %s (NaN count: %d)", col_name, df[col_name].isna().sum())

        return df

    # ------------------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------------------

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics: mean and std over N-day windows.

        Rolling features capture trend direction and demand volatility:
        - rolling_mean_7: short-term trend (last week average)
        - rolling_mean_28: medium-term trend (monthly smoothing)
        - rolling_std_7: short-term demand variability → safety stock signal

        We use min_periods=1 to avoid NaN propagation but the first few
        rows per group will still have lag-1 NaN from the shift(1) applied
        before the rolling window. This shift-by-1 prevents look-ahead bias:
        the rolling stat on day T only uses data up to day T-1.
        """
        log.debug("Adding rolling features for windows: %s", self.rolling_windows)
        df = df.copy()

        for window in self.rolling_windows:
            # Shift-1 inside rolling to avoid using today's sales in today's feature
            shifted = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
                lambda s: s.shift(1)
            )

            df[f"rolling_mean_{window}"] = (
                shifted
                .groupby(df["store_id"].astype(str) + "_" + df["item_id"].astype(str))
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .astype("float32")
            )

            df[f"rolling_std_{window}"] = (
                shifted
                .groupby(df["store_id"].astype(str) + "_" + df["item_id"].astype(str))
                .transform(lambda s: s.rolling(window, min_periods=1).std())
                .astype("float32")
            )

        return df

    # ------------------------------------------------------------------
    # Price features
    # ------------------------------------------------------------------

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive price-derived features capturing elasticity signals.

        Features:
        - price_rel_mean: sell_price / item's historical mean price
          → detects promotional pricing (< 1.0) or premium positioning (> 1.0)
        - price_change_pct: % change from previous week's price
          → captures price shock effects on demand
        - log_price: log-transformed price for models assuming multiplicative elasticity

        Why not use raw sell_price? Because raw prices vary hugely across
        items (a bottle of water vs a TV). Relative features normalize across
        the product catalog and encode economically meaningful signals.
        """
        if "sell_price" not in df.columns:
            log.warning("sell_price column missing — skipping price features")
            return df

        log.debug("Adding price features")
        df = df.copy()

        # Price relative to item's all-time mean across all stores
        item_mean_price = df.groupby("item_id", observed=True)["sell_price"].transform("mean")
        df["price_rel_mean"] = (df["sell_price"] / item_mean_price.replace(0, np.nan)).astype("float32")

        # Week-over-week price change (shift by 7 within group)
        df["price_change_pct"] = (
            df.groupby(["store_id", "item_id"], observed=True)["sell_price"]
            .transform(lambda s: s.pct_change(7))
            .astype("float32")
        )

        # Log price (handles skewed price distributions)
        df["log_price"] = np.log1p(df["sell_price"]).astype("float32")

        return df

    # ------------------------------------------------------------------
    # Quality control
    # ------------------------------------------------------------------

    def _drop_sparse_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove (store, item) groups with fewer than min_group_rows records.

        Rationale: groups with very short histories produce unreliable lag
        features — mostly NaN — and degrade model quality. Dropping them
        has minimal business impact since these are typically items that
        were discontinued or newly listed.
        """
        counts = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform("count")
        before = len(df)
        df = df[counts >= self.min_group_rows].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            log.info(
                "Dropped %d rows from groups with < %d records",
                dropped, self.min_group_rows,
            )
        return df

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        from pathlib import Path
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)
