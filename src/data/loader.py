"""
M5 Forecasting Dataset Loader.

Responsibilities:
1. Read the three raw M5 CSVs (sales, calendar, prices).
2. Melt wide-format sales (d_1 … d_1913) into long format (date, sales).
3. Merge calendar metadata (events, SNAP flags, date fields).
4. Merge sell prices joined at (store_id, item_id, wm_yr_wk).
5. Return a clean DataFrame at store × item × date granularity.

Design decisions:
- We use pyarrow engine for fast CSV I/O on the large M5 files.
- Wide→long conversion is done with pd.melt, which is more memory-efficient
  than stack() for this dataset shape (30,490 items × 1,913 days).
- Calendar join uses the 'd' column (d_1 … d_1913) as the key so we keep
  the full date context without any string manipulation.
- Missing sell prices are forward-filled within each (store, item) group
  because M5 prices are stable week-over-week and gaps represent shelf
  absence rather than true nulls.
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


class M5DataLoader:
    """
    Loads and merges the M5 Forecasting dataset into a long-format DataFrame.

    Args:
        config_path: Path to config.yaml.

    Attributes:
        cfg (dict): Parsed data configuration section.
        raw_dir (Path): Directory containing raw M5 CSV files.

    Example::

        loader = M5DataLoader("config/config.yaml")
        df = loader.load()
        # df columns: store_id, item_id, dept_id, cat_id, state_id,
        #             d, date, sales, wm_yr_wk, event_name_1, event_type_1,
        #             event_name_2, event_type_2, snap_CA, snap_TX, snap_WI,
        #             sell_price
    """

    # M5 sales CSV columns: id, item_id, dept_id, cat_id, store_id, state_id, d_1…d_1913
    _ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # Calendar columns we keep (drop others to reduce memory)
    _CALENDAR_KEEP = [
        "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.cfg = self._load_config()
        self.raw_dir = Path(self.cfg["data"]["raw_dir"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Full pipeline: load → melt → merge calendar → merge prices.

        Returns:
            pd.DataFrame: Long-format dataframe ready for feature engineering.

        Raises:
            FileNotFoundError: If any of the three M5 CSVs are missing.
        """
        t0 = time.perf_counter()
        log.info("Starting M5 data load pipeline")

        sales_raw = self._load_sales()
        calendar = self._load_calendar()
        prices = self._load_prices()

        log.info("Melting wide sales format → long format")
        df = self._melt_sales(sales_raw)

        log.info("Merging calendar metadata")
        df = self._merge_calendar(df, calendar)

        log.info("Merging sell prices")
        df = self._merge_prices(df, prices)

        df = self._apply_filters(df)
        df = self._cast_dtypes(df)

        elapsed = time.perf_counter() - t0
        log.info(
            "Data load complete | rows=%d | cols=%d | elapsed=%.1fs",
            len(df), len(df.columns), elapsed,
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_sales(self) -> pd.DataFrame:
        """Load sales_train_validation.csv with fast pyarrow engine."""
        path = self.raw_dir / self.cfg["data"]["sales_file"]
        self._assert_exists(path)
        log.info("Loading sales file: %s", path)
        df = pd.read_csv(path, engine="pyarrow")
        log.debug("Sales raw shape: %s", df.shape)
        return df

    def _load_calendar(self) -> pd.DataFrame:
        """Load calendar.csv and keep only the columns we need."""
        path = self.raw_dir / self.cfg["data"]["calendar_file"]
        self._assert_exists(path)
        log.info("Loading calendar file: %s", path)
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[[c for c in self._CALENDAR_KEEP if c in df.columns]]
        # SNAP flags should be integer (0/1)
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            if col in df.columns:
                df[col] = df[col].astype("int8")
        log.debug("Calendar shape: %s", df.shape)
        return df

    def _load_prices(self) -> pd.DataFrame:
        """Load sell_prices.csv."""
        path = self.raw_dir / self.cfg["data"]["prices_file"]
        self._assert_exists(path)
        log.info("Loading prices file: %s", path)
        df = pd.read_csv(path, engine="pyarrow")
        log.debug("Prices shape: %s", df.shape)
        return df

    def _melt_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide-format M5 sales to long format.

        Wide: one row per item, columns d_1…d_1913 (each = daily units sold).
        Long: one row per (item, day), column 'sales' = units sold.

        Memory note: pd.melt creates a copy. For the full M5 dataset
        (~30k items × 1913 days = ~57M rows), peak RAM can hit ~4 GB.
        Subsetting stores/items before melting (via _apply_filters) is
        strongly recommended for local dev.
        """
        # Identify day columns
        day_cols = [c for c in df.columns if c.startswith("d_")]
        id_cols = [c for c in self._ID_COLUMNS if c in df.columns]

        long_df = pd.melt(
            df,
            id_vars=id_cols,
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )
        long_df["sales"] = long_df["sales"].astype("float32")
        log.debug("Melted shape: %s", long_df.shape)
        return long_df

    def _merge_calendar(self, df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
        """Join on 'd' (e.g. 'd_1', 'd_2') to bring in dates and event data."""
        df = df.merge(calendar, on="d", how="left")
        return df

    def _merge_prices(self, df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Join sell_prices on (store_id, item_id, wm_yr_wk).

        M5 prices are recorded weekly (wm_yr_wk), not daily, so this join
        looks up the price for the week that each date falls in.
        Forward-fill is applied within each (store, item) group to bridge
        any weeks where prices are missing (e.g. seasonal items).
        """
        df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        ffill_limit = self.cfg.get("features", {}).get("price_ffill_limit", 7)
        df = df.sort_values(["store_id", "item_id", "date"])
        df["sell_price"] = (
            df.groupby(["store_id", "item_id"])["sell_price"]
            .transform(lambda s: s.ffill(limit=ffill_limit))
        )
        df["sell_price"] = df["sell_price"].astype("float32")
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subset to configured stores/items for dev runs."""
        stores_filter: Optional[list] = self.cfg["data"].get("stores_filter")
        items_filter: Optional[list] = self.cfg["data"].get("items_filter")

        if stores_filter:
            log.info("Filtering to stores: %s", stores_filter)
            df = df[df["store_id"].isin(stores_filter)]

        if items_filter:
            log.info("Filtering to items: %s", items_filter)
            df = df[df["item_id"].isin(items_filter)]

        return df.reset_index(drop=True)

    def _cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast types to reduce memory footprint.

        Using categorical for high-cardinality string columns cuts memory
        by ~8× vs object dtype on typical M5 subsets.
        """
        cat_cols = ["store_id", "item_id", "dept_id", "cat_id", "state_id",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    @staticmethod
    def _assert_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                "Download the M5 dataset from Kaggle and place CSVs in "
                "data/raw/. See README.md for instructions."
            )
