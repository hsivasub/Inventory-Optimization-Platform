"""
Generate synthetic sample M5-format data for testing and CI/CD.

This script creates three small CSVs that mirror the M5 schema exactly:
  - sample_sales.csv    (2 stores × 10 items × 60 days)
  - sample_calendar.csv (60 days of calendar data)
  - sample_prices.csv   (weekly prices for each store-item combination)

All values are synthetic but statistically plausible:
  - Demand follows a negative binomial distribution (count data, overdispersed)
  - Prices have occasional promotional discounts (10–20% off)
  - Events and SNAP flags match real M5 patterns

Usage:
    python tests/fixtures/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
RNG = np.random.default_rng(SEED)

# ── Dataset dimensions ─────────────────────────────────────────────────────
STORES = ["CA_1", "TX_1"]
DEPARTMENTS = ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOBBIES_1", "HOBBIES_2"]
N_ITEMS_PER_DEPT = 2  # 2 items per dept × 5 depts = 10 items total
N_DAYS = 60           # 60-day window

OUTPUT_DIR = Path(__file__).parent


def make_item_ids() -> list[str]:
    items = []
    for dept in DEPARTMENTS:
        cat = dept.split("_")[0]
        for i in range(1, N_ITEMS_PER_DEPT + 1):
            items.append(f"{dept}_{i:03d}")
    return items


def make_calendar(n_days: int = N_DAYS) -> pd.DataFrame:
    """Create a calendar table with the M5 schema."""
    start_date = pd.Timestamp("2016-01-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # M5 uses a week ID (wm_yr_wk) = YYYYWW
    wm_yr_wk = [int(f"{d.year}{d.isocalendar()[1]:02d}") for d in dates]

    # Sporadic events (roughly 1 per fortnight)
    event_types = [None] * n_days
    event_names = [None] * n_days
    event_indices = RNG.choice(n_days, size=4, replace=False)
    event_catalog = [
        ("SuperBowl", "Sporting"),
        ("ValentinesDay", "Cultural"),
        ("Easter", "Religious"),
        ("IndependenceDay", "National"),
    ]
    for i, (name, etype) in zip(event_indices, event_catalog):
        event_names[i] = name
        event_types[i] = etype

    # SNAP flags: each state has roughly 10 days/month where SNAP is active
    def snap_flags(n: int) -> list[int]:
        flags = [0] * n
        active_days = RNG.choice(n, size=int(n * 0.33), replace=False)
        for d in active_days:
            flags[d] = 1
        return flags

    calendar = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": wm_yr_wk,
        "weekday": [d.strftime("%A") for d in dates],
        "wday": [(d.dayofweek + 2) % 7 + 1 for d in dates],   # M5: 1=Sat
        "month": [d.month for d in dates],
        "year": [d.year for d in dates],
        "d": [f"d_{i+1}" for i in range(n_days)],
        "event_name_1": event_names,
        "event_type_1": event_types,
        "event_name_2": [None] * n_days,
        "event_type_2": [None] * n_days,
        "snap_CA": snap_flags(n_days),
        "snap_TX": snap_flags(n_days),
        "snap_WI": snap_flags(n_days),
    })
    return calendar


def make_sales(items: list[str], n_days: int = N_DAYS) -> pd.DataFrame:
    """
    Create wide-format sales CSV (M5 schema).

    Demand model: negative binomial with demand multiplied by a
    day-of-week factor (weekends sell ~40% more than weekdays).
    """
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]

    records = []
    for store in STORES:
        state = store.split("_")[0]
        for item in items:
            dept = "_".join(item.split("_")[:2])
            cat = item.split("_")[0]

            # Base demand varies by category
            base_demand = {"FOODS": 8, "HOUSEHOLD": 3, "HOBBIES": 2}.get(cat, 5)
            # Day-of-week multiplier (index 0=Mon in our synthetic data)
            dow_mult = np.tile(
                [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.3], n_days // 7 + 1
            )[:n_days]

            sales = RNG.negative_binomial(
                base_demand,
                0.5,
                size=n_days,
            ) * dow_mult
            sales = np.clip(np.round(sales), 0, None).astype(int)

            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item,
                "dept_id": dept,
                "cat_id": cat,
                "store_id": store,
                "state_id": state,
            }
            for d_col, s in zip(day_cols, sales):
                row[d_col] = int(s)
            records.append(row)

    return pd.DataFrame(records)


def make_prices(items: list[str], calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Create sell_prices.csv — weekly prices per (store, item, wm_yr_wk).

    Price model:
    - Each item has a base price drawn from a log-normal distribution.
    - Occasional promotional weeks drop price by 10–20%.
    """
    weeks = calendar["wm_yr_wk"].unique()
    records = []

    for store in STORES:
        for item in items:
            cat = item.split("_")[0]
            base_price = float(RNG.lognormal(
                mean={"FOODS": 2.5, "HOUSEHOLD": 3.5, "HOBBIES": 4.0}.get(cat, 3.0),
                sigma=0.4,
            ))

            for wk in weeks:
                # Occasional promo: ~15% of weeks
                is_promo = RNG.random() < 0.15
                price = base_price * (RNG.uniform(0.8, 0.9) if is_promo else 1.0)
                records.append({
                    "store_id": store,
                    "item_id": item,
                    "wm_yr_wk": int(wk),
                    "sell_price": round(price, 2),
                })

    return pd.DataFrame(records)


def main():
    items = make_item_ids()
    print(f"Generating sample data: {len(STORES)} stores × {len(items)} items × {N_DAYS} days")

    calendar = make_calendar(N_DAYS)
    sales = make_sales(items, N_DAYS)
    prices = make_prices(items, calendar)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calendar_path = OUTPUT_DIR / "sample_calendar.csv"
    sales_path = OUTPUT_DIR / "sample_sales.csv"
    prices_path = OUTPUT_DIR / "sample_prices.csv"

    calendar.to_csv(calendar_path, index=False)
    sales.to_csv(sales_path, index=False)
    prices.to_csv(prices_path, index=False)

    print(f"  ✓ {calendar_path}  ({len(calendar)} rows)")
    print(f"  ✓ {sales_path}     ({len(sales)} rows)")
    print(f"  ✓ {prices_path}    ({len(prices)} rows)")
    print("Done. Run: python scripts/run_data_pipeline.py --sample")


if __name__ == "__main__":
    main()
