# Inventory Optimization Platform

A production-grade inventory optimization system that combines demand forecasting, OR-Tools-based allocation, and supply chain simulation on the [M5 Forecasting dataset](https://www.kaggle.com/c/m5-forecasting-accuracy).

Built to demonstrate how a modern retail company could replace spreadsheet-driven replenishment decisions with a data-driven, solver-backed decision engine.

---

## Business Problem

Retail supply chains face a fundamental tension:

- **Too much inventory** → high holding costs, dead stock, cash tied up
- **Too little inventory** → stockouts, lost sales, poor service level

This platform simulates a Walmart-like supply chain across multiple stores and SKUs and uses optimization to allocate warehouse inventory in a way that:

| Goal | Metric |
|------|--------|
| Minimize stockouts | Stockout rate ↓ |
| Maximize fill rate | Fill rate % ↑ |
| Reduce holding cost | Inventory turnover ↑ |
| Improve service level | Days fully served / total days ↑ |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Inventory Optimization Platform                │
│                                                                 │
│  ┌───────────┐    ┌─────────────┐    ┌──────────────────────┐  │
│  │  M5 Data  │───▶│  Data       │───▶│  Feature Engineering │  │
│  │  (raw CSV)│    │  Loader     │    │  (lags, rolling,     │  │
│  └───────────┘    │  (loader.py)│    │   events, SNAP)      │  │
│                   └─────────────┘    └──────────┬───────────┘  │
│                                                 │              │
│                   ┌─────────────────────────────▼───────────┐  │
│                   │         Demand Forecasting               │  │
│                   │  (LightGBM / XGBoost / Moving Average)  │  │
│                   └─────────────────────────────┬───────────┘  │
│                                                 │ forecast     │
│  ┌────────────────────────────────────────────▼ │ ───────────┐ │
│  │                Optimization Engine            │           │ │
│  │           (OR-Tools LP/MIP solver)            │           │ │
│  │  minimize: stockout_penalty + holding_cost    │           │ │
│  │  subject to: warehouse capacity, service SLA  │           │ │
│  └─────────────────────────────┬─────────────────────────────┘ │
│                                │ allocation_qty(store, item)   │
│  ┌─────────────────────────────▼─────────────────────────────┐ │
│  │              Simulation Engine (30–60 day rollout)        │ │
│  │  Policy A: Naive  |  Policy B: Rule-based  |  Policy C: ◀─┤ │
│  │             (proportional)    (min-max)      Optimization │ │
│  └─────────────────────────────┬─────────────────────────────┘ │
│                                │ KPIs                          │
│  ┌─────────────────────────────▼─────────────────────────────┐ │
│  │              Streamlit Dashboard                          │ │
│  │  inventory levels · stockouts · costs · service level     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Roadmap

| Module | Description | Status |
|--------|-------------|--------|
| **Data Processing** | M5 ingestion, wide→long, feature engineering | ✅ Complete |
| **Demand Forecasting** | LightGBM/XGBoost with time-based CV | 🔜 Next |
| **Inventory Simulation** | Simulated inventory, lead times, stockout tracking | 🔜 Planned |
| **Optimization Engine** | OR-Tools LP allocation across stores | 🔜 Planned |
| **Replenishment Policies** | Naive · Rule-based · Solver-based comparison | 🔜 Planned |
| **Simulation Runner** | Rolling 30–60 day multi-policy simulation | 🔜 Planned |
| **Experimentation** | A/B policy comparison with statistical significance | 🔜 Planned |
| **KPI Dashboard** | Streamlit real-time KPI monitoring | 🔜 Planned |

---

## Dataset

This project uses the [M5 Forecasting — Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) dataset from Kaggle.

| File | Description |
|------|-------------|
| `sales_train_validation.csv` | Daily unit sales per item per store (wide format) |
| `calendar.csv` | Date metadata, events, SNAP flags |
| `sell_prices.csv` | Weekly sell prices per store-item |

### Dataset Download

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API key at ~/.kaggle/kaggle.json)
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

> **No Kaggle account?** Use the built-in synthetic sample data — see [Quick Start](#quick-start).

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/hsivasub/Inventory-Optimization-Platform.git
cd Inventory-Optimization-Platform
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Run with sample data (no Kaggle required)

```bash
# Generate synthetic M5-format sample data
python tests/fixtures/generate_sample_data.py

# Run the full data pipeline on sample data
python scripts/run_data_pipeline.py --sample
```

### 3. Run with real M5 data

```bash
# Place M5 CSVs in data/raw/ then:
python scripts/run_data_pipeline.py --config config/config.yaml

# Restrict to specific stores for faster iteration
python scripts/run_data_pipeline.py --stores CA_1 --stores CA_2
```

### 4. Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
Inventory Optimization Platform/
├── config/
│   └── config.yaml              # All tunable parameters (no magic numbers in code)
│
├── data/
│   ├── raw/                     # M5 CSVs go here (gitignored)
│   └── processed/               # Pipeline output parquet (gitignored)
│
├── src/
│   ├── data/
│   │   ├── loader.py            # M5DataLoader: wide→long, calendar/price merge
│   │   └── features.py          # FeatureEngineer: lags, rolling, SNAP, price, events
│   └── utils/
│       └── logger.py            # Centralized logging (rotating file + console)
│
├── scripts/
│   └── run_data_pipeline.py     # CLI entry point
│
├── tests/
│   ├── fixtures/
│   │   └── generate_sample_data.py   # Synthetic M5-format data generator
│   └── test_data_pipeline.py         # Unit + integration tests
│
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis (optional)
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Configuration

All pipeline parameters live in `config/config.yaml`. Key sections:

```yaml
data:
  stores_filter: ["CA_1", "CA_2"]   # null = use all stores
  items_filter: null

features:
  lag_days: [7, 14, 28]             # Autocorrelation lags
  rolling_windows: [7, 28]          # Trend/volatility windows

optimization:
  stockout_penalty: 100             # Cost weight for unmet demand
  holding_cost: 1                   # Cost per unit per day held
  solver: "GLOP"                    # OR-Tools solver backend
```

No configuration lives in the code. This enables config-driven experiments and CI overrides without modifying source.

---

## Feature Engineering

| Feature | Type | Signal |
|---------|------|--------|
| `day_of_week` | Calendar | Weekly demand rhythm |
| `is_weekend` | Calendar | Weekend demand lift |
| `month`, `quarter` | Calendar | Seasonal patterns |
| `has_event` | Event | Holiday demand spikes |
| `event_type_encoded` | Event | Event category impact |
| `snap_active` | SNAP | Gov't food assistance demand lift |
| `lag_7`, `lag_14`, `lag_28` | Lag | Autocorrelation (same weekday history) |
| `rolling_mean_7`, `rolling_mean_28` | Rolling | Short/medium-term trend |
| `rolling_std_7` | Rolling | Demand volatility → safety stock |
| `price_rel_mean` | Price | Promotional pricing signal |
| `price_change_pct` | Price | Price shock elasticity |
| `log_price` | Price | Multiplicative elasticity model |

> All lag/rolling features are computed **within each store-item group** to prevent data leakage across products.

---

## Optimization Formulation

*(Preview — full implementation in Module 4)*

**Decision variables:**
```
x[s, i] = units allocated from warehouse to store s for item i
```

**Objective:**
```
minimize:
  Σ(s,i) [ stockout_penalty × max(0, demand[s,i] - inventory[s,i] - x[s,i]) ]
  + Σ(s,i) [ holding_cost × (inventory[s,i] + x[s,i] - demand[s,i]) ]
  + Σ(s,i) [ transport_cost × x[s,i] ]
```

**Constraints:**
```
Σ(s) x[s, i]  ≤  warehouse_stock[i]    ∀ i       (warehouse capacity)
x[s, i]        ≥  0                     ∀ s, i    (non-negativity)
fill_rate[s]   ≥  service_level_min     ∀ s       (service SLA)
```

---

## Design Philosophy

1. **Config over code** — Every tunable value lives in `config.yaml`, not buried in functions.
2. **OOP modules** — `M5DataLoader`, `FeatureEngineer` are importable classes, not scripts.
3. **Leak-safe features** — Lag/rolling transforms are grouped by store-item; no future data bleeds into training.
4. **Fallback logic** — Solver failures fall back to heuristic policies (implemented in Module 9).
5. **Production logging** — Rotating file handler with timestamps; no `print()` statements.
6. **Sample data** — Zero external dependencies needed to run tests or demo the pipeline.

---

## Contributing

This is a portfolio project. Issues and PRs welcome.

```bash
# Code style
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/
```

---

## License

MIT License. See [LICENSE](LICENSE).
