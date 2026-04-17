"""
Demand Forecasting Module.

Provides DemandForecaster for taking engineered features and training
predictive models (LightGBM/XGBoost) using time-series cross validation.
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class DemandForecaster:
    """
    Trains and evaluates forecast models for inventory demand.
    
    Supports Time-based Cross Validation to prevent data leakage.
    Args:
        config (Dict): The application configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fcst_cfg = config.get("forecasting", {})
        self.model_type = self.fcst_cfg.get("model", "lightgbm").lower()
        self.model = None

        # Columns that are keys/targets and should not be used as features
        self.drop_cols = ["date", "store_id", "item_id", "sales"]

    def _get_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        target = df["sales"]
        features = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        
        # Ensure categorical types for LightGBM/XGBoost optimal runtime
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = features[col].astype('category')
                
        return features, target

    def train_cv(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Perform time-series expanding window Cross Validation.
        
        Args:
            df: The full dataset containing past dates.
            
        Returns:
            Dict containing average CV metrics (RMSE, WMAPE).
        """
        log.info(f"Starting time-series CV using {self.model_type} model")
        
        # Important: Sort by date to maintain temporal structure
        df = df.sort_values("date").reset_index(drop=True)
        dates = np.sort(df["date"].unique())
        
        n_splits = self.fcst_cfg.get("cv_splits", 3)
        forecast_horizon = self.fcst_cfg.get("forecast_horizon", 14)
        fold_size = forecast_horizon
        
        metrics = []
        
        # Create temporal splits predicting the last M days recursively.
        # Folds are generated from end of dataset moving backwards.
        for i in range(n_splits, 0, -1):
            test_start_idx = len(dates) - (i * fold_size)
            test_end_idx = test_start_idx + fold_size
            
            if test_start_idx <= 0:
                log.warning("Not enough dates for CV split. Skipping fold.")
                continue
                
            train_dates = dates[:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]
            
            log.info(f"Fold {n_splits - i + 1}/{n_splits} | Train dates: {pd.to_datetime(train_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(train_dates[-1]).strftime('%Y-%m-%d')} | Test dates: {pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(test_dates[-1]).strftime('%Y-%m-%d')}")
            
            X_train, y_train = self._get_features_target(train_df)
            X_test, y_test = self._get_features_target(test_df)
            
            t0 = time.perf_counter()
            model = self._train_single_model(X_train, y_train)
            log.info(f"Fold {n_splits - i + 1} trained in {time.perf_counter() - t0:.1f}s")
            
            preds = model.predict(X_test)
            fold_metrics = self._evaluate(y_test, preds)
            metrics.append(fold_metrics)
            
            log.info(f"Fold metrics: RMSE={fold_metrics['rmse']:.3f}, WMAPE={fold_metrics['wmape']:.3f}")
            
        if not metrics:
            return {"rmse": np.nan, "wmape": np.nan}
            
        avg_metrics = pd.DataFrame(metrics).mean().to_dict()
        log.info(f"Average CV metrics: RMSE={avg_metrics['rmse']:.3f}, WMAPE={avg_metrics['wmape']:.3f}")
        return avg_metrics

    def fit(self, df: pd.DataFrame):
        """Fit model on all provided data for production use."""
        log.info(f"Fitting final {self.model_type} model on full dataset ({len(df)} rows)")
        t0 = time.perf_counter()
        X_train, y_train = self._get_features_target(df)
        self.model = self._train_single_model(X_train, y_train)
        log.info(f"Final model fitted in {time.perf_counter() - t0:.1f}s")
        return self

    def _train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the selected underlying algorithm."""
        if self.model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=150, 
                learning_rate=0.05, 
                random_state=42, 
                n_jobs=-1
            )
            cat_cols = X_train.select_dtypes(include=['category']).columns.tolist()
            # Suppress excessive info logs from LightGBM
            model.fit(X_train, y_train, categorical_feature=cat_cols)
        elif self.model_type == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=150, 
                learning_rate=0.05, 
                random_state=42, 
                enable_categorical=True, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        X, _ = self._get_features_target(df)
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)  # Demand cannot be negative

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute primary business metrics (RMSE, WMAPE)."""
        y_pred = np.clip(y_pred, 0, None)
        rmse = compute_rmse(y_true, y_pred)
        
        sum_abs_err = np.sum(np.abs(y_true - y_pred))
        sum_actual = np.sum(y_true)
        wmape = float(sum_abs_err / sum_actual) if sum_actual > 0 else 0.0
        
        return {"rmse": rmse, "wmape": wmape}

    def save(self, output_dir: str):
        """Serialize trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
            
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        # Use config model type as part of filename
        filepath = path / f"{self.model_type}_forecaster.pkl"
        
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        log.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load trained model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        log.info(f"Model loaded from {filepath}")
