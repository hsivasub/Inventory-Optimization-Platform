"""
Tests for DemandForecaster (Phase 2).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.forecaster import DemandForecaster, compute_rmse


@pytest.fixture
def dummy_config():
    return {
        "forecasting": {
            "model": "lightgbm",
            "forecast_horizon": 5,
            "cv_splits": 2
        }
    }


@pytest.fixture
def dummy_features_df():
    """Generates 30 days of data for 2 items in 1 store."""
    dates = pd.date_range("2020-01-01", periods=30)
    data = []
    
    for d in dates:
        for item in ["item_1", "item_2"]:
            data.append({
                "date": d,
                "store_id": "store_1",
                "item_id": item,
                "sales": np.random.randint(10, 50),
                "lag_7": np.random.randint(10, 50),
                "rolling_mean_7": np.random.uniform(10, 50),
                "is_weekend": 1 if d.dayofweek >= 5 else 0,
                "event_type_encoded": "Category_A"  # Test categorical feature handling
            })
            
    df = pd.DataFrame(data)
    # Ensure types match the expected inputs
    df["event_type_encoded"] = df["event_type_encoded"].astype("category")
    return df


def test_forecaster_initialization(dummy_config):
    model = DemandForecaster(dummy_config)
    assert model.model_type == "lightgbm"
    assert model.model is None


def test_compute_rmse():
    y_true = np.array([10, 10, 10])
    y_pred = np.array([10, 12, 10])
    rmse = compute_rmse(y_true, y_pred)
    assert np.isclose(rmse, 1.1547, atol=1e-4)


def test_forecaster_fit_predict(dummy_config, dummy_features_df):
    model = DemandForecaster(dummy_config)
    
    model.fit(dummy_features_df)
    assert model.model is not None
    
    # Predict on the same data just to ensure API works
    preds = model.predict(dummy_features_df)
    
    assert len(preds) == len(dummy_features_df)
    # Ensure no negative predictions
    assert (preds >= 0).all()


def test_forecaster_train_cv(dummy_config, dummy_features_df):
    model = DemandForecaster(dummy_config)
    
    # Run CV
    metrics = model.train_cv(dummy_features_df)
    
    # Should return RMSE and WMAPE
    assert "rmse" in metrics
    assert "wmape" in metrics
    
    # Since it's random data, metrics could be anything, but shouldn't be nan
    assert not np.isnan(metrics["rmse"])
    assert not np.isnan(metrics["wmape"])


def test_forecaster_save_load(tmp_path, dummy_config, dummy_features_df):
    model = DemandForecaster(dummy_config)
    model.fit(dummy_features_df)
    
    preds_before = model.predict(dummy_features_df)
    
    model.save(str(tmp_path))
    
    # Create new instance and load
    new_model = DemandForecaster(dummy_config)
    
    filepath = tmp_path / f"{model.model_type}_forecaster.pkl"
    new_model.load(str(filepath))
    
    preds_after = new_model.predict(dummy_features_df)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(preds_before, preds_after)
