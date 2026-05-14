import pandas as pd
import pytest
from src.simulation.environment import InventorySimulationEnvironment
from src.simulation.policies import NaivePolicy, RuleBasedPolicy, OptimizationPolicy

@pytest.fixture
def env_config():
    return {
        "simulation": {
            "horizon_days": 10,
            "lead_time_min": 2,
            "lead_time_max": 2,
            "initial_inventory_multiplier": 5,
            "naive_buffer_days": 1,
            "rule_based": {
                "min_days_supply": 2,
                "max_days_supply": 5
            }
        },
        "optimization": {
            "holding_cost": 1.0,
            "stockout_penalty": 10.0,
            "transport_cost": 0.5,
            "solver": "GLOP"
        }
    }

@pytest.fixture
def mock_env(env_config):
    initial_demand = pd.DataFrame([
        {"store_id": "S1", "item_id": "I1", "avg_demand": 10.0}
    ])
    env = InventorySimulationEnvironment(env_config, initial_demand)
    # Reset inventory manually for deterministic testing
    env.inventory = {"S1": {"I1": 15.0}}
    env.in_transit = []
    return env

@pytest.fixture
def forecasted_demand():
    return pd.DataFrame([
        {"store_id": "S1", "item_id": "I1", "forecasted_demand": 10.0}
    ])

def test_naive_policy(env_config, mock_env, forecasted_demand):
    policy = NaivePolicy(env_config)
    
    # Lead time expected = 2
    # Buffer days = 1
    # Target stock = 10 * (2 + 1) = 30
    # Current stock = 15, In transit = 0
    # Order Qty = 30 - 15 = 15
    
    orders = policy.get_orders(mock_env, forecasted_demand, {})
    
    assert len(orders) == 1
    assert orders[0]["store_id"] == "S1"
    assert orders[0]["item_id"] == "I1"
    assert orders[0]["qty"] == 15.0

def test_rule_based_policy(env_config, mock_env, forecasted_demand):
    policy = RuleBasedPolicy(env_config)
    
    # Min = 2 * 10 = 20
    # Max = 5 * 10 = 50
    # Current = 15. Since 15 < 20, order up to 50.
    # Order Qty = 50 - 15 = 35.
    
    orders = policy.get_orders(mock_env, forecasted_demand, {})
    
    assert len(orders) == 1
    assert orders[0]["qty"] == 35.0

    # Add in-transit so position > 20
    mock_env.in_transit = [{"store_id": "S1", "item_id": "I1", "qty": 10.0, "arrival_day": 2}]
    # Total pos = 15 + 10 = 25. 25 is not < 20. Order qty = 0.
    orders = policy.get_orders(mock_env, forecasted_demand, {})
    assert len(orders) == 0

def test_optimization_policy(env_config, mock_env, forecasted_demand):
    policy = OptimizationPolicy(env_config)
    
    # With 15 stock and demand of 10*2=20 over lead time, we might be short 5.
    warehouse_stock = {"I1": 100.0}
    
    orders = policy.get_orders(mock_env, forecasted_demand, warehouse_stock)
    
    # We should see an order placed to cover the deficit.
    # The solver will try to balance holding vs stockout.
    # Stockout penalty = 10, holding cost = 1.
    # We expect some order quantity.
    
    assert len(orders) > 0
    assert orders[0]["store_id"] == "S1"
    assert orders[0]["item_id"] == "I1"
    assert orders[0]["qty"] > 0
