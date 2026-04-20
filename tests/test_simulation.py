"""
Unit tests for InventorySimulationEnvironment.
"""

import math
import pytest
import pandas as pd

from src.simulation.environment import InventorySimulationEnvironment


@pytest.fixture
def base_config():
    return {
        "simulation": {
            "horizon_days": 5,
            "lead_time_min": 2,
            "lead_time_max": 2,  # Fixed for testing deterministic behavior
            "initial_inventory_multiplier": 2
        },
        "optimization": {
            "holding_cost": 1.0,
            "stockout_penalty": 10.0
        }
    }


@pytest.fixture
def initial_demand():
    return pd.DataFrame([
        {"store_id": "Store_1", "item_id": "Item_A", "avg_demand": 10.0}
    ])


def test_initialization(base_config, initial_demand):
    env = InventorySimulationEnvironment(base_config, initial_demand)
    
    # 10 avg_demand * 2 multiplier = 20 initial stock
    assert env.inventory["Store_1"]["Item_A"] == 20
    assert env.current_day == 0


def test_demand_fulfillment_no_stockout(base_config, initial_demand):
    env = InventorySimulationEnvironment(base_config, initial_demand)
    
    daily_demand = pd.DataFrame([
        {"store_id": "Store_1", "item_id": "Item_A", "demand": 15.0}
    ])
    
    env.step(daily_demand)
    
    assert env.inventory["Store_1"]["Item_A"] == 5.0
    
    hist = env.get_history()
    assert len(hist) == 1
    assert hist.iloc[0]["fulfillment_rate"] == 1.0
    assert hist.iloc[0]["stockouts"] == 0.0
    assert hist.iloc[0]["holding_cost"] == 5.0 * 1.0  # 5 units left * $1
    assert hist.iloc[0]["stockout_penalty"] == 0.0


def test_demand_fulfillment_stockout(base_config, initial_demand):
    env = InventorySimulationEnvironment(base_config, initial_demand)
    
    daily_demand = pd.DataFrame([
        {"store_id": "Store_1", "item_id": "Item_A", "demand": 25.0} # Needs 25, has 20
    ])
    
    env.step(daily_demand)
    
    assert env.inventory["Store_1"]["Item_A"] == 0.0
    
    hist = env.get_history()
    assert hist.iloc[0]["stockouts"] == 5.0
    assert hist.iloc[0]["fulfillment_rate"] == 20.0 / 25.0
    assert hist.iloc[0]["stockout_penalty"] == 5.0 * 10.0  # 5 stockouts * $10
    assert hist.iloc[0]["holding_cost"] == 0.0


def test_replenishment_arrival(base_config, initial_demand):
    env = InventorySimulationEnvironment(base_config, initial_demand)
    
    # Place order on day 0, lead time is exactly 2, should arrive on day 0+2=2
    orders = [{"store_id": "Store_1", "item_id": "Item_A", "qty": 100}]
    
    daily_demand_zero = pd.DataFrame([
        {"store_id": "Store_1", "item_id": "Item_A", "demand": 0.0}
    ])
    
    env.step(daily_demand_zero, orders=orders) # Day 0 -> becomes Day 1
    assert env.inventory["Store_1"]["Item_A"] == 20.0  # Still 20, hasn't arrived
    
    env.step(daily_demand_zero) # Day 1 -> becomes Day 2
    assert env.inventory["Store_1"]["Item_A"] == 20.0
    
    env.step(daily_demand_zero) # Day 2 -> Order arrives BEFORE demand fulfillment! Output becomes Day 3.
    assert env.inventory["Store_1"]["Item_A"] == 120.0

