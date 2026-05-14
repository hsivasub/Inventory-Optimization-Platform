import pandas as pd
import pytest

from src.simulation.environment import InventorySimulationEnvironment
from src.simulation.policies import RuleBasedPolicy
from src.simulation.runner import SimulationRunner

@pytest.fixture
def base_config():
    return {
        "simulation": {
            "horizon_days": 3,
            "lead_time_min": 1,
            "lead_time_max": 1,
            "initial_inventory_multiplier": 0, # Start with 0 inventory to force orders
            "rule_based": {
                "min_days_supply": 2,
                "max_days_supply": 5
            }
        },
        "optimization": {
            "holding_cost": 1.0,
            "stockout_penalty": 10.0
        }
    }

@pytest.fixture
def dummy_data():
    true_demand = []
    forecast_demand = []
    
    for day in range(3):
        true_demand.append({"day": day, "store_id": "S1", "item_id": "I1", "demand": 10.0})
        forecast_demand.append({"day": day, "store_id": "S1", "item_id": "I1", "forecasted_demand": 10.0})
        
    return pd.DataFrame(true_demand), pd.DataFrame(forecast_demand)

def test_simulation_runner(base_config, dummy_data):
    true_demand_df, forecast_demand_df = dummy_data
    
    initial_stats = pd.DataFrame([{"store_id": "S1", "item_id": "I1", "avg_demand": 10.0}])
    env = InventorySimulationEnvironment(base_config, initial_stats)
    env.inventory = {"S1": {"I1": 0.0}} # Force 0 stock
    
    policy = RuleBasedPolicy(base_config)
    runner = SimulationRunner(base_config, env, policy)
    
    warehouse_stock = {"I1": 1000.0}
    
    hist_df = runner.run(true_demand_df, forecast_demand_df, warehouse_stock)
    
    assert len(hist_df) == 3
    
    # Day 0: Starts with 0 stock. 10 demand -> 10 stockout.
    # Policy runs on day 0: min=20, max=50. Order qty = 50. Arrives on day 1 (LT=1).
    assert hist_df.iloc[0]["stockouts"] == 10.0
    
    # Day 1: 50 arrives. 10 demand -> 0 stockout, 40 stock remaining.
    assert hist_df.iloc[1]["stockouts"] == 0.0
    assert hist_df.iloc[1]["holding_cost"] == 40.0 * 1.0
    
    kpis = runner.get_summary_kpis()
    assert kpis["total_stockouts"] == 10.0
    assert "avg_fulfillment_rate" in kpis
