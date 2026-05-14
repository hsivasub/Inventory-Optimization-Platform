from typing import Dict, Any, List, Optional
import pandas as pd

from src.simulation.environment import InventorySimulationEnvironment
from src.simulation.policies import BasePolicy
from src.utils.logger import get_logger

log = get_logger(__name__)


class SimulationRunner:
    """
    Orchestrates the inventory simulation over a given horizon by 
    combining the Environment with a Replenishment Policy.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        env: InventorySimulationEnvironment, 
        policy: BasePolicy
    ):
        self.config = config
        self.env = env
        self.policy = policy
        self.horizon_days = env.horizon_days

    def run(
        self, 
        true_demand_df: pd.DataFrame, 
        forecast_demand_df: pd.DataFrame,
        warehouse_stock: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Executes the simulation loop.
        
        Args:
            true_demand_df: DataFrame with ['day', 'store_id', 'item_id', 'demand']
            forecast_demand_df: DataFrame with ['day', 'store_id', 'item_id', 'forecasted_demand']
            warehouse_stock: Dict of available stock at the warehouse {item_id: quantity}. 
                             If None, assumes infinite capacity for naive/rule-based.
                             
        Returns:
            DataFrame containing the simulation history/KPIs.
        """
        log.info(f"Starting simulation run for {self.horizon_days} days using {self.policy.__class__.__name__}")
        
        if warehouse_stock is None:
            warehouse_stock = {}
            # For infinite capacity, if an item isn't in warehouse_stock, 
            # policies handle it based on their logic (optimizer needs a large number).
            # We'll rely on the caller to provide large capacities if needed.
            
        for day in range(self.horizon_days):
            # 1. Get today's true demand and forecast
            daily_true_demand = true_demand_df[true_demand_df["day"] == day]
            daily_forecast = forecast_demand_df[forecast_demand_df["day"] == day]
            
            # 2. Get replenishment orders from policy
            orders = self.policy.get_orders(
                env_state=self.env,
                forecasted_demand=daily_forecast,
                warehouse_stock=warehouse_stock
            )
            
            # Deduct orders from warehouse stock (simplified warehouse logic)
            for order in orders:
                i = str(order["item_id"])
                q = order["qty"]
                if i in warehouse_stock:
                    warehouse_stock[i] = max(0.0, warehouse_stock[i] - q)
            
            # 3. Step the environment
            self.env.step(daily_demand=daily_true_demand, orders=orders)
            
        log.info("Simulation run completed.")
        return self.env.get_history()

    def get_summary_kpis(self) -> Dict[str, float]:
        """Calculates summary KPIs from the simulation history."""
        hist_df = self.env.get_history()
        if hist_df.empty:
            return {}
            
        return {
            "avg_fulfillment_rate": float(hist_df["fulfillment_rate"].mean()),
            "total_stockouts": float(hist_df["stockouts"].sum()),
            "total_stockout_penalty": float(hist_df["stockout_penalty"].sum()),
            "total_holding_cost": float(hist_df["holding_cost"].sum()),
            "total_cost": float(hist_df["total_cost"].sum())
        }
