"""
Inventory Simulation Environment.

Acts as a state-machine that processes daily store demands, applies replenishment
orders, enforces varying lead times, and calculates stockouts & carrying costs.
"""

import math
import random
from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class InventorySimulationEnvironment:
    """
    Simulates a retail supply chain environment over a fixed horizon.
    """

    def __init__(self, config: Dict[str, Any], initial_demand_stats: pd.DataFrame):
        """
        initial_demand_stats: A dataframe with average daily demand per store/item
        so we can compute starting inventory (buffer multiplier).
        Should contain at least: store_id, item_id, avg_demand
        """
        self.config = config
        self.sim_cfg = config.get("simulation", {})
        self.opt_cfg = config.get("optimization", {})
        
        self.horizon_days = self.sim_cfg.get("horizon_days", 30)
        self.lt_min = self.sim_cfg.get("lead_time_min", 2)
        self.lt_max = self.sim_cfg.get("lead_time_max", 5)
        self.inv_mult = self.sim_cfg.get("initial_inventory_multiplier", 14)
        
        # Costs
        self.holding_cost = self.opt_cfg.get("holding_cost", 1.0)
        self.stockout_penalty = self.opt_cfg.get("stockout_penalty", 100.0)
        
        # State variables
        self.current_day = 0
        self.inventory: Dict[str, Dict[str, float]] = {}  # {store: {item: qty}}
        self.in_transit: List[Dict[str, Any]] = []        # [{store, item, qty, arrival_day}]
        self.history: List[Dict[str, Any]] = []
        
        self._initialize_inventory(initial_demand_stats)

    def _initialize_inventory(self, demand_stats: pd.DataFrame):
        """Sets up day 0 inventory based on historical average demand."""
        for _, row in demand_stats.iterrows():
            store = str(row["store_id"])
            item = str(row["item_id"])
            avg_demand = max(0, float(row["avg_demand"]))
            
            if store not in self.inventory:
                self.inventory[store] = {}
            
            # Start with an initial buffer of stock
            self.inventory[store][item] = math.ceil(avg_demand * self.inv_mult)
            
        log.info(f"Initialized simulation environment for {self.horizon_days} days.")

    def _process_arrivals(self):
        """Adds orders that have completed their lead time to inventory."""
        arrived = [order for order in self.in_transit if order["arrival_day"] <= self.current_day]
        pending = [order for order in self.in_transit if order["arrival_day"] > self.current_day]
        
        for order in arrived:
            s, i, q = order["store_id"], order["item_id"], order["qty"]
            if s in self.inventory and i in self.inventory[s]:
                self.inventory[s][i] += q
            else:
                if s not in self.inventory:
                    self.inventory[s] = {}
                self.inventory[s][i] = q
                
        # Keep only the pending orders
        self.in_transit = pending

    def _get_random_lead_time(self) -> int:
        """Sample a lead time to simulate vendor uncertainty."""
        return random.randint(self.lt_min, self.lt_max)

    def place_orders(self, orders: List[Dict[str, Any]]):
        """
        Accepts replenishment orders placed by policy on current day.
        orders format: [{"store_id": X, "item_id": Y, "qty": Z}, ...]
        """
        for order in orders:
            qty = order.get("qty", 0)
            if qty > 0:
                self.in_transit.append({
                    "store_id": str(order["store_id"]),
                    "item_id": str(order["item_id"]),
                    "qty": qty,
                    "arrival_day": self.current_day + self._get_random_lead_time()
                })

    def step(self, daily_demand: pd.DataFrame, orders: List[Dict[str, Any]] = None):
        """
        Progresses simulation by 1 day.
        
        daily_demand: DataFrame with store_id, item_id, demand (true sales)
        orders: Optional list of dicts specifying replenishment quantities.
        """
        if self.current_day >= self.horizon_days:
            log.warning("Simulation horizon reached. Further steps ignored.")
            return

        # 1. Place new orders (they are placed *today* and arrive in future)
        if orders:
            self.place_orders(orders)

        # 2. Receive inbound stock that arrives today
        self._process_arrivals()

        total_stockouts = 0
        total_holding_cost = 0.0
        total_stockout_penalty = 0.0
        daily_fulfillment_rate = 0.0
        total_demand = 0.0
        total_sales = 0.0
        
        # 3. Fulfill demand & calculate costs
        for _, row in daily_demand.iterrows():
            s = str(row["store_id"])
            i = str(row["item_id"])
            # Demand is true real-world sales capacity
            demand = float(row.get("demand", row.get("sales", 0.0)))
            
            # Setup default missing items safely
            if s not in self.inventory:
                self.inventory[s] = {}
            current_stock = self.inventory[s].get(i, 0.0)
            
            sales = min(demand, current_stock)
            stockout = demand - sales
            
            self.inventory[s][i] = current_stock - sales
            new_stock = self.inventory[s][i]
            
            # Metrics
            total_demand += demand
            total_sales += sales
            if stockout > 0:
                total_stockouts += stockout
                total_stockout_penalty += (stockout * self.stockout_penalty)
            
            total_holding_cost += (new_stock * self.holding_cost)
            
        if total_demand > 0:
            daily_fulfillment_rate = total_sales / total_demand
        else:
            daily_fulfillment_rate = 1.0
            
        # Log state
        self.history.append({
            "day": self.current_day,
            "fulfillment_rate": daily_fulfillment_rate,
            "stockouts": total_stockouts,
            "stockout_penalty": total_stockout_penalty,
            "holding_cost": total_holding_cost,
            "total_cost": total_holding_cost + total_stockout_penalty
        })
        
        self.current_day += 1

    def get_history(self) -> pd.DataFrame:
        """Returns the logged metrics as a DataFrame."""
        return pd.DataFrame(self.history)
