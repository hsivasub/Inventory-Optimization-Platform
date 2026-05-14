from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pandas as pd

from src.optimization.solver import InventoryOptimizer
from src.simulation.environment import InventorySimulationEnvironment


class BasePolicy(ABC):
    """Abstract base class for all replenishment policies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_orders(
        self, 
        env_state: InventorySimulationEnvironment, 
        forecasted_demand: pd.DataFrame, 
        warehouse_stock: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Determine replenishment orders based on current environment state and forecast.
        
        Args:
            env_state: The current simulation environment.
            forecasted_demand: DataFrame with 'store_id', 'item_id', 'forecasted_demand'
            warehouse_stock: Dict of available warehouse stock {item_id: quantity}
            
        Returns:
            List of order dictionaries: [{"store_id": str, "item_id": str, "qty": float}]
        """
        pass


class NaivePolicy(BasePolicy):
    """
    Naive policy: Orders exactly what the forecasted demand for the next day is, 
    plus a buffer for lead time, disregarding current inventory entirely,
    or just ordering a flat amount. 
    A more standard naive policy orders enough to cover the forecasted demand 
    over the expected lead time minus current stock and in-transit.
    """
    
    def get_orders(
        self, 
        env_state: InventorySimulationEnvironment, 
        forecasted_demand: pd.DataFrame, 
        warehouse_stock: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        orders = []
        lt_expected = (env_state.lt_min + env_state.lt_max) / 2.0
        buffer_days = self.config.get("simulation", {}).get("naive_buffer_days", 1)
        
        # Calculate in-transit per store-item
        in_transit_totals = {}
        for order in env_state.in_transit:
            s, i, q = str(order["store_id"]), str(order["item_id"]), order["qty"]
            if s not in in_transit_totals:
                in_transit_totals[s] = {}
            in_transit_totals[s][i] = in_transit_totals[s].get(i, 0.0) + q

        for _, row in forecasted_demand.iterrows():
            s = str(row["store_id"])
            i = str(row["item_id"])
            d = float(row.get("forecasted_demand", 0.0))
            
            # Target inventory to cover lead time + buffer
            target_stock = d * (lt_expected + buffer_days)
            
            current_stock = env_state.inventory.get(s, {}).get(i, 0.0)
            in_transit_stock = in_transit_totals.get(s, {}).get(i, 0.0)
            
            qty_to_order = target_stock - current_stock - in_transit_stock
            
            if qty_to_order > 0:
                orders.append({"store_id": s, "item_id": i, "qty": qty_to_order})
                
        return orders


class RuleBasedPolicy(BasePolicy):
    """
    Min-Max inventory policy.
    If current inventory + in-transit drops below MIN, order up to MAX.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        rule_cfg = config.get("simulation", {}).get("rule_based", {})
        self.min_days = rule_cfg.get("min_days_supply", 3)
        self.max_days = rule_cfg.get("max_days_supply", 7)
        
    def get_orders(
        self, 
        env_state: InventorySimulationEnvironment, 
        forecasted_demand: pd.DataFrame, 
        warehouse_stock: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        orders = []
        
        in_transit_totals = {}
        for order in env_state.in_transit:
            s, i, q = str(order["store_id"]), str(order["item_id"]), order["qty"]
            if s not in in_transit_totals:
                in_transit_totals[s] = {}
            in_transit_totals[s][i] = in_transit_totals[s].get(i, 0.0) + q
            
        for _, row in forecasted_demand.iterrows():
            s = str(row["store_id"])
            i = str(row["item_id"])
            d = float(row.get("forecasted_demand", 0.0))
            
            current_stock = env_state.inventory.get(s, {}).get(i, 0.0)
            in_transit_stock = in_transit_totals.get(s, {}).get(i, 0.0)
            total_position = current_stock + in_transit_stock
            
            min_stock = d * self.min_days
            max_stock = d * self.max_days
            
            if total_position < min_stock:
                qty_to_order = max_stock - total_position
                if qty_to_order > 0:
                    orders.append({"store_id": s, "item_id": i, "qty": qty_to_order})
                    
        return orders


class OptimizationPolicy(BasePolicy):
    """
    Uses the OR-Tools InventoryOptimizer to allocate warehouse stock to stores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimizer = InventoryOptimizer(config)
        
    def get_orders(
        self, 
        env_state: InventorySimulationEnvironment, 
        forecasted_demand: pd.DataFrame, 
        warehouse_stock: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        
        # Calculate total inventory position = current + in_transit
        # Because we don't want to over-allocate to stores that already have stock on the way.
        inventory_position: Dict[str, Dict[str, float]] = {}
        
        # Initialize with current stock
        for s, items in env_state.inventory.items():
            inventory_position[s] = {i: qty for i, qty in items.items()}
            
        # Add in-transit
        for order in env_state.in_transit:
            s, i, q = str(order["store_id"]), str(order["item_id"]), order["qty"]
            if s not in inventory_position:
                inventory_position[s] = {}
            inventory_position[s][i] = inventory_position[s].get(i, 0.0) + q
            
        # The optimizer expects `inventory` which we treat as `inventory_position`
        # and `demand` which we pass as `forecasted_demand`.
        # However, to let the optimizer look slightly ahead, we could pass lead-time adjusted demand.
        # For simplicity, we just pass the daily forecast, but multiplied by lead time expected.
        
        adjusted_demand = forecasted_demand.copy()
        lt_expected = (env_state.lt_min + env_state.lt_max) / 2.0
        if "forecasted_demand" in adjusted_demand.columns:
            adjusted_demand["forecasted_demand"] = adjusted_demand["forecasted_demand"] * lt_expected
            
        allocation_df = self.optimizer.solve_allocation(
            demand=adjusted_demand,
            inventory=inventory_position,
            warehouse_stock=warehouse_stock
        )
        
        orders = []
        if not allocation_df.empty:
            for _, row in allocation_df.iterrows():
                qty = float(row["allocated_qty"])
                if qty > 0:
                    orders.append({
                        "store_id": str(row["store_id"]),
                        "item_id": str(row["item_id"]),
                        "qty": qty
                    })
                    
        return orders
