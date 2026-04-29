import pandas as pd
from ortools.linear_solver import pywraplp
from typing import Dict, Any

from src.utils.logger import get_logger

log = get_logger(__name__)


class InventoryOptimizer:
    """
    OR-Tools based optimization engine for daily inventory allocation.
    """

    def __init__(self, config: Dict[str, Any]):
        opt_cfg = config.get("optimization", {})
        self.solver_name = opt_cfg.get("solver", "GLOP")
        self.stockout_penalty = float(opt_cfg.get("stockout_penalty", 100.0))
        self.holding_cost = float(opt_cfg.get("holding_cost", 1.0))
        self.transport_cost = float(opt_cfg.get("transport_cost", 2.0))
        self.time_limit = int(opt_cfg.get("solver_time_limit_sec", 30))

    def solve_allocation(
        self,
        demand: pd.DataFrame,
        inventory: Dict[str, Dict[str, float]],
        warehouse_stock: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Solves the LP allocation problem.

        Args:
            demand: DataFrame containing 'store_id', 'item_id', and 'demand' (or 'forecasted_demand')
            inventory: Nested dict of current inventory {store_id: {item_id: quantity}}
            warehouse_stock: Dict of available stock at warehouse {item_id: quantity}

        Returns:
            DataFrame containing 'store_id', 'item_id', 'allocated_qty'
        """
        solver = pywraplp.Solver.CreateSolver(self.solver_name)
        if not solver:
            log.error(f"Solver {self.solver_name} is not available.")
            return pd.DataFrame(columns=["store_id", "item_id", "allocated_qty"])

        solver.set_time_limit(self.time_limit * 1000)

        # Variables mapping: (store_id, item_id) -> NumVar
        x = {}  # Allocated quantity
        s_out = {}  # Stockout amount
        h_inv = {}  # Holding inventory after allocation and demand fulfillment

        items = set()
        for _, row in demand.iterrows():
            s = str(row["store_id"])
            i = str(row["item_id"])
            items.add(i)

            x[(s, i)] = solver.NumVar(0, solver.infinity(), f"x_{s}_{i}")
            s_out[(s, i)] = solver.NumVar(0, solver.infinity(), f"s_out_{s}_{i}")
            h_inv[(s, i)] = solver.NumVar(0, solver.infinity(), f"h_inv_{s}_{i}")

        # Constraint 1: Warehouse capacity limit for each item
        for i in items:
            wh_stock = warehouse_stock.get(i, 0.0)
            item_stores = [s for (s, item) in x.keys() if item == i]
            solver.Add(sum(x[(s, i)] for s in item_stores) <= wh_stock)

        # Constraint 2: Flow balance equations
        for _, row in demand.iterrows():
            s = str(row["store_id"])
            i = str(row["item_id"])
            d = float(row.get("demand", row.get("forecasted_demand", 0.0)))
            inv = inventory.get(s, {}).get(i, 0.0)

            # End of day inventory minus stockout = Start inventory + Allocation - Demand
            # h_inv[s, i] - s_out[s, i] = inv + x[s, i] - d
            solver.Add(h_inv[(s, i)] - s_out[(s, i)] == inv + x[(s, i)] - d)

        # Objective Function
        objective = solver.Objective()
        for s, i in x.keys():
            objective.SetCoefficient(x[(s, i)], self.transport_cost)
            objective.SetCoefficient(s_out[(s, i)], self.stockout_penalty)
            objective.SetCoefficient(h_inv[(s, i)], self.holding_cost)

        objective.SetMinimization()

        status = solver.Solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            log.warning(
                "Solver could not find an optimal/feasible solution. Returning empty allocations."
            )
            return pd.DataFrame(columns=["store_id", "item_id", "allocated_qty"])

        results = []
        for s, i in x.keys():
            alloc = x[(s, i)].solution_value()
            results.append(
                {
                    "store_id": s,
                    "item_id": i,
                    "allocated_qty": alloc if alloc > 1e-6 else 0.0,
                }
            )

        df_results = pd.DataFrame(results)
        log.info(
            f"Solved allocation for {len(results)} store-item pairs. Total allocated: {df_results['allocated_qty'].sum():.2f}"
        )
        return df_results
