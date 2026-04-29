import pytest
import pandas as pd
from src.optimization.solver import InventoryOptimizer


@pytest.fixture
def config():
    return {
        "optimization": {
            "solver": "GLOP",
            "stockout_penalty": 100.0,
            "holding_cost": 1.0,
            "transport_cost": 2.0,
        }
    }


def test_abundant_warehouse(config):
    """
    When the warehouse has infinite stock, it should fulfill all demand precisely,
    avoiding stockouts and excess holding costs.
    """
    optimizer = InventoryOptimizer(config)
    demand = pd.DataFrame(
        [
            {"store_id": "Store_1", "item_id": "Item_A", "demand": 50},
            {"store_id": "Store_2", "item_id": "Item_A", "demand": 30},
        ]
    )

    # Stores have 0 inventory
    inventory = {"Store_1": {"Item_A": 0}, "Store_2": {"Item_A": 0}}

    # Abundant stock
    warehouse_stock = {"Item_A": 1000}

    allocations = optimizer.solve_allocation(demand, inventory, warehouse_stock)

    assert len(allocations) == 2
    alloc_1 = allocations[allocations["store_id"] == "Store_1"]["allocated_qty"].values[
        0
    ]
    alloc_2 = allocations[allocations["store_id"] == "Store_2"]["allocated_qty"].values[
        0
    ]

    # Should allocate exactly what is demanded to minimize holding cost and stockout
    assert pytest.approx(alloc_1, 0.01) == 50.0
    assert pytest.approx(alloc_2, 0.01) == 30.0


def test_constrained_warehouse(config):
    """
    When warehouse stock is less than total demand, the solver should allocate all
    available stock to minimize stockout penalty.
    """
    optimizer = InventoryOptimizer(config)
    demand = pd.DataFrame(
        [
            {"store_id": "Store_1", "item_id": "Item_A", "demand": 50},
            {"store_id": "Store_2", "item_id": "Item_A", "demand": 50},
        ]
    )

    inventory = {"Store_1": {"Item_A": 0}, "Store_2": {"Item_A": 0}}
    warehouse_stock = {"Item_A": 60}  # Only 60 available for 100 demand

    allocations = optimizer.solve_allocation(demand, inventory, warehouse_stock)

    total_allocated = allocations["allocated_qty"].sum()
    assert pytest.approx(total_allocated, 0.01) == 60.0


def test_high_transport_cost(config):
    """
    If transport cost > stockout penalty, it's cheaper to take the stockout than to ship.
    (Testing edge case for objective balance).
    """
    cfg = config.copy()
    cfg["optimization"]["transport_cost"] = 200.0
    cfg["optimization"]["stockout_penalty"] = 100.0

    optimizer = InventoryOptimizer(cfg)
    demand = pd.DataFrame([{"store_id": "Store_1", "item_id": "Item_A", "demand": 50}])
    inventory = {"Store_1": {"Item_A": 0}}
    warehouse_stock = {"Item_A": 1000}

    allocations = optimizer.solve_allocation(demand, inventory, warehouse_stock)
    total_allocated = allocations["allocated_qty"].sum()

    # It should not allocate anything because shipping is too expensive
    assert pytest.approx(total_allocated, 0.01) == 0.0


def test_existing_inventory(config):
    """
    If a store already has enough inventory, it shouldn't receive an allocation.
    """
    optimizer = InventoryOptimizer(config)
    demand = pd.DataFrame(
        [
            {"store_id": "Store_1", "item_id": "Item_A", "demand": 50},
            {"store_id": "Store_2", "item_id": "Item_A", "demand": 50},
        ]
    )

    # Store_1 has enough, Store_2 has none
    inventory = {"Store_1": {"Item_A": 60}, "Store_2": {"Item_A": 0}}
    warehouse_stock = {"Item_A": 1000}

    allocations = optimizer.solve_allocation(demand, inventory, warehouse_stock)

    alloc_1 = allocations[allocations["store_id"] == "Store_1"]["allocated_qty"].values[
        0
    ]
    alloc_2 = allocations[allocations["store_id"] == "Store_2"]["allocated_qty"].values[
        0
    ]

    assert pytest.approx(alloc_1, 0.01) == 0.0
    assert pytest.approx(alloc_2, 0.01) == 50.0
