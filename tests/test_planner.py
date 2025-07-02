from plancraft.environment.planner import optimal_planner
from plancraft.environment.planner import get_subplans


def test_optimal_planner_difficult():
    inv = {
        "coal_block": 1,
        "jungle_planks": 1,
        "stripped_oak_wood": 1,
        "gray_carpet": 39,
        "ice": 6,
        "cracked_nether_bricks": 28,
        "peony": 34,
    }
    target = "torch"
    steps = optimal_planner(target, inv)
    assert len(steps) == 4

    inv = {
        "red_dye": 3,
        "cornflower": 3,
        "string": 24,
        "birch_planks": 1,
        "oak_log": 1,
    }
    target = "purple_banner"
    steps = optimal_planner(target, inv)
    assert len(steps) == 21

    inv = {
        "nether_quartz_ore": 24,
        "yellow_bed": 1,
        "red_sandstone_slab": 5,
    }

    target = "smooth_quartz_stairs"
    steps = optimal_planner(target, inv)
    assert len(steps) == 37


def test_optimal_planner_impossible():
    target = "purple_banner"
    steps = optimal_planner(target, {})
    assert steps is None
    inv = {
        "red_dye": 3,
        "cornflower": 3,
        "birch_planks": 1,
        "oak_log": 1,
    }
    steps = optimal_planner(target, inv)
    assert steps is None


def test_get_subplans():
    observation = {
        "inventory": {
            1: {"type": "oak_planks", "quantity": 1},  # A1
            7: {"type": "jungle_planks", "quantity": 1},  # C1
            2: {"type": "jungle_planks", "quantity": 1},  # A2
            3: {"type": "jungle_planks", "quantity": 1},  # A3
            11: {"type": "coal", "quantity": 1},  # B3
            4: {"type": "jungle_planks", "quantity": 1},  # B1
        },
        "target": "torch",
    }
    plan = get_subplans(observation)
