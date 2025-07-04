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
    subplans, plan = get_subplans(observation)
    assert len(plan) == 2


def test_get_subplans_clear_first():
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
    subplans_clear, plan_clear = get_subplans(observation, clear_first=True)
    subplans_no_clear, plan_no_clear = get_subplans(observation, clear_first=False)
    # adds the clear action to plan
    assert len(plan_clear) == len(plan_no_clear) + 1

    # flatten 2d list
    flat_subplan_clear = [action for subplan in subplans_clear for action in subplan]
    flat_subplan_no_clear = [
        action for subplan in subplans_no_clear for action in subplan
    ]
    assert len(flat_subplan_clear) >= len(flat_subplan_no_clear)
