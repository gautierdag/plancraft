from plancraft.environment.actions import MoveAction, SmeltAction
from plancraft.environment.env import PlancraftEnvironment


def test_symbolic_env_symbolic_commands():
    env = PlancraftEnvironment(
        inventory={
            10: dict(type="iron_ore", quantity=10),
            23: dict(type="oak_log", quantity=23),
        }
    )

    obs = env.step()
    actions_list = [
        SmeltAction(slot_from=10, slot_to=11, quantity=1),
        SmeltAction(slot_from=10, slot_to=11, quantity=1),
        SmeltAction(slot_from=10, slot_to=11, quantity=1),
        MoveAction(slot_from=11, slot_to=1, quantity=1),
        MoveAction(slot_from=11, slot_to=3, quantity=1),
        MoveAction(slot_from=11, slot_to=5, quantity=1),
        MoveAction(slot_from=0, slot_to=12, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=23, slot_to=1, quantity=1),
        MoveAction(slot_from=0, slot_to=30, quantity=4),
        MoveAction(slot_from=0, slot_to=30, quantity=4),
        MoveAction(slot_from=0, slot_to=30, quantity=4),
        MoveAction(slot_from=0, slot_to=30, quantity=4),
        None,
        None,
        None,
        None,
    ]

    for action in actions_list:
        # Take action
        obs = env.step(action)

    assert all([k["type"] != "iron_ingot" for k in obs["inventory"].values()])
    assert any([k for k in obs["inventory"].values() if k["type"] != "bucket"])
    assert any([k for k in obs["inventory"].values() if k["type"] != "oak_planks"])

    number_of_planks = sum(
        [
            k["quantity"]
            for s, k in obs["inventory"].items()
            if (k["type"] == "oak_planks" and s != 0)
        ]
    )
    # # made 4 planks 4 times
    assert number_of_planks == 16

    number_of_oak_logs = sum(
        [k["quantity"] for k in obs["inventory"].values() if (k["type"] == "oak_log")]
    )
    # used 4 logs 4 times
    assert number_of_oak_logs == 19


def test_add_item_to_slot():
    env = PlancraftEnvironment()
    env.add_item_to_slot("iron_ore", 1, 10)
    assert env.state[1]["type"] == "iron_ore"
    assert env.state[1]["quantity"] == 10


def test_remove_item_from_slot():
    env = PlancraftEnvironment()
    env.add_item_to_slot("iron_ore", 1, 10)
    env.remove_item_from_slot(1)
    assert 1 not in env.state


def test_change_quantity_in_slot():
    env = PlancraftEnvironment()
    env.add_item_to_slot("iron_ore", 1, 10)
    env.change_quantity_in_slot(1, 5)
    assert env.state[1]["quantity"] == 5
    env.change_quantity_in_slot(1, 0)
    assert 1 not in env.state


def test_reset():
    env = PlancraftEnvironment(inventory={1: {"type": "iron_ore", "quantity": 10}})
    action = MoveAction(slot_from=1, slot_to=2, quantity=5)
    env.step(action)
    assert env.state[2]["quantity"] == 5

    env.reset({1: {"type": "iron_ore", "quantity": 10}})
    assert env.state[1]["type"] == "iron_ore"
    assert env.state[1]["quantity"] == 10


def test_step_move():
    env = PlancraftEnvironment(
        # inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
        inventory={1: {"type": "iron_ore", "quantity": 10}}
    )
    action = MoveAction(slot_from=1, slot_to=2, quantity=5)
    env.step(action)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ore"
    assert env.state[2]["quantity"] == 5


def test_step_smelt():
    env = PlancraftEnvironment(
        # inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
        inventory={1: {"type": "iron_ore", "quantity": 10}}
    )
    action = SmeltAction(slot_from=1, slot_to=2, quantity=5)
    env.step(action)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ingot"
    assert env.state[2]["quantity"] == 5


def test_clean_state():
    env = PlancraftEnvironment()
    env.add_item_to_slot("iron_ore", 1, 10)
    env.change_quantity_in_slot(1, 0)
    env.clean_state()
    assert 1 not in env.state


def test_move_item():
    env = PlancraftEnvironment(
        # inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
        inventory={1: {"type": "iron_ore", "quantity": 10}}
    )
    env.move_item(1, 2, 5)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ore"
    assert env.state[2]["quantity"] == 5


def test_smelt_item():
    env = PlancraftEnvironment(inventory={1: {"type": "iron_ore", "quantity": 10}})
    env.smelt_item(1, 2, 5)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ingot"
    assert env.state[2]["quantity"] == 5


def test_populate_craft_slot_craft_item():
    env = PlancraftEnvironment(inventory={1: {"type": "oak_log", "quantity": 10}})
    env.add_item_to_slot("oak_log", 1, 1)
    env.populate_craft_slot_craft_item()
    assert env.state[0]["type"] == "oak_planks"
    assert env.state[0]["quantity"] == 4


def test_use_ingredients():
    env = PlancraftEnvironment(inventory={1: {"type": "oak_log", "quantity": 10}})
    env.add_item_to_slot("oak_log", 1, 1)
    env.populate_craft_slot_craft_item()
    env.use_ingredients()
    assert 1 not in env.state
