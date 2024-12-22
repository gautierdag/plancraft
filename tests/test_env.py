from plancraft.environment.env import PlancraftEnvironment


def test_symbolic_env_symbolic_commands():
    env = PlancraftEnvironment(
        inventory=[dict(type="iron_ore", quantity=10, slot=10)]
        + [dict(type="oak_log", quantity=23, slot=23)],
    )
    obs = env.step()
    actions_list = [
        ("smelt", [10, 11, 1]),
        ("smelt", [10, 11, 1]),
        ("smelt", [10, 11, 1]),
        ("move", [11, 1, 1]),
        ("move", [11, 3, 1]),
        ("move", [11, 5, 1]),
        ("move", [0, 12, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [23, 1, 1]),
        ("move", [0, 30, 4]),
        ("move", [0, 30, 4]),
        ("move", [0, 30, 4]),
        ("move", [0, 30, 4]),
        ("move", [0, 0, 0]),
        ("move", [0, 0, 0]),
        ("move", [0, 0, 0]),
        ("move", [0, 0, 0]),
    ]

    for action_name, action_args in actions_list:
        # Take a random action
        obs = env.step({action_name: action_args})
        print(env.state[11])

    assert all([k["type"] != "iron_ingot" for k in obs["inventory"]])
    assert any([k for k in obs["inventory"] if k["type"] != "bucket"])
    assert any([k for k in obs["inventory"] if k["type"] != "oak_planks"])

    number_of_planks = sum(
        [
            k["quantity"]
            for k in obs["inventory"]
            if (k["type"] == "oak_planks" and k["slot"] != 0)
        ]
    )
    # made 4 planks 4 times
    assert number_of_planks == 16

    number_of_oak_logs = sum(
        [k["quantity"] for k in obs["inventory"] if (k["type"] == "oak_log")]
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
    assert env.state[1]["type"] == "air"
    assert env.state[1]["quantity"] == 0


def test_change_quantity_in_slot():
    env = PlancraftEnvironment()
    env.add_item_to_slot("iron_ore", 1, 10)
    env.change_quantity_in_slot(1, 5)
    assert env.state[1]["quantity"] == 5
    env.change_quantity_in_slot(1, 0)
    assert env.state[1]["type"] == "air"
    assert env.state[1]["quantity"] == 0


def test_reset_state():
    env = PlancraftEnvironment(
        inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
    )
    env.reset_state()
    assert env.state[1]["type"] == "iron_ore"
    assert env.state[1]["quantity"] == 10


def test_step_move():
    env = PlancraftEnvironment(
        inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
    )
    action = {"move": [1, 2, 5]}
    env.step(action)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ore"
    assert env.state[2]["quantity"] == 5


def test_step_smelt():
    env = PlancraftEnvironment(
        inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
    )
    action = {"smelt": [1, 2, 5]}
    env.step(action)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ingot"
    assert env.state[2]["quantity"] == 5


def test_clean_state():
    env = PlancraftEnvironment()
    env.add_item_to_slot("iron_ore", 1, 10)
    env.change_quantity_in_slot(1, 0)
    env.clean_state()
    assert env.state[1]["type"] == "air"
    assert env.state[1]["quantity"] == 0


def test_move_item():
    env = PlancraftEnvironment(
        inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
    )
    env.move_item(1, 2, 5)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ore"
    assert env.state[2]["quantity"] == 5


def test_smelt_item():
    env = PlancraftEnvironment(
        inventory=[{"type": "iron_ore", "quantity": 10, "slot": 1}]
    )
    env.smelt_item(1, 2, 5)
    assert env.state[1]["quantity"] == 5
    assert env.state[2]["type"] == "iron_ingot"
    assert env.state[2]["quantity"] == 5


def test_populate_craft_slot_craft_item():
    env = PlancraftEnvironment(
        inventory=[{"type": "oak_log", "quantity": 10, "slot": 1}]
    )
    env.add_item_to_slot("oak_log", 1, 1)
    env.populate_craft_slot_craft_item()
    assert env.state[0]["type"] == "oak_planks"
    assert env.state[0]["quantity"] == 4


def test_use_ingredients():
    env = PlancraftEnvironment(
        inventory=[{"type": "oak_log", "quantity": 10, "slot": 1}]
    )
    env.add_item_to_slot("oak_log", 1, 1)
    env.populate_craft_slot_craft_item()
    env.use_ingredients()
    assert env.state[1]["type"] == "air"
    assert env.state[1]["quantity"] == 0


def test_reset():
    env = PlancraftEnvironment()
    new_inventory = [{"type": "iron_ore", "quantity": 10, "slot": 1}]
    env.reset(new_inventory)
    assert env.state[1]["type"] == "iron_ore"
    assert env.state[1]["quantity"] == 10
