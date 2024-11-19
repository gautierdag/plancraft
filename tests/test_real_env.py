import pytest

try:
    import minerl
    from plancraft.environments.env_real import RealPlancraft
except ImportError:
    pytest.skip("minerl package not found", allow_module_level=True)


@pytest.mark.slow
def test_real_env_symbolic_commands():
    env = RealPlancraft(
        symbolic_observation_space=True,
        symbolic_action_space=True,
        inventory=[dict(type="iron_ore", quantity=10, slot=10)]
        + [dict(type="oak_log", quantity=23, slot=23)],
    )

    done = False
    action = env.action_space.no_op()
    obs, _, done, _ = env.step(action)

    actions_list = [
        ("smelt", [10, 11, 1]),
        ("smelt", [10, 11, 1]),
        ("smelt", [10, 11, 1]),
        ("inventory_command", [11, 1, 1]),
        ("inventory_command", [11, 3, 1]),
        ("inventory_command", [11, 5, 1]),
        ("inventory_command", [0, 12, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [23, 1, 1]),
        ("inventory_command", [0, 30, 4]),
        ("inventory_command", [0, 30, 4]),
        ("inventory_command", [0, 30, 4]),
        ("inventory_command", [0, 30, 4]),
        ("inventory_command", [0, 0, 0]),
        ("inventory_command", [0, 0, 0]),
        ("inventory_command", [0, 0, 0]),
        ("inventory_command", [0, 0, 0]),
    ]

    for action_name, action_args in actions_list:
        # Take a random action
        action = env.action_space.no_op()
        action[action_name] = action_args
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.close()

    assert all([k["type"] != "iron_ingot" for k in obs["inventory"]])
    assert any([k for k in obs["inventory"] if k["type"] != "bucket"])
    assert any([k for k in obs["inventory"] if k["type"] != "oak_planks"])

    number_of_planks = sum(
        [
            k["quantity"]
            for k in obs["inventory"]
            if (k["type"] == "oak_planks" and k["index"] != 0)
        ]
    )
    # made 4 planks 4 times
    assert number_of_planks == 16

    number_of_oak_logs = sum(
        [k["quantity"] for k in obs["inventory"] if (k["type"] == "oak_log")]
    )
    # used 4 logs 4 times
    assert number_of_oak_logs == 19
