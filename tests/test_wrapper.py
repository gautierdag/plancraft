import pytest

from plancraft.environment.actions import (
    ImpossibleActionHandler,
    MoveActionHandler,
    SmeltActionHandler,
    ThinkActionHandler,
)
from plancraft.environment.planner import get_subplans
from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples


@pytest.fixture
def plancraft_examples():
    return get_plancraft_examples()


def test_all_examples_loading(plancraft_examples):
    """Test loading examples from data directory"""
    for mock_example in plancraft_examples[:5]:
        PlancraftGymWrapper(mock_example, max_steps=30)


def test_max_steps_actions(plancraft_examples):
    example = plancraft_examples[0]

    env = PlancraftGymWrapper(example, max_steps=5)

    action = "move: from [I18] to [B1] with quantity 1"
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert not truncated
        assert "image" in obs
        assert reward == 0
    action = "move: from [I18] to [B1] with quantity 1"
    obs, reward, terminated, truncated, info = env.step(action)
    assert "Max steps" in obs["text"]
    assert reward == 0
    assert not terminated
    assert truncated
    assert info["reason"] == "max_steps_reached"


def test_impossible_task(plancraft_examples):
    """Test handling of impossible tasks"""
    # Find an impossible example
    example = None
    for ex in plancraft_examples:
        if ex.impossible:
            example = ex
            break

    assert example is not None, "No impossible example found for testing"

    env = PlancraftGymWrapper(example, max_steps=10)

    # Try a few actions and then declare impossible
    action = "move: from [I1] to [B1] with quantity 1"
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert not truncated

    # Declare impossible
    action = "impossible: Cannot craft the target with available materials"
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated
    assert not truncated
    assert reward == 1.0
    assert env.success
    assert info["reason"] == "correctly_identified_impossible"


def test_invalid_action_handling(plancraft_examples):
    """Test handling of invalid actions"""
    example = plancraft_examples[0]
    env = PlancraftGymWrapper(example, max_steps=10)

    # Try an invalid action format
    action = "invalid action format"
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert not truncated
    assert reward == 0
    assert "Only select actions from the following" in obs["text"]

    # Try an action with invalid slots
    action = "move: from [Z9] to [B1] with quantity 1"
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert not truncated
    assert reward == 0
    assert "Format Error" in obs["text"]


def test_custom_action_handlers(plancraft_examples):
    """Test PlancraftGymWrapper with custom action handlers"""
    example = plancraft_examples[0]

    # Add ThinkActionHandler to the default handlers
    env = PlancraftGymWrapper(
        example,
        actions=[
            MoveActionHandler(),
            SmeltActionHandler(),
            ImpossibleActionHandler(),
            ThinkActionHandler(),
        ],
        max_steps=10,
    )

    # Try a think action
    action = "think: I need to move items to craft the target"
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert not truncated
    assert reward == 0
    assert obs["text"] == "Ok"


def test_use_text_inventory_parameter(plancraft_examples):
    """Test the use_text_inventory parameter"""
    example = plancraft_examples[0]

    # Test with text inventory enabled (default)
    env1 = PlancraftGymWrapper(example, max_steps=5, use_text_inventory=True)
    obs1, _, _, _, _ = env1.step("move: from [I1] to [B1] with quantity 1")
    assert "inventory" in obs1["text"].lower()

    # Test with text inventory disabled
    env2 = PlancraftGymWrapper(example, max_steps=5, use_text_inventory=False)
    obs2, _, _, _, _ = env2.step("move: from [I1] to [B1] with quantity 1")
    assert "inventory" not in obs2["text"].lower()
    assert "craft an item of type" in obs2["text"].lower()


def test_successful_completion(plancraft_examples):
    """Test successful task completion"""
    # Find a simple example that can be completed in a few steps
    example = next(
        ex for ex in plancraft_examples if not ex.impossible and ex.complexity == 1
    )

    env = PlancraftGymWrapper(example, max_steps=10)

    # Execute actions to complete the task (simplified for test)
    # This assumes we know the right sequence for this specific example
    obs = {
        "text": "Start",
        "inventory": example.slotted_inventory,
        "target": example.target,
    }
    subplans, _ = get_subplans(obs)
    actions = [action for plan in subplans for action in plan]

    terminated = False
    truncated = False

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # Check if task was successfully completed
    assert terminated
    assert not truncated
    assert reward == 1.0
    assert env.success
    assert info["reason"] == "success"


def test_environment_terminated_after_stop(plancraft_examples):
    """Test environment is terminated after stop action"""
    example = plancraft_examples[0]
    env = PlancraftGymWrapper(example, max_steps=10)

    # First do a valid action
    obs, reward, terminated, truncated, info = env.step(
        "move: from [I1] to [B1] with quantity 1"
    )
    assert not terminated
    assert not truncated

    # Then stop
    obs, reward, terminated, truncated, info = env.step(
        "impossible: This task cannot be completed"
    )
    assert terminated

    # Try another action after stopping
    obs, reward, terminated, truncated, info = env.step(
        "move: from [I2] to [B2] with quantity 1"
    )
    assert terminated
    # Should be truncated because the environment is already terminated
    assert truncated
    assert "terminated" in obs["text"].lower()
