import pytest

from plancraft.simple import EnvWrapper, get_plancraft_examples
from plancraft.environment.actions import (
    MoveActionHandler,
    SmeltActionHandler,
    ImpossibleActionHandler,
    ThinkActionHandler,
)


@pytest.fixture
def plancraft_examples():
    return get_plancraft_examples()


def test_all_examples_loading(plancraft_examples):
    """Test loading examples from data directory"""
    for mock_example in plancraft_examples[:5]:
        EnvWrapper(mock_example, max_steps=30)


def test_max_steps_actions(plancraft_examples):
    example = plancraft_examples[0]

    env = EnvWrapper(example, max_steps=5)

    action = "move: from [I18] to [B1] with quantity 1"
    for i in range(5):
        obs, reward, done = env.step(action)
        assert not done
        assert "image" in obs
        assert reward == 0
    action = "move: from [I18] to [B1] with quantity 1"
    obs, reward, done = env.step(action)
    assert "Max steps" in obs["text"]
    assert reward == 0
    assert done


def test_impossible_task(plancraft_examples):
    """Test handling of impossible tasks"""
    # Find an impossible example
    example = None
    for ex in plancraft_examples:
        if ex.impossible:
            example = ex
            break

    assert example is not None, "No impossible example found for testing"

    env = EnvWrapper(example, max_steps=10)

    # Try a few actions and then declare impossible
    action = "move: from [I1] to [B1] with quantity 1"
    obs, reward, done = env.step(action)
    assert not done

    # Declare impossible
    action = "impossible: Cannot craft the target with available materials"
    obs, reward, done = env.step(action)
    assert done
    assert reward == 1.0
    assert env.success


def test_invalid_action_handling(plancraft_examples):
    """Test handling of invalid actions"""
    example = plancraft_examples[0]
    env = EnvWrapper(example, max_steps=10)

    # Try an invalid action format
    action = "invalid action format"
    obs, reward, done = env.step(action)
    assert not done
    assert reward == 0
    assert "Only select actions from the following" in obs["text"]

    # Try an action with invalid slots
    action = "move: from [Z9] to [B1] with quantity 1"
    obs, reward, done = env.step(action)
    assert not done
    assert reward == 0
    assert "Format Error" in obs["text"]


def test_custom_action_handlers(plancraft_examples):
    """Test EnvWrapper with custom action handlers"""
    example = plancraft_examples[0]

    # Add ThinkActionHandler to the default handlers
    env = EnvWrapper(
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
    obs, reward, done = env.step(action)
    assert not done
    assert reward == 0
    assert obs["text"] == "Ok"


def test_use_text_inventory_parameter(plancraft_examples):
    """Test the use_text_inventory parameter"""
    example = plancraft_examples[0]

    # Test with text inventory enabled (default)
    env1 = EnvWrapper(example, max_steps=5, use_text_inventory=True)
    obs1, _, _ = env1.step("move: from [I1] to [B1] with quantity 1")
    assert "inventory" in obs1["text"].lower()

    # Test with text inventory disabled
    env2 = EnvWrapper(example, max_steps=5, use_text_inventory=False)
    obs2, _, _ = env2.step("move: from [I1] to [B1] with quantity 1")
    assert "inventory" not in obs2["text"].lower()
    assert "craft an item of type" in obs2["text"].lower()
