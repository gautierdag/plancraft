import json
import os
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from plancraft.config import EvalConfig, PlancraftExample
from plancraft.environment import (
    GoldSearchActionHandler,
    ImpossibleActionHandler,
    MoveActionHandler,
    SmeltActionHandler,
    ThinkActionHandler,
)
from plancraft.environment.actions import MoveAction, SmeltAction, StopAction
from plancraft.evaluator import Evaluator
from plancraft.models import get_model


@pytest.fixture
def mock_cfg():
    cfg = MagicMock(spec=EvalConfig)
    cfg.plancraft = MagicMock()
    cfg.plancraft.output_dir = "test_output"
    cfg.plancraft.split = "test_split"
    cfg.plancraft.environment = MagicMock()
    cfg.plancraft.environment.resolution = "high"
    cfg.plancraft.model = "test_model"
    cfg.plancraft.adapter = ""
    cfg.plancraft.mode = "test_mode"
    cfg.plancraft.valid_actions = ["move", "smelt"]
    cfg.plancraft.max_steps = 10
    cfg.plancraft.num_generations = 1
    cfg.plancraft.resume = False
    cfg.wandb = MagicMock()
    cfg.wandb.project = "test_project"
    cfg.wandb.entity = "test_entity"
    cfg.wandb.mode = "disabled"
    return cfg


@pytest.fixture
def mock_example_json():
    return {
        "inventory": {
            "cyan_stained_glass": 6,
            "purpur_stairs": 26,
            "birch_trapdoor": 28,
            "zombie_head": 30,
            "jungle_fence": 8,
            "acacia_slab": 60,
            "netherite_helmet": 1,
            "cooked_salmon": 61,
            "purple_terracotta": 48,
            "cod_bucket": 1,
            "ancient_debris": 42,
            "cobblestone_wall": 60,
            "magenta_bed": 1,
            "cat_spawn_egg": 32,
            "yellow_dye": 6,
            "chain": 42,
            "purple_concrete": 22,
        },
        "slotted_inventory": {
            45: {"type": "cyan_stained_glass", "quantity": 6},
            20: {"type": "purpur_stairs", "quantity": 26},
            12: {"type": "birch_trapdoor", "quantity": 28},
            38: {"type": "zombie_head", "quantity": 30},
            25: {"type": "jungle_fence", "quantity": 8},
            39: {"type": "acacia_slab", "quantity": 60},
            22: {"type": "netherite_helmet", "quantity": 1},
            42: {"type": "cooked_salmon", "quantity": 61},
            10: {"type": "purple_terracotta", "quantity": 48},
            23: {"type": "cod_bucket", "quantity": 1},
            19: {"type": "ancient_debris", "quantity": 42},
            13: {"type": "cobblestone_wall", "quantity": 60},
            16: {"type": "magenta_bed", "quantity": 1},
            44: {"type": "cat_spawn_egg", "quantity": 32},
            37: {"type": "yellow_dye", "quantity": 6},
            24: {"type": "chain", "quantity": 42},
            28: {"type": "purple_concrete", "quantity": 22},
        },
        "target": "cyan_stained_glass_pane",
        "num_distractors": 16,
        "impossible": False,
        "optimal_path_length": 1.0,
        "optimal_path": ["cyan_stained_glass_pane"],
        "inventory_trace": [
            {
                "purpur_stairs": 26,
                "birch_trapdoor": 28,
                "zombie_head": 30,
                "jungle_fence": 8,
                "acacia_slab": 60,
                "netherite_helmet": 1,
                "cooked_salmon": 61,
                "purple_terracotta": 48,
                "cod_bucket": 1,
                "ancient_debris": 42,
                "cobblestone_wall": 60,
                "magenta_bed": 1,
                "cat_spawn_egg": 32,
                "yellow_dye": 6,
                "chain": 42,
                "purple_concrete": 22,
                "cyan_stained_glass_pane": 16,
            }
        ],
        "items_used": 6.0,
        "unique_items_used": 1.0,
        "complexity": 6.0,
        "complexity_bin": 1.0,
        "complexity_split": "easy",
        "unseen_in_train": False,
        "unseen_in_val": False,
        "split": "train",
        "id": "TRAIN0000",
    }


@pytest.fixture
def mock_example(mock_example_json):
    return PlancraftExample(**mock_example_json)


@pytest.fixture
def evaluator(mock_cfg, mock_example):
    mock_model = MagicMock()
    mock_model.history.trace.return_value = {"tokens_used": 10}
    mock_model.history.images 

    with patch("plancraft.evaluator.Evaluator.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = [mock_example]
        return Evaluator(run_name="test_run")


def test_load_dataset(evaluator):
    with pytest.raises(FileNotFoundError):
        evaluator.load_dataset("fake_split")


def test_check_done(evaluator):
    inventory = {1: {"type": "iron_ingot", "quantity": 1}}
    assert evaluator.check_done(inventory, "iron_ingot")
    assert not evaluator.check_done(inventory, "diamond")


# Sample test function for parse_raw_model_response
def test_parse_raw_model_response(evaluator):
    evaluator.actions = [
        MoveActionHandler(),
        SmeltActionHandler(),
        ThinkActionHandler(),
        ImpossibleActionHandler(),
        GoldSearchActionHandler(),
    ]

    # Define example inputs and expected outputs
    content_move = "move: from [A1] to [B2] with quantity 10"
    content_move_format_err = "move: from AAA to [B2] with quantity 10"
    content_smelt = "smelt: from [A1] to [I1] with quantity 1"
    content_think = "think: some thought"
    content_impossible = "impossible: Cannot reach the target"
    content_search = "search: bucket"
    invalid_content = "dance: some random action"

    # Test case: move action
    result = evaluator.parse_raw_model_response(content_move)
    assert isinstance(result, MoveAction)
    assert result.slot_from == 1
    assert result.slot_to == 5
    assert result.quantity == 10

    # Test case: invalid format
    result = evaluator.parse_raw_model_response(content_move_format_err)
    assert "Format Error" in result

    # Test case: smelt action
    result = evaluator.parse_raw_model_response(content_smelt)
    assert isinstance(result, SmeltAction)
    assert result.slot_from == 1
    assert result.slot_to == 10
    assert result.quantity == 1

    # Test case: think action
    result = evaluator.parse_raw_model_response(content_think)
    assert result == "Ok"

    # Test case: impossible action
    result = evaluator.parse_raw_model_response(content_impossible)
    assert isinstance(result, StopAction)
    assert result.reason == "Cannot reach the target"

    # Test case: search action
    result = evaluator.parse_raw_model_response(content_search)
    assert "Recipes to craft bucket:\nrecipe 1:\niron_ingot at" in result

    # Test case: invalid action
    result = evaluator.parse_raw_model_response(invalid_content)
    assert "Only select actions from the following:" in result


def test_convert_observation_to_message(evaluator):
    target = "iron_ingot"
    inventory = {1: {"type": "iron_ingot", "quantity": 1}}
    observation = {
        "target": target,
        "inventory": inventory,
    }

    evaluator.use_fasterrcnn = False
    evaluator.use_text_inventory = False
    evaluator.use_multimodal_content_format = False

    message = evaluator.convert_observation_to_message(observation)
    assert message == "Craft an item of type: iron_ingot"

    evaluator.use_text_inventory = True
    message = evaluator.convert_observation_to_message(observation)
    assert (
        message
        == "Craft an item of type: iron_ingot\ninventory:\n - iron_ingot [A1] quantity 1"
    )

    evaluator.use_multimodal_content_format = True
    evaluator.use_images = False
    message = evaluator.convert_observation_to_message(observation)
    message = {
        "content": [
            {
                "type": "text",
                "text": "Craft an item of type: iron_ingot\ninventory:\n - iron_ingot [A1] quantity 1",
            }
        ],
    }

    evaluator.use_multimodal_content_format = True
    evaluator.use_images = True
    message = evaluator.convert_observation_to_message(observation)
    message = {
        "content": [
            {
                "type": "text",
                "text": "Craft an item of type: iron_ingot\ninventory:\n - iron_ingot [A1] quantity 1",
            },
            {"type": "image"},
        ],
    }


def test_dummy_model(mock_cfg, mock_example_json):
    mock_cfg.plancraft.mode = "dummy"
    mock_cfg.plancraft.use_fasterrcnn = False
    example = PlancraftExample(**mock_example_json)
    model = get_model(mock_cfg)
    with patch("plancraft.evaluator.Evaluator.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = [example]
        evaluator = Evaluator(run_name="test_run")
        result = evaluator.eval_example(example, model=model)
        assert result["example_id"] == "TRAIN0000"
        assert result["model_trace"]["tokens_used"] == 0
        assert not result["success"]


def test_oracle_model(mock_cfg, mock_example_json):
    mock_cfg.plancraft.mode = "oracle"
    mock_cfg.plancraft.use_fasterrcnn = False
    example = PlancraftExample(**mock_example_json)
    model = get_model(mock_cfg)
    with patch("plancraft.evaluator.Evaluator.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = [example]
        evaluator = Evaluator(run_name="test_run")
        result = evaluator.eval_example(example, model=model)
        assert result["example_id"] == "TRAIN0000"
        assert result["model_trace"]["tokens_used"] == 0
        assert result["success"]


def test_batch_eval_examples(mock_cfg, mock_example_json):
    mock_cfg.plancraft.mode = "dummy"
    mock_cfg.plancraft.use_fasterrcnn = False

    # Create multiple examples
    examples = [PlancraftExample(**mock_example_json) for _ in range(3)]

    model = get_model(mock_cfg)
    with patch("plancraft.evaluator.Evaluator.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = examples
        evaluator = Evaluator(run_name="test_run")

        # Test batch evaluation
        batch_results = evaluator.batch_eval_examples(
            examples,
            model,
        )

        # Test individual evaluation for comparison
        individual_results = [
            evaluator.eval_example(ex, model=model) for ex in examples
        ]

        # Verify batch results
        assert len(batch_results) == len(examples)
        for result in batch_results:
            assert result["example_id"] == "TRAIN0000"
            assert result["model_trace"]["tokens_used"] == 0
            assert "success" in result
            assert "number_of_steps" in result
            assert "images" in result

        # Verify batch results match structure of individual results
        assert len(batch_results) == len(individual_results)
        for batch_result, individual_result in zip(batch_results, individual_results):
            assert batch_result.keys() == individual_result.keys()


def test_batch_oracle_model(mock_cfg, mock_example_json):
    mock_cfg.plancraft.mode = "oracle"
    mock_cfg.plancraft.use_fasterrcnn = False

    # Create multiple examples - using same example since we know it works
    model = get_model(mock_cfg)
    with patch("plancraft.evaluator.Evaluator.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = []
        evaluator = Evaluator(run_name="test_run")
        # Test batch evaluation
        examples = [PlancraftExample(**mock_example_json) for _ in range(2)]
        batch_results = evaluator.batch_eval_examples(
            examples,
            model=model,
        )
        # Verify batch results
        assert len(batch_results) == len(examples)
        for result in batch_results:
            assert result["example_id"] == "TRAIN0000"
            assert result["model_trace"]["tokens_used"] == 0
            assert result["success"]  # Oracle should succeed

        # Test individual evaluation for comparison
        individual_results = []
        # examples = [PlancraftExample(**mock_example_json) for _ in range(1)]
        for ex in examples:
            model.reset()
            result = evaluator.eval_example(ex, model=model)
            assert result["example_id"] == "TRAIN0000"
            assert result["success"]
            individual_results.append(result)

        # # Since oracle is deterministic, batch and individual should match exactly
        for batch_result, individual_result in zip(batch_results, individual_results):
            assert batch_result["success"] == individual_result["success"]
            assert (
                batch_result["number_of_steps"] == individual_result["number_of_steps"]
            )
            assert batch_result["model_trace"] == individual_result["model_trace"]


def test_real_val_example():
    """Test a real validation example with Oracle model to verify end-to-end success"""
    # Setup minimal configuration for Oracle model
    cfg = MagicMock(spec=EvalConfig)
    cfg.plancraft = MagicMock()
    cfg.plancraft.mode = "oracle"
    cfg.plancraft.use_fasterrcnn = False
    cfg.plancraft.valid_actions = ["move", "smelt", "think", "impossible"]
    cfg.plancraft.model = "test_model"

    # Setup evaluator and model
    model = get_model(cfg)
    evaluator = Evaluator(
        run_name="test_run",
        max_steps=30,
        use_multimodal_content_format=False,
        use_images=False,
        use_text_inventory=True,
        split="val",
    )
    example = list(filter(lambda x: x.id == "VAL0439", evaluator.examples))[0]
    # Run evaluation
    result = evaluator.eval_example(example, model=model)

    # Verify results
    assert result["success"], "Oracle model should succeed on validation example"
    assert result["example_id"].startswith("VAL")
    assert "number_of_steps" in result
    assert result["number_of_steps"] > 0
