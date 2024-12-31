from unittest.mock import MagicMock, patch

import pytest

from plancraft.config import EvalConfig, PlancraftExample
from plancraft.environment.env import PlancraftEnvironment
from plancraft.evaluator import Evaluator


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
        "slotted_inventory": [
            {"slot": 45, "type": "cyan_stained_glass", "quantity": 6},
            {"slot": 20, "type": "purpur_stairs", "quantity": 26},
            {"slot": 12, "type": "birch_trapdoor", "quantity": 28},
            {"slot": 38, "type": "zombie_head", "quantity": 30},
            {"slot": 25, "type": "jungle_fence", "quantity": 8},
            {"slot": 39, "type": "acacia_slab", "quantity": 60},
            {"slot": 22, "type": "netherite_helmet", "quantity": 1},
            {"slot": 42, "type": "cooked_salmon", "quantity": 61},
            {"slot": 10, "type": "purple_terracotta", "quantity": 48},
            {"slot": 23, "type": "cod_bucket", "quantity": 1},
            {"slot": 19, "type": "ancient_debris", "quantity": 42},
            {"slot": 13, "type": "cobblestone_wall", "quantity": 60},
            {"slot": 16, "type": "magenta_bed", "quantity": 1},
            {"slot": 44, "type": "cat_spawn_egg", "quantity": 32},
            {"slot": 37, "type": "yellow_dye", "quantity": 6},
            {"slot": 24, "type": "chain", "quantity": 42},
            {"slot": 28, "type": "purple_concrete", "quantity": 22},
        ],
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
    with patch("plancraft.evaluator.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.history.trace.return_value = {"tokens_used": 10}
        mock_model.history.images = [MagicMock()]
        mock_get_model.return_value = mock_model
        with patch("plancraft.evaluator.Evaluator.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = [mock_example]
            return Evaluator(mock_cfg)


def test_load_dataset(evaluator):
    with pytest.raises(FileNotFoundError):
        evaluator.load_dataset("fake_split")


def test_reset(evaluator, mock_example):
    evaluator.reset(mock_example)
    assert evaluator.environment.inventory == mock_example.slotted_inventory


def test_check_done(evaluator):
    inventory = [{"type": "iron_ingot", "quantity": 1, "slot": 1}]
    assert evaluator.check_done(inventory, "iron_ingot")
    assert not evaluator.check_done(inventory, "diamond")


# def test_eval_all_examples(evaluator):
#     evaluator.save_images = MagicMock()
#     evaluator.save_results_dict = MagicMock()
#     results = evaluator.eval_all_examples()
#     assert len(results) == 1
#     assert results[0]["example_id"] == "TRAIN0000"
#     assert evaluator.save_images.call_count == 1
#     assert evaluator.save_results_dict.call_count == 1
