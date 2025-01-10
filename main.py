import warnings

import hydra
from loguru import logger

from plancraft.config import EvalConfig
from plancraft.environment import (
    GoldSearchActionHandler,
    ImpossibleActionHandler,
    MoveActionHandler,
    SmeltActionHandler,
    ThinkActionHandler,
)
from plancraft.evaluator import Evaluator
from plancraft.models import get_model

warnings.filterwarnings("ignore")


def flatten_cfg(cfg):
    # for some reason hydra wraps file paths from config path
    if len(cfg) == 1:
        return flatten_cfg(cfg[list(cfg.keys())[0]])
    return cfg


def evaluator_name(cfg: EvalConfig) -> str:
    if cfg.plancraft.use_text_inventory and cfg.plancraft.use_images:
        name_str = "both"
    elif cfg.plancraft.use_images:
        name_str = "images"
    elif cfg.plancraft.use_text_inventory:
        name_str = "text"
    else:
        raise ValueError(
            "At least one of use_text_inventory or use_images should be True"
        )

    if cfg.plancraft.use_fasterrcnn:
        name_str += "_fasterrcnn"

    model_name = cfg.plancraft.model.split("/")[-1]
    if cfg.plancraft.adapter != "":
        model_name = cfg.plancraft.adapter.split("/")[-1]

    mode = cfg.plancraft.mode
    if mode in ["dummy", "oracle"]:
        return f"{mode}_{name_str}"

    valid_actions_to_str = {
        "move": "m",
        "smelt": "s",
        "think": "t",
        "search": "se",
        "impossible": "i",
    }
    actions = "|".join(
        [valid_actions_to_str[action] for action in cfg.plancraft.valid_actions]
    )
    return f"{cfg.plancraft.mode}_{name_str}_{model_name}_{actions}"


@hydra.main(config_path="configs", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = EvalConfig(**flatten_cfg(dict(cfg)))
    model = get_model(cfg)
    run_name = evaluator_name(cfg)

    action_handlers = []
    for action_name in cfg.plancraft.valid_actions:
        if action_name == "move":
            action_handlers.append(MoveActionHandler())
        elif action_name == "smelt":
            action_handlers.append(SmeltActionHandler())
        elif action_name == "think":
            action_handlers.append(ThinkActionHandler())
        elif action_name == "search":
            action_handlers.append(GoldSearchActionHandler())
        elif action_name == "impossible":
            action_handlers.append(ImpossibleActionHandler())

    evaluator = Evaluator(
        cfg=cfg, run_name=run_name, model=model, actions=action_handlers
    )
    evaluator.eval_all_seeds()


if __name__ == "__main__":
    main()
