import os
import random
import string
import time
import warnings

import hydra
import pandas as pd
from loguru import logger

import wandb
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


def eval_all_seeds(
    run_name: str,
    cfg: EvalConfig,
    evaluator: Evaluator,
):
    num_seeds = cfg.plancraft.num_generations
    logger.info(
        f"Running evaluation over {len(evaluator.examples)} examples {num_seeds} times."
    )
    wandb_run_name = (
        f"{run_name} {cfg.plancraft.split}".replace(" ", "_").replace(".", "_").strip()
    )
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    for n in range(num_seeds):
        logger.info(f"Generation {n+1}/{num_seeds}")
        run_id = "".join(random.choices(string.ascii_lowercase, k=5))
        generation_run_name = wandb_run_name + f"_{run_id}"
        wandb.init(
            name=generation_run_name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            group=cfg.plancraft.model,
            job_type=cfg.plancraft.mode,
            config=cfg.model_dump(),
        )
        time_now = time.time()

        results_list = evaluator.eval_all_examples(progress_bar=True)
        results_df = pd.DataFrame(results_list)

        output = {
            "avg_success_rate": results_df["success"].mean(),
            "avg_number_of_steps": results_df["number_of_steps"].mean(),
            "avg_num_tokens_used": results_df["model_trace"]
            .apply(pd.Series)["tokens_used"]
            .mean(),
        }

        # calculate success rate for each recipe type
        recipe_types = results_df["recipe_type"].unique()
        for recipe_type in recipe_types:
            mask = results_df["recipe_type"] == recipe_type
            success_rate = results_df[mask]["success"].mean()
            output[f"{recipe_type}_success_rate"] = success_rate

        # calculate success rate for each complexity (easy, medium, hard, impossible)
        for complexity in results_df["complexity"].unique():
            mask = results_df["complexity"] == complexity
            success_rate = results_df[mask]["success"].mean()
            output[f"{complexity}_success_rate"] = success_rate

        time_elapsed = time.time() - time_now
        logger.info(f"Time elapsed: {time_elapsed:.2f}s")

        logger.info(output)
        if wandb.run is not None:
            wandb.log(output)
            table = wandb.Table(
                dataframe=results_df[["success", "number_of_steps", "example_id"]]
            )
            wandb.log({"results": table})
            wandb.finish()

        evaluator.generation_number += 1

    logger.info("Done")


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
        run_name=run_name,
        model=model,
        actions=action_handlers,
        output_dir=cfg.plancraft.output_dir,
        split=cfg.plancraft.split,
        max_steps=cfg.plancraft.max_steps,
        resume=cfg.plancraft.resume,
        use_multimodal_content_format=cfg.plancraft.use_multimodal_content_format,
        use_images=cfg.plancraft.use_images,
        use_text_inventory=cfg.plancraft.use_text_inventory,
        use_fasterrcnn=cfg.plancraft.use_fasterrcnn,
        resolution=cfg.plancraft.environment.resolution,
    )
    eval_all_seeds(run_name, cfg, evaluator)


if __name__ == "__main__":
    main()
