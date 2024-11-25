import json
import os
import random
import string
import time

import imageio
import pandas as pd
import torch
import wandb
from loguru import logger
from tqdm import tqdm

from plancraft.config import EvalConfig, PlancraftExample
from plancraft.environments.actions import StopAction
from plancraft.environments.env import PlancraftEnv
from plancraft.models import get_model

wandb.require("core")


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment is created based on the configuration and the examples are loaded from the dataset.
    """

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.output_dir = (
            f"{cfg.plancraft.output_dir}/{self.evaluator_name()}/{cfg.plancraft.split}"
        )
        self.generation_number = 0

        self.examples: list[PlancraftExample] = self.load_dataset(cfg.plancraft.split)

        self.environment = self.create_env(cfg)
        self.model = get_model(cfg)

        self.record_frames = not (cfg.plancraft.environment.symbolic)

    def evaluator_name(self) -> str:
        symb_str = "real"
        if self.cfg.plancraft.environment.symbolic:
            symb_str = "symb"

        if self.cfg.plancraft.use_maskrcnn:
            symb_str += "_mrcnn"

        model_name = self.cfg.plancraft.model.split("/")[-1]
        if self.cfg.plancraft.adapter != "":
            model_name = self.cfg.plancraft.adapter.split("/")[-1]

        mode = self.cfg.plancraft.mode
        if mode in ["dummy", "oracle"]:
            return f"{mode}_{symb_str}"

        actions = "|".join(self.cfg.plancraft.valid_actions)
        return f"{self.cfg.plancraft.mode}_{symb_str}_{model_name}_{actions}"

    def save_results_dict(self, example: PlancraftExample, results_dict: dict):
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        json_path = f"{output_dir}/{example.id}.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        wandb.save(json_path, policy="now")

    def save_images(self, example: PlancraftExample, frames: list):
        if len(frames) == 0:
            return
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(f"{output_dir}/{example.id}.gif", frames)
        # upload to wandb
        wandb.save(f"{output_dir}/{example.id}.gif", policy="now")

    def load_results_dict(self, example: PlancraftExample) -> dict:
        path = f"{self.output_dir}/{self.generation_number}/{example.id}.json"
        if not os.path.exists(path) or not self.cfg.plancraft.resume:
            return None
        with open(path, "r") as f:
            return json.load(f)

    def create_env(self, cfg: EvalConfig) -> PlancraftEnv:
        return PlancraftEnv(
            inventory=[],
            resolution=cfg.plancraft.environment.resolution,
        )

    def load_dataset(self, dataset_split: str) -> list[PlancraftExample]:
        with open(f"data/{dataset_split}.json", "r") as f:
            dataset = json.load(f)
            return [PlancraftExample(**example) for example in dataset]

    def reset(
        self,
        example: PlancraftExample,
    ):
        current_inventory = example.slotted_inventory
        self.environment.fast_reset(new_inventory=current_inventory)
        # do a no op to an initial observation
        obs = self.environment.step()
        # assert that the inventory is correct
        if "inventory" in obs:
            for item in current_inventory:
                slot = item["slot"]
                if (
                    obs["inventory"][slot]["type"] != item["type"]
                    or obs["inventory"][slot]["quantity"] != item["quantity"]
                ) and item["type"] != "air":
                    logger.warning(f"Inventory does not match expected for slot {slot}")
                    logger.warning(f"Expected {item}")
                    logger.warning(f"Got {obs['inventory'][slot]}")
                    # try again
                    self.reset(example)

        objective = f"Craft an item of type: {example.target}"
        self.model.reset_history(objective=objective)

    def check_done(self, inventory: list[dict[str, int]], target: str):
        """
        Check that target object is obtained
        """
        for item in inventory:
            if target == item["type"]:
                # ensure item is taken out of crafting slot
                if "slot" in item and item["slot"] != 0:
                    return True
                if "index" in item and item["index"] != 0:
                    return True
        return False

    @torch.no_grad()
    def eval_all_examples(self, progress_bar=False) -> list:
        results = []
        pbar = tqdm(
            total=len(self.examples),
            disable=not progress_bar,
        )
        correct = 0
        count = 0

        for example in self.examples:
            if resume_result := self.load_results_dict(example):
                pbar.update(self.cfg.plancraft.max_steps)
                results.append(resume_result)
                continue

            success = False

            self.reset(example)
            action = None

            while (
                not self.model.history.check_stuck()
                and self.model.history.num_steps < self.cfg.plancraft.max_steps
            ):
                # if the action is stop then we end the episode
                if isinstance(action, StopAction):
                    # if the action is stop and task is impossible then success
                    # otherwise we should not have stopped
                    success = example.impossible
                    break

                # step action
                observation = self.environment.step(action)

                # check if the episode is done
                success = self.check_done(observation["inventory"], example.target)
                # exit if success
                if success:
                    break

                # predict next action
                action = self.model.step(observation)

            # save results and reset
            result = {
                "success": success,
                "recipe_type": example.recipe_type,
                "number_of_steps": self.model.history.num_steps,
                "model_trace": self.model.history.trace(),
                "example_id": example.id,
                "impossible": example.impossible,
            }
            results.append(result)
            self.save_results_dict(example, result)
            self.save_images(example, self.model.history.images)

            correct += int(result["success"])
            count += 1

            acc = correct / count
            pbar.set_postfix(correct=correct, count=count, acc=acc)
            pbar.update(1)

        return results

    def eval_all(self):
        logger.info(
            f"Running evaluation over {len(self.examples)} examples {self.cfg.plancraft.num_generations} times."
        )
        run_name = (
            f"{self.evaluator_name()} {self.cfg.plancraft.split}".replace(" ", "_")
            .replace(".", "_")
            .strip()
        )

        for n in range(self.cfg.plancraft.num_generations):
            logger.info(f"Generation {n+1}/{self.cfg.plancraft.num_generations}")
            run_id = "".join(random.choices(string.ascii_lowercase, k=5))
            generation_run_name = run_name + f"_{run_id}"

            wandb.init(
                name=generation_run_name,
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                mode=self.cfg.wandb.mode,
                group=self.cfg.plancraft.model,
                job_type=self.cfg.plancraft.mode,
                config=self.cfg.model_dump(),
            )
            time_now = time.time()

            results_list = self.eval_all_examples(progress_bar=True)

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

            time_elapsed = time.time() - time_now
            logger.info(f"Time elapsed: {time_elapsed:.2f}s")

            logger.info(output)
            wandb.log(output)
            table = wandb.Table(
                dataframe=results_df[["success", "number_of_steps", "example_id"]]
            )
            wandb.log({"results": table})
            wandb.finish()

            self.generation_number += 1

        logger.info("Done")
