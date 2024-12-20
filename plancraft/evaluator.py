import json
import os
import random
import string
import time

import imageio
import pandas as pd
import wandb
from loguru import logger
from tqdm import tqdm

from plancraft.config import EvalConfig, PlancraftExample
from plancraft.environments.actions import StopAction
from plancraft.environments.env import PlancraftEnvironment
from plancraft.models import get_model


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment is created based on the configuration and the examples are loaded from the dataset.

    The Evaluator uses the dataset examples and initializes the environment with the example's inventory.

    It is also responsible for early stopping and verifying the target object has been craft.
    Finally, it also saves the results of the evaluation and the images generated during the evaluation.
    """

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.output_dir = (
            f"{cfg.plancraft.output_dir}/{self.evaluator_name()}/{cfg.plancraft.split}"
        )
        self.generation_number = 0

        self.examples: list[PlancraftExample] = self.load_dataset(cfg.plancraft.split)

        self.environment = PlancraftEnvironment(
            inventory=[],
            resolution=cfg.plancraft.environment.resolution,
        )

        self.model = get_model(cfg)

    def evaluator_name(self) -> str:
        if self.cfg.plancraft.use_text_inventory and self.cfg.plancraft.use_images:
            name_str = "both"
        elif self.cfg.plancraft.use_images:
            name_str = "images"
        elif self.cfg.plancraft.use_text_inventory:
            name_str = "text"
        else:
            raise ValueError(
                "At least one of use_text_inventory or use_images should be True"
            )

        if self.cfg.plancraft.use_fasterrcnn:
            name_str += "_fasterrcnn"

        model_name = self.cfg.plancraft.model.split("/")[-1]
        if self.cfg.plancraft.adapter != "":
            model_name = self.cfg.plancraft.adapter.split("/")[-1]

        mode = self.cfg.plancraft.mode
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
            [
                valid_actions_to_str[action]
                for action in self.cfg.plancraft.valid_actions
            ]
        )
        return f"{self.cfg.plancraft.mode}_{name_str}_{model_name}_{actions}"

    def save_results_dict(self, example: PlancraftExample, results_dict: dict):
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        json_path = f"{output_dir}/{example.id}.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)

        if wandb.run is not None:
            wandb.save(json_path, policy="now")

    def save_images(self, example: PlancraftExample, frames: list):
        if len(frames) == 0:
            return
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(f"{output_dir}/{example.id}.gif", frames)
        # upload to wandb
        if wandb.run is not None:
            wandb.save(f"{output_dir}/{example.id}.gif", policy="now")

    def load_results_dict(self, example: PlancraftExample) -> dict:
        path = f"{self.output_dir}/{self.generation_number}/{example.id}.json"
        if not os.path.exists(path) or not self.cfg.plancraft.resume:
            return None
        with open(path, "r") as f:
            return json.load(f)

    def load_dataset(self, dataset_split: str) -> list[PlancraftExample]:
        with open(f"data/{dataset_split}.json", "r") as f:
            dataset = json.load(f)
            return [PlancraftExample(**example) for example in dataset]

    def reset(
        self,
        example: PlancraftExample,
    ):
        objective = f"Craft an item of type: {example.target}"
        self.environment.reset(new_inventory=example.slotted_inventory)
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
        return False

    def eval_example(self, example: PlancraftExample) -> dict:
        success = False
        self.reset(example)
        action = None

        # run episode until stuck or until max steps is reached
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
                # add final observation to history
                self.model.history.add_observation_to_history(observation)
                break

            # predict next action
            action = self.model.step(observation)

        # save results and reset
        return {
            "success": success,
            "recipe_type": example.recipe_type,
            "complexity": example.complexity,
            "number_of_steps": self.model.history.num_steps,
            "model_trace": self.model.history.trace(),
            "example_id": example.id,
            "impossible": example.impossible,
        }

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
            # skip impossible examples if impossible is not in valid actions
            if (
                example.impossible
                and "impossible" not in self.cfg.plancraft.valid_actions
            ):
                continue

            result = self.eval_example(example)
            results.append(result)
            self.save_results_dict(example, result)
            self.save_images(example, self.model.history.images)

            correct += int(result["success"])
            count += 1

            acc = correct / count
            pbar.set_postfix(correct=correct, count=count, acc=acc)
            pbar.update(1)

        return results

    def eval_all_seeds(self):
        logger.info(
            f"Running evaluation over {len(self.examples)} examples {self.cfg.plancraft.num_generations} times."
        )
        run_name = (
            f"{self.evaluator_name()} {self.cfg.plancraft.split}".replace(" ", "_")
            .replace(".", "_")
            .strip()
        )

        wandb.login(key=self.cfg.env_variables.wandb_api_key)
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

            # calculate success rate for each complexity bin
            complexity_mapping = {
                0: "easy",
                1: "easy",
                2: "medium",
                3: "hard",
                4: "hard",
                5: "impossible",
            }
            results_df["complexity_bin"] = results_df["complexity"].map(
                complexity_mapping
            )
            # when impossible is True set complexity_bin to impossible
            results_df["complexity_bin"] = results_df["complexity_bin"].fillna(
                "impossible"
            )
            for complexity_bin in results_df["complexity_bin"].unique():
                mask = results_df["complexity_bin"] == complexity_bin
                success_rate = results_df[mask]["success"].mean()
                output[f"{complexity_bin}_success_rate"] = success_rate

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

            self.generation_number += 1

        logger.info("Done")
