import json
import os
import random
import string
import time

import imageio
import pandas as pd
from loguru import logger
from tqdm import tqdm

import wandb
from plancraft.config import EvalConfig, PlancraftExample
from plancraft.environment.actions import (
    StopAction,
    ActionHandlerBase,
    MoveActionHandler,
    SmeltActionHandler,
)
from plancraft.environment.env import (
    PlancraftEnvironment,
    get_objective_str,
    target_and_inventory_to_text_obs,
)
from plancraft.utils import History
from plancraft.models.base import PlancraftBaseModel


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment is created based on the configuration and the examples are loaded from the dataset.

    The Evaluator uses the dataset examples and initializes the environment with the example's inventory.

    It is also responsible for early stopping and verifying the target object has been craft.
    Finally, it also saves the results of the evaluation and the images generated during the evaluation.
    """

    def __init__(
        self,
        cfg: EvalConfig,
        run_name: str,
        model: PlancraftBaseModel,
        actions: list[ActionHandlerBase] = [MoveActionHandler(), SmeltActionHandler()],
    ):
        self.cfg = cfg
        self.run_name = run_name
        self.output_dir = f"{cfg.plancraft.output_dir}/{run_name}/{cfg.plancraft.split}"
        self.generation_number = 0
        self.actions = actions

        # load all examples
        self.examples: list[PlancraftExample] = self.load_dataset(cfg.plancraft.split)

        # start environment
        self.environment = PlancraftEnvironment(
            inventory=[],
            resolution=cfg.plancraft.environment.resolution,
        )

        # initialise history/dialogue tracking
        self.history = History(
            actions=actions,
            use_multimodal_content_format=cfg.plancraft.use_multimodal_content_format,
            use_images=cfg.plancraft.use_images,
            use_text_inventory=cfg.plancraft.use_text_inventory,
            resolution=cfg.plancraft.environment.resolution,
        )

        # load model
        self.model = model

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
        folder = os.path.dirname(os.path.abspath(__file__))
        with open(f"{folder}/data/{dataset_split}.json", "r") as f:
            dataset = json.load(f)
            return [PlancraftExample(**example) for example in dataset]

    def reset(
        self,
        example: PlancraftExample,
    ):
        self.environment.reset(new_inventory=example.slotted_inventory)
        self.model.reset()
        self.history.reset()

    def check_done(self, inventory: dict, target: str):
        """
        Check that target object is obtained
        """
        for slot, item in inventory.items():
            # ensure the target is in the inventory (not in slot 0)
            if target == item["type"] and slot != 0:
                return True
        return False

    def parse_raw_model_response(self, generated_text: str):
        """
        Given a message and set of action handlers, parse the content to return the action
        or a message if the action is not valid/requires message response
        """
        for handler in self.actions:
            match_output = handler.match(generated_text)
            if match_output:
                return match_output
        action_names = [handler.action_name for handler in self.actions]
        return f"Only select actions from the following: {', '.join(action_names)}"

    def convert_observation_to_message(
        self,
        observation: dict,
    ) -> str | dict:
        """
        Convert an environment observation to the message format used by an LLM chat model

        Parameters:
        - observation: dict - The observation to convert.
        - use_text_inventory: bool - Whether to use text inventory.
        - use_multimodal_content_format: bool - Whether to use multimodal content format.
        - use_images: bool - Whether to append an image to the message content - must be used with use_multimodal_content_format.
        """
        if self.cfg.plancraft.use_fasterrcnn:
            # convert image to inventory using fasterrcnn
            inventory = self.model.bbox_model.get_inventory(observation["image"].copy())
            text_content = target_and_inventory_to_text_obs(
                observation["target"], inventory
            )
        elif not self.cfg.plancraft.use_text_inventory:
            text_content = get_objective_str(observation["target"])
        else:
            # if not multimodal, we only have text - we format the inventory as text
            text_content = target_and_inventory_to_text_obs(
                observation["target"], observation["inventory"]
            )
        if not self.cfg.plancraft.use_multimodal_content_format:
            return text_content

        content_list = [{"type": "text", "text": text_content}]
        if self.cfg.plancraft.use_images:
            content_list.append({"type": "image"})
        return {"content": content_list}

    def eval_example(self, example: PlancraftExample) -> dict:
        """Given the loaded model and an example from Plancraft
        run the episode until success or termination."""
        success = False
        num_non_env_actions = 0
        self.reset(example)
        action = None

        # run episode until stuck or until max steps is reached
        while (
            not self.history.check_stuck()
            and self.history.num_steps < self.cfg.plancraft.max_steps
        ):
            # if the action is stop then we end the episode
            if isinstance(action, StopAction):
                # if the action is stop and task is impossible then success
                # otherwise we should not have stopped
                success = example.impossible
                break
            # action is external tool then it is str
            # limit the number of consecutive non-env actions to 3
            elif isinstance(action, str) and num_non_env_actions < 3:
                observation = {"message": action}
                num_non_env_actions += 1
            # action is environment action
            else:
                # add action to history
                # TODO: fix the next two lines being triggered with a str response if num_non_env_actions >= 3
                if isinstance(action, str):
                    observation = self.environment.step()
                else:
                    self.history.add_action_to_history(action)
                    observation = self.environment.step(action)

                # convert inventory observation to text message
                observation["target"] = example.target
                observation["message"] = self.convert_observation_to_message(
                    observation
                )
                num_non_env_actions = 0

                # check if the episode is done
                success = self.check_done(observation["inventory"], example.target)

            # add observation to history
            self.history.add_observation_to_history(observation)
            # add observation message to history
            self.history.add_message_to_history(
                content=observation["message"], role="user"
            )

            # exit if success
            if success:
                break

            # predict next action
            raw_action = self.model.step(observation, dialogue_history=self.history)
            # add message to history
            self.history.add_message_to_history(content=raw_action, role="assistant")
            # parse the raw action
            action = self.parse_raw_model_response(raw_action)

        # save results and reset
        return {
            "success": success,
            "recipe_type": example.recipe_type,
            "complexity": example.complexity,
            "number_of_steps": self.history.num_steps,
            "model_trace": self.history.trace(),
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
            logger.debug(f"Running example {example.id}")

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
            self.save_images(example, self.history.images)

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
            f"{self.run_name} {self.cfg.plancraft.split}".replace(" ", "_")
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
