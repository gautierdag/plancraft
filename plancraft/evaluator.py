import json
import os
from typing import Optional

import imageio
from loguru import logger
from tqdm import tqdm

import wandb
from plancraft.config import PlancraftExample
from plancraft.environment.actions import (
    ActionHandlerBase,
    MoveActionHandler,
    SmeltActionHandler,
    StopAction,
)
from plancraft.environment.env import (
    PlancraftEnvironment,
    get_objective_str,
    target_and_inventory_to_text_obs,
)
from plancraft.models.base import PlancraftBaseModel
from plancraft.utils import History


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
        run_name: str,
        model: PlancraftBaseModel,
        actions: list[ActionHandlerBase] = [MoveActionHandler(), SmeltActionHandler()],
        output_dir: str = "output",
        split: str = "val.small",
        resolution: str = "high",
        max_steps: int = 30,
        resume: bool = False,
        use_multimodal_content_format: bool = False,
        use_images: bool = False,
        use_text_inventory: bool = False,
        use_fasterrcnn: bool = False,
        system_prompt: Optional[dict] = None,
        prompt_examples: list[dict] = [],
        prompt_images: list[str] = [],
        few_shot: bool = True,
    ):
        self.run_name = run_name
        self.use_multimodal_content_format = use_multimodal_content_format
        self.use_images = use_images
        self.use_text_inventory = use_text_inventory
        self.use_fasterrcnn = use_fasterrcnn
        self.max_steps = max_steps
        self.resume = resume

        self.output_dir = f"{output_dir}/{run_name}/{split}"
        self.generation_number = 0
        self.actions = actions

        # load all examples
        self.examples: list[PlancraftExample] = self.load_dataset(split)

        # start environment
        self.environment = PlancraftEnvironment(
            inventory=[],
            resolution=resolution,
        )

        # initialise history/dialogue tracking
        self.history = History(
            actions=actions,
            use_multimodal_content_format=use_multimodal_content_format,
            use_images=use_images,
            use_text_inventory=use_text_inventory,
            resolution=resolution,
            few_shot=few_shot,
            system_prompt=system_prompt,
            prompt_examples=prompt_examples,
            prompt_images=prompt_images,
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
        if not os.path.exists(path) or not self.resume:
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

    def parse_raw_model_response(self, generated_text: str, observation=None) -> str:
        """
        Given a message and set of action handlers, parse the content to return the action
        or a message if the action is not valid/requires message response
        """
        for handler in self.actions:
            match_output = handler.match(
                generated_text, observation=observation, history=self.history
            )
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
        if self.use_fasterrcnn:
            # convert image to inventory using fasterrcnn
            inventory = self.model.bbox_model.get_inventory(observation["image"].copy())
            text_content = target_and_inventory_to_text_obs(
                observation["target"], inventory
            )
        elif not self.use_text_inventory:
            text_content = get_objective_str(observation["target"])
        else:
            # if not multimodal, we only have text - we format the inventory as text
            text_content = target_and_inventory_to_text_obs(
                observation["target"], observation["inventory"]
            )
        if not self.use_multimodal_content_format:
            return text_content

        content_list = [{"type": "text", "text": text_content}]
        if self.use_images:
            content_list.append({"type": "image"})
        return {"content": content_list}

    def eval_example(self, example: PlancraftExample) -> dict:
        """Given the loaded model and an example from Plancraft
        run the episode until success or termination."""
        success = False
        self.reset(example)
        action = None

        # run episode until stuck or until max steps is reached
        while self.history.num_steps < self.max_steps:
            # if the action is stop then we end the episode
            if isinstance(action, StopAction):
                # if the action is stop and task is impossible then success
                # otherwise we should not have stopped
                success = example.impossible
                break
            # action is external tool then it is str
            if isinstance(action, str):
                observation = self.environment.step()
                observation["target"] = example.target
                observation["message"] = action
            # action is environment action
            else:
                observation = self.environment.step(action)
                # convert inventory observation to text message
                observation["target"] = example.target
                observation["message"] = self.convert_observation_to_message(
                    observation
                )
                # check if the episode is done
                success = self.check_done(observation["inventory"], example.target)
            # exit if success
            if success:
                break

            # add observation to history
            self.history.add_observation_to_history(observation)
            # add observation message to history
            self.history.add_message_to_history(
                content=observation["message"], role="user"
            )
            # predict next action
            raw_action = self.model.step(observation, dialogue_history=self.history)
            # add message to history
            self.history.add_message_to_history(content=raw_action, role="assistant")
            # parse the raw action
            action = self.parse_raw_model_response(raw_action, observation=observation)

        # save results and reset
        return {
            "success": success,
            "recipe_type": example.recipe_type,
            "complexity": example.complexity_split,
            "number_of_steps": self.history.num_steps,
            "model_trace": self.history.trace(),
            "example_id": example.id,
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
                pbar.update(self.max_steps)
                results.append(resume_result)
                continue

            # skip impossible examples if impossible is not in valid actions
            if example.impossible and "impossible" not in [
                action.action_name for action in self.actions
            ]:
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
