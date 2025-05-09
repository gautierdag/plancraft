import json
import os
from typing import Optional
from copy import deepcopy
from collections import deque

import imageio
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
from plancraft.models.base import PlancraftBaseModel, PlancraftModelOutput
from plancraft.utils import HistoryBase, History, HistoryConfig


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment is created based on the configuration and the examples are loaded from the dataset.

    The Evaluator uses the dataset examples and initializes the environment with the example's inventory.

    It is also responsible for early stopping and verifying the target object has been craft.
    Finally, it also saves the results of the evaluation and the images generated during the evaluation.

    This evaluator is designed to work with a PlancraftBaseModel and a set of ActionHandlerBase instances.
    It supports multimodal content format and image-based inventory.

    Importantly, it tracks the history of the dialogue and the environment state to provide a trace of the model's actions.

    If you would want a simpler interface that just wraps the environment and actions to evaluate a single Plancraft example, you should use the PlancraftGymWrapper class.
    """

    def __init__(
        self,
        run_name: str,
        actions: list[ActionHandlerBase] = [MoveActionHandler(), SmeltActionHandler()],
        output_dir: str = "output",
        split: str = "val.small",
        max_steps: int = 30,
        resume: bool = False,
        use_fasterrcnn: bool = False,
        use_multimodal_content_format: bool = False,
        use_images: bool = False,
        use_text_inventory: bool = False,
        resolution: str = "high",
        history_config: Optional[HistoryConfig] = None,
        history_class: type[HistoryBase] = History,
    ):
        self.run_name = run_name
        self.actions = actions
        self.output_dir = f"{output_dir}/{run_name}/{split}"
        self.max_steps = max_steps
        self.resume = resume
        self.use_fasterrcnn = use_fasterrcnn
        self.generation_number = 0

        self.use_multimodal_content_format = use_multimodal_content_format
        self.use_images = use_images
        self.use_text_inventory = use_text_inventory
        self.resolution = resolution

        # Set up history configuration
        self.history_config = history_config or HistoryConfig()
        self.history_class = history_class

        # load examples
        self.examples: list[PlancraftExample] = self.load_dataset(split)

    def create_history(self) -> HistoryBase:
        """Create a new History instance with current configuration"""
        return self.history_class(
            actions=self.actions,
            config=self.history_config,
            resolution=self.resolution,
            use_multimodal_content_format=self.use_multimodal_content_format,
            use_images=self.use_images,
            use_text_inventory=self.use_text_inventory,
        )

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

    def check_done(self, inventory: dict, target: str):
        """
        Check that target object is obtained
        """
        for slot, item in inventory.items():
            # ensure the target is in the inventory (not in slot 0)
            if target == item["type"] and slot != 0:
                return True
        return False

    def parse_raw_model_response(
        self, generated_text: str, observation=None, history=None
    ) -> str:
        """
        Given a message and set of action handlers, parse the content to return the action
        or a message if the action is not valid/requires message response
        """
        for handler in self.actions:
            match_output = handler.match(
                generated_text, observation=observation, history=history
            )
            if match_output:
                return match_output
        action_names = [handler.action_name for handler in self.actions]
        return f"Only select actions from the following: {', '.join(action_names)}"

    def convert_observation_to_message(
        self,
        observation: dict,
        model: PlancraftBaseModel = None,
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
            assert model is not None, "Model must be provided to convert image to text"
            # convert image to inventory using fasterrcnn
            inventory = model.bbox_model.get_inventory(observation["image"].copy())
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

    def _init_environment(self, example: PlancraftExample, model=None) -> tuple:
        """Initialize environment and history for an example"""
        environment = PlancraftEnvironment(
            inventory=deepcopy(example.slotted_inventory),
            resolution=self.resolution,
        )
        history = self.create_history()
        obs = environment.step()
        obs["target"] = example.target
        obs["message"] = self.convert_observation_to_message(obs, model=model)
        return environment, history, obs

    def _process_model_output(
        self, raw_action, observation: dict, history: HistoryBase
    ) -> tuple:
        """Process model output and update history"""
        if isinstance(raw_action, PlancraftModelOutput):
            history.add_message_to_history(
                content=raw_action.action,
                role="assistant",
                **(raw_action.kwargs or {}),
            )
            raw_action = raw_action.action
        else:
            history.add_message_to_history(content=raw_action, role="assistant")

        action = self.parse_raw_model_response(
            raw_action,
            observation=observation,
            history=history,
        )
        return action

    def _execute_action(
        self, action, example: PlancraftExample, environment, model=None
    ) -> tuple[dict, bool]:
        """Execute action and return next observation and success status"""
        success = False

        # stop action
        if isinstance(action, StopAction):
            observation = None
            #  success is True if example was truly impossible
            success = example.impossible
        #  if action is a string, it is a message response
        elif isinstance(action, str):
            observation = environment.step()
            observation["target"] = example.target
            observation["message"] = action
        #  execute action and check if target is obtained
        else:
            observation = environment.step(action)
            observation["target"] = example.target
            observation["message"] = self.convert_observation_to_message(
                observation, model=model
            )
            success = self.check_done(observation["inventory"], example.target)

        return observation, success

    def _create_result(
        self, example: PlancraftExample, success: bool, history: HistoryBase
    ) -> dict:
        """Create result dictionary for an example"""
        return {
            "success": success,
            "recipe_type": example.recipe_type,
            "complexity": example.complexity_split,
            "number_of_steps": history.num_steps,
            "model_trace": history.trace(),
            "example_id": example.id,
            "images": history.trace_images(),
        }

    def eval_example(
        self,
        example: PlancraftExample,
        model: PlancraftBaseModel,
    ) -> dict:
        environment, history, observation = self._init_environment(example, model=model)
        success = False

        while history.num_steps < self.max_steps:
            history.add_observation_to_history(observation)
            history.add_message_to_history(content=observation["message"], role="user")

            raw_action = model.step(observation, dialogue_history=history)
            action = self._process_model_output(raw_action, observation, history)

            observation, success = self._execute_action(
                action, example, environment, model
            )
            if success or isinstance(action, StopAction):
                break

        return self._create_result(example, success, history)

    def batch_eval_examples(
        self,
        examples: list[PlancraftExample],
        model,
        batch_size: int = 4,
        callback_fn: Optional[callable] = None,
    ) -> list:
        pending_examples = deque(examples)
        active_examples = []
        active_environments = []
        active_histories = []
        active_observations = []
        results = {ex.id: None for ex in examples}

        # Initialize first batch
        while len(active_examples) < batch_size and pending_examples:
            example = pending_examples.popleft()
            env, history, obs = self._init_environment(example)

            active_examples.append(example)
            active_environments.append(env)
            active_histories.append(history)
            active_observations.append(obs)

        while active_examples:
            # Add observations to histories
            for i in range(len(active_examples)):
                active_histories[i].add_observation_to_history(active_observations[i])
                active_histories[i].add_message_to_history(
                    content=active_observations[i]["message"], role="user"
                )

            raw_actions = model.batch_step(
                active_observations, dialogue_histories=active_histories
            )

            completed_indices = []
            successes = []
            actions = []

            for i, (example, raw_action) in enumerate(
                zip(active_examples, raw_actions)
            ):
                action = self._process_model_output(
                    raw_action, active_observations[i], active_histories[i]
                )
                actions.append(action)

                obs, success = self._execute_action(
                    action, example, active_environments[i], model
                )
                active_observations[i] = obs
                successes.append(success)

                if (
                    success
                    or isinstance(action, StopAction)
                    or active_histories[i].num_steps >= self.max_steps
                ):
                    results[example.id] = self._create_result(
                        example, success, active_histories[i]
                    )
                    completed_indices.append(i)
                    if callback_fn:
                        callback_fn(example=example, results=results[example.id])

            # Remove completed environments and replace with new ones
            for i in reversed(completed_indices):
                active_examples.pop(i)
                active_environments.pop(i)
                active_histories.pop(i)
                active_observations.pop(i)

                if pending_examples:
                    example = pending_examples.popleft()
                    env, history, obs = self._init_environment(example)

                    active_examples.append(example)
                    active_environments.append(env)
                    active_histories.append(history)
                    active_observations.append(obs)

        return list(results.values())

    def eval_all_examples(self, model, progress_bar=False) -> list:
        results = []
        pbar = tqdm(
            total=len(self.examples),
            disable=not progress_bar,
        )
        correct = 0
        count = 0
        for example in self.examples:
            if resume_result := self.load_results_dict(example):
                pbar.update(self.max_steps)
                results.append(resume_result)
                continue

            # skip impossible examples if impossible is not in valid actions
            if example.impossible and "impossible" not in [
                action.action_name for action in self.actions
            ]:
                continue

            result = self.eval_example(example, model=model)
            model.reset()

            # save images and results
            self.save_images(example, result["images"])
            del result["images"]
            results.append(result)
            self.save_results_dict(example, result)

            correct += int(result["success"])
            count += 1

            acc = correct / count
            pbar.set_postfix(correct=correct, count=count, acc=acc)
            pbar.update(1)

        return results
