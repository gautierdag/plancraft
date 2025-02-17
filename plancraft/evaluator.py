import json
import os
from typing import Optional
from copy import deepcopy

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
        return self.history_class(actions=self.actions, config=self.history_config)

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

    def eval_example(
        self,
        example: PlancraftExample,
        model: PlancraftBaseModel,
    ) -> dict:
        """
        Given the loaded model and an example from Plancraft
        run the episode until success or termination.
        """

        # start environment
        environment = PlancraftEnvironment(
            inventory=deepcopy(example.slotted_inventory),
            resolution=self.resolution,
        )

        # initialise history/dialogue tracking
        history = self.create_history()

        success = False
        action = None

        # run episode until stuck or until max steps is reached
        while history.num_steps < self.max_steps:
            # if the action is stop then we end the episode
            if isinstance(action, StopAction):
                # if the action is stop and task is impossible then success
                # otherwise we should not have stopped
                success = example.impossible
                break
            # action is external tool then it is str
            if isinstance(action, str):
                observation = environment.step()
                observation["target"] = example.target
                observation["message"] = action
            # action is environment action
            else:
                observation = environment.step(action)
                # convert inventory observation to text message
                observation["target"] = example.target
                observation["message"] = self.convert_observation_to_message(
                    observation, model=model
                )
                # check if the episode is done
                success = self.check_done(observation["inventory"], example.target)
            # exit if success
            if success:
                break

            # add observation to history
            history.add_observation_to_history(observation)
            # add observation message to history
            history.add_message_to_history(content=observation["message"], role="user")
            # predict next action
            raw_action = model.step(observation, dialogue_history=history)

            # if the model returns a PlancraftModelOutput, extract the action
            if isinstance(raw_action, PlancraftModelOutput):
                # add message to history
                history.add_message_to_history(
                    content=raw_action.action,
                    role="assistant",
                    **(raw_action.kwargs or {}),
                )
                raw_action = raw_action.action
            elif isinstance(raw_action, str):
                # add message to history
                history.add_message_to_history(content=raw_action, role="assistant")
            else:
                raise ValueError(
                    f"model.step() output must be a string or PlancraftModelOutput, got {type(raw_action)}"
                )

            # parse the raw action
            action = self.parse_raw_model_response(
                raw_action, observation=observation, history=history
            )

        # save results and reset
        return {
            "success": success,
            "recipe_type": example.recipe_type,
            "complexity": example.complexity_split,
            "number_of_steps": history.num_steps,
            "model_trace": history.trace(),
            "example_id": example.id,
            "images": history.images,
        }

    def batch_eval_examples(
        self,
        examples: list[PlancraftExample],
        model,
    ) -> list:
        # Initialize environments and histories
        environments = [
            PlancraftEnvironment(
                inventory=deepcopy(examples[i].slotted_inventory),
                resolution=self.resolution,
            )
            for i in range(len(examples))
        ]

        histories = [self.create_history() for _ in range(len(examples))]

        # Track which environments are still active
        active_mask = [True for _ in range(len(examples))]
        results = [None for _ in range(len(examples))]
        steps_taken = [0 for _ in range(len(examples))]
        actions = [None for _ in range(len(examples))]

        while any(active_mask) and all(steps < self.max_steps for steps in steps_taken):
            # Get observations for all active environments
            observations = []
            active_indices = []
            active_histories = []

            for i, (env, action, active) in enumerate(
                zip(environments, actions, active_mask)
            ):
                if not active:
                    continue

                if isinstance(action, StopAction):
                    # Handle stop action
                    active_mask[i] = False
                    results[i] = {
                        "success": examples[i].impossible,
                        "recipe_type": examples[i].recipe_type,
                        "complexity": examples[i].complexity_split,
                        "number_of_steps": steps_taken[i],
                        "model_trace": histories[i].trace(),
                        "example_id": examples[i].id,
                        "images": histories[i].images,
                    }
                    continue

                # Get observation
                if isinstance(action, str):
                    obs = env.step()
                    obs["target"] = examples[i].target
                    obs["message"] = action
                else:
                    obs = env.step(action)
                    obs["target"] = examples[i].target
                    obs["message"] = self.convert_observation_to_message(
                        obs, model=model
                    )

                    # Check if done
                    if self.check_done(obs["inventory"], examples[i].target):
                        active_mask[i] = False
                        results[i] = {
                            "success": True,
                            "recipe_type": examples[i].recipe_type,
                            "complexity": examples[i].complexity_split,
                            "number_of_steps": steps_taken[i],
                            "model_trace": histories[i].trace(),
                            "example_id": examples[i].id,
                            "images": histories[i].images,
                        }
                        continue

                # Add to batch lists
                active_indices.append(i)
                observations.append(obs)
                active_histories.append(histories[i])

                # Update history
                histories[i].add_observation_to_history(obs)
                histories[i].add_message_to_history(content=obs["message"], role="user")
                steps_taken[i] += 1

            if not observations:
                break

            # Batch predict actions for active environments
            raw_actions = model.batch_step(
                observations, dialogue_histories=active_histories
            )

            # Process actions for each active environment
            for batch_idx, (idx, raw_action) in enumerate(
                zip(active_indices, raw_actions)
            ):
                # if the model returns a PlancraftModelOutput, extract the action
                if isinstance(raw_action, PlancraftModelOutput):
                    # add message to history
                    histories[idx].add_message_to_history(
                        content=raw_action.action,
                        role="assistant",
                        **(raw_action.kwargs or {}),
                    )
                    actions[idx] = self.parse_raw_model_response(
                        raw_action.action,
                        observation=observations[batch_idx],
                        history=histories[idx],
                    )
                # if the model returns a string, parse the raw action
                elif isinstance(raw_action, str):
                    # add message to history
                    histories[idx].add_message_to_history(
                        content=raw_action, role="assistant"
                    )
                    actions[idx] = self.parse_raw_model_response(
                        raw_action,
                        observation=observations[batch_idx],
                        history=histories[idx],
                    )
                else:
                    raise ValueError(
                        f"model.step() output must be a string or PlancraftModelOutput, got {type(raw_action)}"
                    )

        # Fill in results for environments that didn't finish
        for i, result in enumerate(results):
            if result is None:
                results[i] = {
                    "success": False,
                    "recipe_type": examples[i].recipe_type,
                    "complexity": examples[i].complexity_split,
                    "number_of_steps": steps_taken[i],
                    "model_trace": histories[i].trace(),
                    "example_id": examples[i].id,
                    "images": histories[i].images,
                }

        return results

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
