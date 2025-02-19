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
        observation = environment.step()
        # add target and first message to history
        observation["target"] = example.target
        observation["message"] = self.convert_observation_to_message(
            observation, model=model
        )

        success = False
        # run episode until stuck or until max steps is reached
        while history.num_steps < self.max_steps:
            # add observation to history
            history.add_observation_to_history(observation)
            history.add_message_to_history(content=observation["message"], role="user")
            # predict next action
            raw_action = model.step(observation, dialogue_history=history)

            # if the model returns a PlancraftModelOutput, extract the action
            if isinstance(raw_action, PlancraftModelOutput):
                history.add_message_to_history(
                    content=raw_action.action,
                    role="assistant",
                    **(raw_action.kwargs or {}),
                )
                raw_action = raw_action.action
            elif isinstance(raw_action, str):
                history.add_message_to_history(content=raw_action, role="assistant")
            else:
                raise ValueError(
                    f"model.step() output must be a string or PlancraftModelOutput, got {type(raw_action)}"
                )

            # parse the raw action
            action = self.parse_raw_model_response(
                raw_action, observation=observation, history=history
            )

            # if the action is stop then we end the episode
            if isinstance(action, StopAction):
                # if the action is stop and task is impossible then success
                # otherwise we should not have stopped
                observation = None
                success = example.impossible
            # action is external tool then it is str
            if isinstance(action, str):
                observation = environment.step()
                observation["target"] = example.target
                observation["message"] = action
            # action is environment action
            else:
                observation = environment.step(action)
                observation["target"] = example.target
                observation["message"] = self.convert_observation_to_message(
                    observation, model=model
                )
                # check if the episode is done
                success = self.check_done(observation["inventory"], example.target)

            # update model with success or failure
            # observation is the next state after the action (s1)
            # history is the dialogue history
            # -- the last message contains the action taken (a0)
            # -- the second to last message is the observation (s0)
            # success is whether the episode is sucessful (r)
            model.update(
                observation=observation, history=history, success=success, action=action
            )

            # exit if success
            if success or isinstance(action, StopAction):
                break

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
        """
        Similar to eval_example, but processes multiple examples at once.

        Tracks which environments are still active until they've either succeeded,
        reached max steps, or invoked StopAction.
        """

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
        observations = []

        # Initialize observations (s0) and user messages from environment
        for i in range(len(examples)):
            obs = environments[i].step()
            obs["target"] = examples[i].target
            obs["message"] = self.convert_observation_to_message(obs, model=model)
            observations.append(obs)

        # Process until all done or max steps reached
        while any(active_mask) and all(
            history.num_steps < self.max_steps for history in histories
        ):
            # Gather active environments
            active_indices = [
                i
                for i, active in enumerate(active_mask)
                if active and histories[i].num_steps < self.max_steps
            ]
            if not active_indices:
                break

            # For each active environment, add new obs to history for next iteration
            for env_idx in active_indices:
                if active_mask[env_idx]:
                    histories[env_idx].add_observation_to_history(observations[env_idx])
                    histories[env_idx].add_message_to_history(
                        content=observations[env_idx]["message"], role="user"
                    )

            batch_observations = [observations[i] for i in active_indices]
            batch_histories = [histories[i] for i in active_indices]

            # Predict next actions in batch
            raw_actions = model.batch_step(
                batch_observations, dialogue_histories=batch_histories
            )

            # Process each raw action and update environment/history
            successes = []
            actions = []
            for env_idx, raw_action in zip(active_indices, raw_actions):
                # Add model's message to history
                if isinstance(raw_action, PlancraftModelOutput):
                    histories[env_idx].add_message_to_history(
                        content=raw_action.action,
                        role="assistant",
                        **(raw_action.kwargs or {}),
                    )
                    raw_action = raw_action.action
                elif isinstance(raw_action, str):
                    histories[env_idx].add_message_to_history(
                        content=raw_action, role="assistant"
                    )
                else:
                    raise ValueError(
                        f"model.batch_step() must return list[str] or list[PlancraftModelOutput], got {type(raw_action)}"
                    )

                # Parse action
                action = self.parse_raw_model_response(
                    raw_action,
                    observation=observations[env_idx],
                    history=histories[env_idx],
                )
                actions.append(action)
                success = False
                # If action is StopAction
                if isinstance(action, StopAction):
                    # if the action is StopAction and the example is impossible,
                    # we consider that a 'success' in the sense that the model recognized it can't be done
                    success = examples[env_idx].impossible
                    observations[env_idx] = None
                # If parsed action is a string, it's a message
                elif isinstance(action, str):
                    obs = environments[env_idx].step()
                    obs["target"] = examples[env_idx].target
                    obs["message"] = action
                    observations[env_idx] = obs
                # Otherwise it's an actual environment action
                else:
                    obs = environments[env_idx].step(action)
                    obs["target"] = examples[env_idx].target
                    obs["message"] = self.convert_observation_to_message(
                        obs, model=model
                    )
                    observations[env_idx] = obs
                    success = self.check_done(
                        obs["inventory"], examples[env_idx].target
                    )

                successes.append(success)

                # If done, or action was stop, mark inactive and store result
                if (
                    success
                    or isinstance(action, StopAction)
                    or histories[env_idx].num_steps >= self.max_steps
                ):
                    active_mask[env_idx] = False
                    results[env_idx] = {
                        "success": success,
                        "recipe_type": examples[env_idx].recipe_type,
                        "complexity": examples[env_idx].complexity_split,
                        "number_of_steps": histories[env_idx].num_steps,
                        "model_trace": histories[env_idx].trace(),
                        "example_id": examples[env_idx].id,
                        "images": histories[env_idx].images,
                    }

            # Update the model for this single environment
            batch_observations = [observations[i] for i in active_indices]
            batch_histories = [histories[i] for i in active_indices]
            model.batch_update(
                observations=batch_observations,
                histories=batch_histories,
                successes=successes,
                actions=actions,
            )

        # Fill in results for any environment that never completed
        for i, result in enumerate(results):
            if result is None:
                results[i] = {
                    "success": False,
                    "recipe_type": examples[i].recipe_type,
                    "complexity": examples[i].complexity_split,
                    "number_of_steps": histories[i].num_steps,
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
