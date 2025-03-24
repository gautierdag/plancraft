import json
import os
from typing import Any, Optional

from plancraft.config import PlancraftExample
from plancraft.environment.actions import (
    ActionHandlerBase,
    MoveActionHandler,
    SmeltActionHandler,
    ImpossibleActionHandler,
    StopAction,
)
from plancraft.environment.env import (
    PlancraftEnvironment,
    get_objective_str,
    target_and_inventory_to_text_obs,
)


def get_plancraft_examples(split: str = "train") -> list[PlancraftExample]:
    """
    Load examples from the data directory
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
        examples = json.load(f)
    return [PlancraftExample(**example) for example in examples]


class PlancraftGymWrapper:
    """
    This wrapper class just wraps the environment and actions to evaluate a single example

    This is useful if you want to bring your own agent/model to interact with the environment and not rely on the History class
    and model class in the plancraft package.
    """

    def __init__(
        self,
        example: PlancraftExample,
        actions: list[ActionHandlerBase] = [
            MoveActionHandler(),
            SmeltActionHandler(),
            ImpossibleActionHandler(),
        ],
        max_steps: int = 30,
        resolution: str = "high",
        use_text_inventory: bool = True,
    ):
        self.actions = actions
        self.max_steps = max_steps
        # whether to convert the inventory to text observation
        # if False, only the objective string is returned
        self.use_text_inventory = use_text_inventory
        self.current_step = 0
        self.stopped = False
        self.success = False
        self.example = example
        self.resolution = resolution
        self.environment = PlancraftEnvironment(
            example.slotted_inventory, resolution=self.resolution
        )
        if example.impossible:
            assert "impossible" in [action.action_name for action in actions]

    def check_done(self, inventory: dict, target: str):
        """
        Check that target object is obtained
        """
        for slot, item in inventory.items():
            # ensure the target is in the inventory (not in slot 0)
            if target == item["type"] and slot != 0:
                return True
        return False

    def parse_raw_model_response(self, generated_text: str) -> str:
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

    def step(
        self, action: Optional[str] = None
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute action and return next observation, reward, termination status, truncation status, and info

        Returns:
            observation: The environment observation after the action, observation is a dictionary with keys:
                - text: The text observation (always present)
                - inventory: The inventory after the action (if action was successful)
                - target: The target object (if action was successful)
                - image: The image observation (if action was successful)
            reward: Reward for the current action (1.0 for success, 0.0 otherwise)
            terminated: Whether the episode is done due to task completion or task failure
            truncated: Whether the episode is done due to external limits (e.g. max steps reached)
            info: Additional diagnostic information (helpful for debugging)
        """
        # Handle initial step
        if not action:
            observation = self.environment.step()
            observation["target"] = self.example.target
            if self.use_text_inventory:
                text = target_and_inventory_to_text_obs(
                    target=self.example.target, inventory=observation["inventory"]
                )
            else:
                text = get_objective_str(self.example.target)
            observation["text"] = text
            return observation, 0.0, False, False, {"steps": self.current_step}

        action = self.parse_raw_model_response(action)
        self.current_step += 1

        # Initialize return values
        reward = 0.0
        terminated = False
        truncated = False
        info = {"steps": self.current_step}

        # Handle already stopped case
        if self.stopped:
            return (
                {"text": "Plancraft environment is terminated"},
                reward,
                True,
                True,
                info,
            )

        # Handle max steps reached (truncate with no reward)
        if self.current_step > self.max_steps:
            self.success = False
            truncated = True
            info["reason"] = "max_steps_reached"
            return (
                {"text": f"Max steps ({self.max_steps}) reached"},
                reward,
                terminated,
                truncated,
                info,
            )

        # Handle stop action
        if isinstance(action, StopAction):
            self.stopped = True
            terminated = True
            #  success is True if example was truly impossible
            self.success = self.example.impossible
            if self.success:
                reward = 1.0
                info["reason"] = "correctly_identified_impossible"
            else:
                info["reason"] = "incorrect_stop"
            observation = {
                "text": "Plancraft environment is terminate due to stop action"
            }

        # Handle invalid action or non-env action
        elif isinstance(action, str):
            observation = self.environment.step()
            observation["target"] = self.example.target
            observation["text"] = action

        # Handle regular action execution
        # NOTE: if the action is valid but does not do anything
        # the environment will return the same observation
        else:
            observation = self.environment.step(action)
            observation["target"] = self.example.target

            # Generate text observation
            if self.use_text_inventory:
                text = target_and_inventory_to_text_obs(
                    target=self.example.target, inventory=observation["inventory"]
                )
            else:
                text = get_objective_str(self.example.target)

            observation["text"] = text

            self.success = self.check_done(
                observation["inventory"], self.example.target
            )

            # Set reward and termination for successful completion
            if self.success:
                reward = 1.0
                terminated = True
                self.stopped = True
                info["reason"] = "success"

        return observation, reward, terminated, truncated, info
