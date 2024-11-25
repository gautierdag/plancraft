import abc
from collections import Counter
from copy import copy

from loguru import logger

from plancraft.environments.actions import (
    SymbolicMoveAction,
    SymbolicSmeltAction,
)


class History:
    def __init__(
        self,
        objective: str = "",
        initial_dialogue: list[dict] = [],
        use_multimodal_content_format=False,
    ):
        self.dialogue_history = initial_dialogue
        self.initial_dialogue_length = len(initial_dialogue)
        self.action_history = []
        self.inventory_history = []
        self.inventory_counters = []
        self.images = []
        self.objective = objective
        self.tokens_used = 0
        self.use_multimodal_content_format = use_multimodal_content_format

    def add_message_to_history(self, content: str | dict, role="user"):
        if role == "assistant":
            logger.info(content)

        if isinstance(content, dict):
            assert "content" in content, "content key not found in message"
            content["role"] = role
            self.dialogue_history.append(content)
        else:
            # fix for listed content type
            if self.use_multimodal_content_format:
                return self.add_message_to_history(
                    content={
                        "content": [{"type": "text", "text": content}],
                        "role": role,
                    },
                    role=role,
                )
            else:
                self.dialogue_history.append({"role": role, "content": content})

    def add_action_to_history(self, action: SymbolicSmeltAction | SymbolicMoveAction):
        if action is None:
            return
        self.action_history.append(action.model_dump())

    def add_inventory_to_history(self, inventory: list[dict[str, int]]):
        self.inventory_history.append(inventory)

        # count inventory
        counter = Counter()
        for item in inventory:
            # ignore slot 0
            if "slot" in item and item["slot"] == 0:
                continue
            if "index" in item and item["index"] == 0:
                continue
            counter[item["type"]] += item["quantity"]

        self.inventory_counters.append(counter)

    def add_image_to_history(self, image):
        self.images.append(image)

    def add_observation_to_history(self, observation: dict):
        if observation is None:
            return
        if "inventory" in observation:
            clean_inv = []
            # remove empty slots
            for item in observation["inventory"]:
                if item["quantity"] > 0:
                    clean_inv.append(item)
            self.add_inventory_to_history(clean_inv)
        if "pov" in observation:
            self.add_image_to_history(observation["pov"])

    def __str__(self):
        return str(self.dialogue_history)

    def reset(self, objective: str = "", initial_dialogue: list[dict] = []):
        self.dialogue_history = initial_dialogue
        self.action_history = []
        self.inventory_history = []
        self.inventory_counters = []
        self.images = []
        self.objective = objective

    def set_objective(self, objective: str):
        self.objective = objective

    def trace(self):
        return {
            "dialogue_history": copy(
                self.dialogue_history[self.initial_dialogue_length :]
            ),
            "action_history": copy(self.action_history),
            "inventory_history": copy(self.inventory_history),
            "objective": copy(self.objective),
            "tokens_used": copy(self.tokens_used),
        }

    @property
    def num_steps(self):
        return len(self.action_history)

    def check_stuck(self, max_steps_no_change: int = 10) -> bool:
        """
        If inventory content does not change for max_steps_no_change steps
        the agent is considered stuck.

        With N=10, the oracle solver can still solve 100% of the examples
        """
        if len(self.inventory_counters) <= max_steps_no_change:
            return False

        return all(
            c == self.inventory_counters[-max_steps_no_change - 1]
            for c in self.inventory_counters[-max_steps_no_change - 1 :]
        )


class ABCModel(abc.ABC):
    """
    Model class must implement the following methods to work with evaluator
    """

    @abc.abstractmethod
    def step(
        self, observation: list[dict]
    ) -> list[SymbolicMoveAction | SymbolicSmeltAction]:
        """
        Model should output a valid action based on the 3 types available

        Note this is a batch operation, so the model should return a list of actions
        for each observation in the batch
        """
        raise NotImplementedError()

    def reset_history(self, objective: str = ""):
        self.history.reset(objective=objective)
