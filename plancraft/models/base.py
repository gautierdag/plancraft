import abc

from copy import copy

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)


class History:
    def __init__(
        self,
        objective: str = "",
        initial_dialogue: list[dict] = [],
        is_multimodal=False,
    ):
        self.dialogue_history = initial_dialogue
        self.action_history = []
        self.inventory_history = []
        self.images = []
        self.objective = objective
        self.tokens_used = 0
        self.is_multimodal = is_multimodal

    def add_message_to_history(self, content: str | dict, role="user"):
        if isinstance(content, dict):
            assert "content" in content, "content key not found in message"
            content["role"] = role
            self.dialogue_history.append(content)
        else:
            # fix for listed content type
            if self.is_multimodal:
                return self.add_message_to_history(
                    content={
                        "content": [{"type": "text", "text": content}],
                        "role": role,
                    },
                    role=role,
                )
            else:
                self.dialogue_history.append({"role": role, "content": content})

    def add_action_to_history(
        self, action: SymbolicSmeltAction | RealActionInteraction | SymbolicMoveAction
    ):
        if action is None:
            return
        self.action_history.append(action.model_dump())

    def add_inventory_to_history(self, inventory: list[dict[str, int]]):
        self.inventory_history.append(inventory)

    def add_image_to_history(self, image):
        self.images.append(image)

    def add_observation_to_history(self, observation: dict):
        if observation is None:
            return
        if "inventory" in observation:
            self.add_inventory_to_history(observation["inventory"])
        if "pov" in observation:
            self.add_image_to_history(observation["pov"])

    def __str__(self):
        return str(self.dialogue_history)

    def reset(self, objective: str = "", initial_dialogue: list[dict] = []):
        self.dialogue_history = initial_dialogue
        self.action_history = []
        self.inventory_history = []
        self.images = []
        self.objective = objective

    def set_objective(self, objective: str):
        self.objective = objective

    def trace(self):
        return {
            "dialogue_history": copy(self.dialogue_history),
            "action_history": copy(self.action_history),
            "inventory_history": copy(self.inventory_history),
            "objective": copy(self.objective),
            "tokens_used": copy(self.tokens_used),
        }

    @property
    def num_steps(self):
        return len(self.action_history)


class ABCModel(abc.ABC):
    """
    Model class must implement the following methods to work with evaluator
    """

    @abc.abstractmethod
    def step(
        self, observation: list[dict]
    ) -> list[SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction]:
        """
        Model should output a valid action based on the 3 types available

        Note this is a batch operation, so the model should return a list of actions
        for each observation in the batch
        """
        raise NotImplementedError()

    def reset_history(self, history_idx: int, objective: str = ""):
        self.histories[history_idx].reset(objective=objective)
