import abc

from copy import copy

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)


class History:
    def __init__(self, objective: str = "", initial_dialogue: list[dict] = []):
        self.dialogue_history = initial_dialogue
        self.action_history = []
        self.objective = objective

    def add_message_to_history(self, content: str, role="user"):
        self.dialogue_history.append({"role": role, "content": content})

    def add_action_to_history(
        self, action: SymbolicSmeltAction | RealActionInteraction | SymbolicMoveAction
    ):
        self.action_history.append(action.model_dump())

    def __str__(self):
        return str(self.dialogue_history)

    def reset(self, objective: str = "", initial_dialogue: list[dict] = []):
        self.dialogue_history = initial_dialogue
        self.action_history = []
        self.objective = objective

    def set_objective(self, objective: str):
        self.objective = objective

    def trace(self):
        return {
            "dialogue_history": copy(self.dialogue_history),
            "action_history": copy(self.action_history),
            "objective": copy(self.objective),
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
