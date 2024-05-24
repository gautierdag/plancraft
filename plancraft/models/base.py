import abc

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)


class History:
    def __init__(self, objective: str):
        self.dialogue_history = []
        self.action_history = []
        self.objective = ""

    def add_message_to_history(self, content: str, role="user"):
        self.history.append({"role": role, "content": content})

    def add_action_to_history(
        self, action: SymbolicSmeltAction | RealActionInteraction | SymbolicMoveAction
    ):
        self.action_history.append(action.model_dump())

    def __str__(self):
        return str(self.dialogue_history)

    def reset(self, objective: str):
        self.dialogue_history = []
        self.action_history = []
        self.objective = objective

    def set_objective(self, objective: str):
        self.objective = objective

    def trace(self):
        return {
            "dialogue_history": self.dialogue_history,
            "action_history": self.action_history,
            "objective": self.objective,
        }


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

    @property
    @abc.abstractmethod
    def histories(self) -> list[History]:
        """
        Return the trace of the model
        """
        raise NotImplementedError()
