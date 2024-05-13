import abc

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)


class ABCModel(abc.ABC):
    """
    Model class must implement the following methods to work with evaluator
    """

    @property
    @abc.abstractmethod
    def trace(self) -> dict:
        """
        Should return a trace dictionary that will be logged
        E.g.: History of dialogue, number of tokens used, etc.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_objective(self, objective: str) -> None:
        """
        Objective is a string that sets the global objective/target
        Model can use this to decide on how to retrieve few-shot examples
        or how to initialise a plan
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(
        self, observation: dict
    ) -> SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction:
        """
        Model should output a valid action based on the 3 types available
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        """
        Should reset any interaction history to prepare for new objective
        """
        raise NotImplementedError()
