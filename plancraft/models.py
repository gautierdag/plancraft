import abc

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)


class BaseModel(abc.ABC):
    @property
    @abc.abstractmethod
    def trace(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_objective(self, objective: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, observation: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()


class DummyModel(BaseModel):
    def __init__(self, symbolic_move_action: bool = True, **kwargs):
        self.symbolic_move_action = symbolic_move_action
        self.action_history = []

    def set_objective(self, objective: str):
        self.objective = objective

    def step(
        self, observation: dict
    ) -> SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction:
        if self.symbolic_move_action:
            # return symbolic move action random
            return SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)
        else:
            # return interactive move action random
            return RealActionInteraction()

    @property
    def trace(self) -> dict:
        return {"objective": self.objective, "action_history": self.action_history}

    def reset(self) -> None:
        self.action_history = []
        self.objective = ""
