from plancraft.models.base import ABCModel

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)


class DummyModel(ABCModel):
    """
    Dummy model returns actions that do nothing - use to test
    """

    def __init__(self, symbolic_move_action: bool = True, **kwargs):
        self.symbolic_move_action = symbolic_move_action
        self.action_history = []

    def set_objective(self, objective: str):
        self.objective = objective

    def step(
        self, observation: dict
    ) -> SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction:
        if self.symbolic_move_action:
            return SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)
        else:
            return RealActionInteraction()

    @property
    def trace(self) -> dict:
        return {"objective": self.objective, "action_history": self.action_history}

    def reset(self) -> None:
        self.action_history = []
        self.objective = ""
