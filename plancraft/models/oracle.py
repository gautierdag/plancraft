from plancraft.models.base import ABCModel

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)

from plancraft.environments.planner import optimal_planner


class OracleModel(ABCModel):
    """
    Oracle model returns actions that solve the task optimally
    """

    def __init__(self, symbolic_move_action: bool = True, **kwargs):
        assert symbolic_move_action, "Only symbolic move actions are supported"
        self.action_history = []
        self.plan = []
        self.plan_idx = 0

    def set_objective(self, objective: str):
        self.objective = objective
        # objective="Craft an item of type: ...."
        self.target = objective.split(": ")[-1]

    def step(
        self, observation: dict
    ) -> SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction:
        if not self.plan:
            self.plan = optimal_planner(observation["inventory"], self.target)

        
        action = SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)
        self.action_history.append(action)
        return action

    @property
    def trace(self) -> dict:
        return {"objective": self.objective, "action_history": self.action_history}

    def reset(self) -> None:
        self.action_history = []
        self.objective = ""
