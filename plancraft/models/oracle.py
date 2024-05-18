from collections import defaultdict

from plancraft.config import Config
from plancraft.environments.actions import (
    RealActionInteraction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
)
from plancraft.environments.planner import optimal_planner
from plancraft.models.base import ABCModel


class OracleModel(ABCModel):
    """
    Oracle model returns actions that solve the task optimally
    """

    def __init__(self, cfg: Config):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Only symbolic actions are supported"
        self.action_history = []
        self.plan = []
        self.plan_idx = 0

    def set_objective(self, objective: str):
        self.objective = objective
        # objective="Craft an item of type: ...."
        # this simply recovering the target item to craft
        self.target = objective.split(": ")[-1]

    def step(
        self, observation: dict
    ) -> SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction:
        if not self.plan:
            inventory_dict = defaultdict(int)
            for item in observation["inventory"]:
                inventory_dict[item["type"]] += item["quantity"]
            self.plan = optimal_planner(target=self.target, inventory=inventory_dict)

        # TODO: Implement the logic to return the next action in the plan
        # note that plans abstract away crafting, so will need to decompose each craft step into the move steps
        action = SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)
        self.action_history.append(action)
        return action

    @property
    def trace(self) -> dict:
        return {"objective": self.objective, "action_history": self.action_history}

    def reset(self) -> None:
        self.action_history = []
        self.objective = ""
