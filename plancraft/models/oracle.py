from collections import defaultdict

from plancraft.config import Config
from plancraft.environments.actions import (
    RealActionInteraction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
)
from plancraft.environments.planner import optimal_planner
from plancraft.environments.recipes import ShapedRecipe, ShapelessRecipe, SmeltingRecipe
from plancraft.models.base import ABCModel, History


class OracleModel(ABCModel):
    """
    Oracle model returns actions that solve the task optimally
    """

    def __init__(self, cfg: Config):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Only symbolic actions are supported for oracle"
        self.histories = [
            History(objective="") for _ in range(cfg.plancraft.batch_size)
        ]
        self.plans = [[] for _ in range(cfg.plancraft.batch_size)]
        self.plan_idxs = [0 for _ in range(cfg.plancraft.batch_size)]
        self.subplans = [[] for _ in range(cfg.plancraft.batch_size)]

    # def set_objective(self, objective: str):
    #     self.objective = objective
    #     # objective="Craft an item of type: ...."
    #     # this simply recovering the target item to craft
    #     self.target = objective.split(": ")[-1]

    def find_empty_slot(self, inventory):
        for item in inventory:
            if item["quantity"] == "0":
                return item["slot"]
        return None

    def populate_subplan(self):
        subplan_task = self.plan[self.plan_idx]
        if isinstance(subplan_task[0], ShapedRecipe):
            # self.subplan = subplan_task[0].inputs
            pass
        elif isinstance(subplan_task[0], ShapelessRecipe):
            # self.subplan = subplan_task[0].inputs
            pass
        elif isinstance(subplan_task[0], SmeltingRecipe):
            pass
            # target_item = SmeltingRecipe.result.item
            # if the recipe is a smelting recipe, we need to move the input item to the furnace
            # self.subplan = [SymbolicSmeltAction()
            # action = SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)

    def step(
        self, batch_observations: list[dict]
    ) -> list[SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction]:
        return
        # for i, observation in enumerate(batch_observations):

        # if not self.plan:
        #     inventory_dict = defaultdict(int)
        #     for item in batch_observations["inventory"]:
        #         inventory_dict[item["type"]] += item["quantity"]
        #     self.plan = optimal_planner(target=self.target, inventory=inventory_dict)

        # TODO: Implement the logic to return the next action in the plan
        # note that plans abstract away crafting, so will need to decompose each craft step into the move steps

        # if subplan is empty, move to the next plan
        # if not self.subplan:
        #     self.plan_idx += 1
        #     self.populate_subplan()

        # # if subplan is still empty
        # if len(self.subplan) > 1:
        #     action = self.subplan.pop(0)
        # else:
        #     # no op
        #     action = SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)

        # self.action_history.append(action)
        # return action
