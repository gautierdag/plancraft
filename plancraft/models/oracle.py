import logging
import copy
from collections import Counter

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    RealActionInteraction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
    StopAction,
)
from plancraft.environments.planner import optimal_planner
from plancraft.environments.recipes import (
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    id_to_item,
)
from plancraft.models.base import ABCModel, History

logger = logging.getLogger(__name__)


def item_set_id_to_type(item_set_ids: set[int]):
    return set(id_to_item(i) for i in item_set_ids)


def find_free_inventory_slot(inventory: list[dict]) -> int:
    for item in inventory:
        if item["quantity"] == 0:
            if "slot" in item and item["slot"] > 10:
                return item["slot"]
            elif "index" in item and item["index"] > 10:
                return item["index"]


def find_item_in_inventory(target: str, inventory: list[dict]) -> int:
    for item in inventory:
        if item["type"] == target and item["quantity"] > 0:
            if "slot" in item:
                return item["slot"]
            elif "index" in item:
                return item["index"]
            raise ValueError("Neither slot or index is set")


def get_inventory_counter(inventory: list[dict]) -> Counter:
    counter = Counter()
    for item in inventory:
        if "slot" in item and item["slot"] == 0:
            continue
        if "index" in item and item["index"] == 0:
            continue
        if item["type"] == "air":
            continue
        counter[item["type"]] += item["quantity"]
    return counter


def get_crafting_slot_item(inventory: list[dict]) -> dict:
    for item in inventory:
        if "slot" in item and item["slot"] == 0 and item["quantity"] > 0:
            return item
        if "index" in item and item["index"] == 0 and item["quantity"] > 0:
            return item
    return None


def update_inventory(
    inventory: list[dict], slot_from: int, slot_to: int, quantity: int
) -> list[dict]:
    """
    decrements quantity of item in slot_from
    NOTE: we don't care about incrementing the items in slot_to

    """
    new_inventory = []
    for item in inventory:
        if "slot" in item and item["slot"] == slot_from:
            item["quantity"] -= quantity
        elif "index" in item and item["index"] == slot_from:
            item["quantity"] -= quantity
        new_inventory.append(item)
    return new_inventory


class OracleModel(ABCModel):
    """
    Oracle model returns actions that solve the task optimally
    """

    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Only symbolic actions are supported for oracle"
        self.history = History(objective="")
        self.plans = []
        self.subplans = []

    def reset_history(self, objective: str = ""):
        self.history.reset(objective=objective)
        self.plans = []
        self.subplans = []

    def get_plan(self, observation: dict):
        # objective="Craft an item of type: ...."
        # this simply recovering the target item to craft
        target = self.history.objective.split(": ")[-1]
        inventory_counter = get_inventory_counter(observation["inventory"])
        self.plans = optimal_planner(target=target, inventory=inventory_counter)


    def get_next_action(
        self, observation: dict
    ) -> SymbolicMoveAction | SymbolicSmeltAction:
        if len(self.subplans) > 0:
            return self.subplans.pop(0)
        if len(self.plans) == 0:
            raise ValueError("No more steps in plan")

        observed_inventory = copy.deepcopy(observation["inventory"])

        # take item from crafting slot
        if slot_item := get_crafting_slot_item(observed_inventory):
            # move item from crafting slot to inventory
            free_slot = find_free_inventory_slot(observed_inventory)
            return SymbolicMoveAction(
                slot_from=0, slot_to=free_slot, quantity=slot_item["quantity"]
            )

        plan_recipe, new_inventory = self.plans.pop(0)
        self.subplans = []
        new_inventory_counter = Counter(new_inventory)
        current_inventory = observed_inventory
        current_inventory_counter = get_inventory_counter(current_inventory)
        items_to_use_counter = current_inventory_counter - new_inventory_counter
        new_items = new_inventory_counter - current_inventory_counter
        assert len(new_items) == 1

        if isinstance(plan_recipe, ShapelessRecipe):
            crafting_slot = 1

            # add each item to crafting slots
            for item, quantity in items_to_use_counter.items():
                n = 0
                while n < quantity:
                    from_slot = find_item_in_inventory(item, current_inventory)

                    # skip if from_slot is the crafting slot
                    if from_slot == crafting_slot:
                        crafting_slot += 1
                        n += 1
                        continue

                    # low_level_plan.append(("move", item, from_slot, crafting_slot, 1))
                    action = SymbolicMoveAction(
                        slot_from=from_slot, slot_to=crafting_slot, quantity=1
                    )
                    # update state of inventory
                    current_inventory = update_inventory(
                        current_inventory, from_slot, crafting_slot, 1
                    )
                    self.subplans.append(action)

                    crafting_slot += 1
                    n += 1

        # if plan_recipe is a smelting recipe
        elif isinstance(plan_recipe, SmeltingRecipe):
            free_slot = find_free_inventory_slot(current_inventory)
            assert len(items_to_use_counter) == 1, "smelting only supports one item"
            for item, quantity in items_to_use_counter.items():
                from_slot = find_item_in_inventory(item, current_inventory)
                action = SymbolicSmeltAction(
                    slot_from=from_slot, slot_to=free_slot, quantity=quantity
                )
                self.subplans.append(action)

        # if plan_recipe is a shaped recipe
        elif isinstance(plan_recipe, ShapedRecipe):
            for i, row in enumerate(plan_recipe.kernel):
                for j, item_set in enumerate(row):
                    inventory_position = (i * 3) + j + 1
                    valid_items = item_set_id_to_type(item_set)
                    for item in valid_items:
                        if items_to_use_counter[item] > 0:
                            from_slot = find_item_in_inventory(item, current_inventory)
                            action = SymbolicMoveAction(
                                slot_from=from_slot,
                                slot_to=inventory_position,
                                quantity=1,
                            )
                            items_to_use_counter[item] -= 1
                            # update state of inventory
                            current_inventory = update_inventory(
                                current_inventory, from_slot, inventory_position, 1
                            )
                            self.subplans.append(action)
                            break
        else:
            raise NotImplementedError(f"Recipe type {type(plan_recipe)} not supported")

        return self.subplans.pop(0)

    def step(
        self, observation: dict
    ) -> list[SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction]:
        # add observation to history
        self.history.add_observation_to_history(observation)

        # get action
        if len(self.plans) == 0:
            self.get_plan(observation)
            if self.plans is None:
                self.plans = []
                return StopAction()

        action = self.get_next_action(observation)

        # add action to history
        self.history.add_action_to_history(action)
        return action
