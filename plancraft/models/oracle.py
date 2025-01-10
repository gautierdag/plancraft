import copy
from collections import Counter

import torch

from plancraft.config import EvalConfig
from plancraft.environment.actions import (
    MoveAction,
    SmeltAction,
    StopAction,
)
from plancraft.environment.planner import optimal_planner
from plancraft.environment.recipes import (
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    id_to_item,
)
from plancraft.environment.sampler import MAX_STACK_SIZE
from plancraft.models.base import PlancraftBaseModel
from plancraft.models.bbox_model import IntegratedBoundingBoxModel


def item_set_id_to_type(item_set_ids: set[int]):
    return set(id_to_item(i) for i in item_set_ids)


def find_free_inventory_slot(inventory: dict, from_slot: int) -> int:
    # find a free slot in the inventory for the item in from_slot
    from_item_type = inventory[from_slot]["type"]
    from_item_quantity = inventory[from_slot]["quantity"]

    empty_slots = set(range(10, 46)) - set(inventory.keys()) - set([from_slot])

    type_to_slot = {}
    slot_to_quantity = {}
    for slot, item in inventory.items():
        if slot == from_slot:
            continue
        item_type = item["type"]
        if item_type not in type_to_slot:
            type_to_slot[item_type] = [slot]
        else:
            type_to_slot[item_type].append(slot)

        slot_to_quantity[slot] = item["quantity"]

    assert from_item_type is not None, f"Item not found in slot {from_slot}"

    # if there is a free slot with the same item type
    if from_item_type in type_to_slot:
        for slot in type_to_slot[from_item_type]:
            if (
                slot_to_quantity[slot] + from_item_quantity
                <= MAX_STACK_SIZE[from_item_type]
            ):
                return slot
    if len(empty_slots) > 0:
        return empty_slots.pop()

    raise ValueError("No free slot found")


def find_item_in_inventory(target: str, inventory: dict) -> int:
    for slot, item in inventory.items():
        if item["type"] == target and item["quantity"] > 0:
            return slot


def get_inventory_counter(inventory: dict) -> Counter:
    counter = Counter()
    for slot, item in inventory.items():
        if slot == 0:
            continue
        counter[item["type"]] += item["quantity"]
    return counter


def get_crafting_slot_item(inventory: dict) -> dict:
    for slot, item in inventory.items():
        if slot == 0 and item["quantity"] > 0:
            return item
    return None


def update_inventory(
    inventory: dict, slot_from: int, slot_to: int, quantity: int
) -> dict:
    """
    decrements quantity of item in slot_from
    NOTE: we don't care about incrementing the items in slot_to

    """
    new_inventory = dict(inventory)
    new_inventory[slot_from]["quantity"] -= quantity
    return new_inventory


class OracleModel(PlancraftBaseModel):
    """
    Oracle model returns actions that solve the task optimally
    """

    def __init__(self, cfg: EvalConfig):
        self.plans = []
        self.subplans = []
        self.use_fasterrcnn = cfg.plancraft.use_fasterrcnn

        self.bbox_model = None
        if self.use_fasterrcnn:
            # fasterrcnn is not multimodal model but a separate model
            self.bbox_model = IntegratedBoundingBoxModel.from_pretrained(
                "gautierdag/plancraft-fasterrcnn"
            )
            self.bbox_model.eval()
            if torch.cuda.is_available():
                self.bbox_model.cuda()

    def reset(self):
        self.plans = []
        self.subplans = []

    def get_plan(self, observation: dict):
        # this simply recovering the target item to craft
        inventory_counter = get_inventory_counter(observation["inventory"])
        return optimal_planner(
            target=observation["target"], inventory=inventory_counter
        )

    def get_next_action(self, observation: dict) -> MoveAction | SmeltAction:
        if len(self.subplans) > 0:
            return self.subplans.pop(0)
        if len(self.plans) == 0:
            raise ValueError("No more steps in plan")

        if self.bbox_model is not None:
            observed_inventory = self.bbox_model.get_inventory(
                observation["image"].copy()
            )
        else:
            observed_inventory = copy.deepcopy(observation["inventory"])

        # take item from crafting slot
        if slot_item := get_crafting_slot_item(observed_inventory):
            # move item from crafting slot to inventory
            free_slot = find_free_inventory_slot(observed_inventory, from_slot=0)
            return MoveAction(
                slot_from=0, slot_to=free_slot, quantity=slot_item["quantity"]
            )

        plan_recipe, new_inventory = self.plans.pop(0)
        self.subplans = []
        new_inventory_counter = Counter(new_inventory)
        current_inventory = observed_inventory
        current_inventory_counter = get_inventory_counter(current_inventory)
        items_to_use_counter = current_inventory_counter - new_inventory_counter
        new_items = new_inventory_counter - current_inventory_counter
        if not self.use_fasterrcnn:
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

                    action = MoveAction(
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
            assert len(items_to_use_counter) == 1, "smelting only supports one item"
            for item, quantity in items_to_use_counter.items():
                from_slot = find_item_in_inventory(item, current_inventory)
                free_slot = find_free_inventory_slot(
                    current_inventory, from_slot=from_slot
                )
                action = SmeltAction(
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
                            action = MoveAction(
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

    def step(self, observation: dict, **kwargs) -> str:
        # get action
        if len(self.plans) == 0:
            self.plans = self.get_plan(observation)
            if self.plans is None:
                self.plans = []
                return str(StopAction())

        action = self.get_next_action(observation)
        return str(action)
