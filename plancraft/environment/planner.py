import copy
import time
from collections import Counter

import networkx as nx

from plancraft.environment.actions import (
    MoveAction,
    SmeltAction,
    StopAction,
)
from plancraft.environment.recipes import (
    RECIPES,
    BaseRecipe,
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    id_to_item,
)
from plancraft.environment.items import all_data

RECIPE_GRAPH = nx.DiGraph()

for item, recipes in RECIPES.items():
    for recipe in recipes:
        RECIPE_GRAPH.add_node(recipe.result.item)
        for ingredient in recipe.inputs:
            RECIPE_GRAPH.add_node(ingredient)
            RECIPE_GRAPH.add_edge(ingredient, recipe.result.item)


MAX_STACK_SIZE = {}
for data_item in all_data["items"]:
    if data_item["stackable"]:
        MAX_STACK_SIZE[data_item["type"]] = data_item["stackSize"]
    else:
        MAX_STACK_SIZE[data_item["type"]] = 1


def get_ancestors(target: str):
    return list(nx.ancestors(RECIPE_GRAPH, source=target))


def optimal_planner(
    target: str,
    inventory: dict[str, int],
    steps=[],
    best_steps=None,
    max_steps=40,
    timeout=30,
) -> list[tuple[BaseRecipe, dict[str, int]]]:
    """
    Optimal planner for crafting the target item from the given inventory.

    Uses depth-first search with memoization to find the shortest path of crafting steps.

    Args:
        target: The target item to craft.
        inventory: The current inventory.
        steps: The current path of crafting steps.
        best_steps: The best path of crafting steps found so far.
        max_steps: The maximum number of steps to take.
        timeout: The maximum time to spend searching for a solution.

    Returns:
        list of tuples of (recipe, inventory) for each step in the optimal path.
    """

    memo = {}
    # only look at recipes that are ancestors of the target
    ancestors = get_ancestors(target)
    # sort to put the closest ancestors first
    ancestors = sorted(
        ancestors,
        key=lambda x: nx.shortest_path_length(RECIPE_GRAPH, source=x, target=target),
    )

    time_now = time.time()

    def dfs(starting_inventory, steps, best_steps):
        # If we have exceeded the timeout, return the best known path so far.
        if time.time() - time_now > timeout:
            raise TimeoutError("Timeout exceeded")

        memo_key = (frozenset(starting_inventory.items()), len(steps))
        if memo_key in memo:
            return memo[memo_key]

        if best_steps is not None and len(steps) >= len(best_steps):
            # If we already have a shorter or equally short solution, do not proceed further.
            return best_steps

        if len(steps) > max_steps:
            # If we have already exceeded the maximum number of steps, do not proceed further.
            return best_steps

        if target in starting_inventory and starting_inventory[target] > 0:
            # If the target item is already in the inventory in the required amount, return the current path.
            if best_steps is None or len(steps) < len(best_steps):
                return steps
            return best_steps

        for recipe_name in [target] + ancestors:
            # skip if already have 9 of the item
            if starting_inventory.get(recipe_name, 0) >= 9:
                continue
            # TODO prevent looping between equivalent recipes (coal <-> coal_block)
            for recipe in RECIPES[recipe_name]:
                if recipe.can_craft_from_inventory(starting_inventory):
                    # Craft this item and update the inventory.
                    new_inventory = recipe.craft_from_inventory(starting_inventory)
                    # Add this step to the path.
                    new_steps = steps + [(recipe, new_inventory)]

                    # Recursively try to craft the target item with the updated inventory.
                    candidate_steps = dfs(new_inventory, new_steps, best_steps)

                    # Update the best known path if the candidate path is better.
                    if candidate_steps is not None and (
                        best_steps is None or len(candidate_steps) < len(best_steps)
                    ):
                        best_steps = candidate_steps

        memo[memo_key] = best_steps
        return best_steps

    try:
        path = dfs(inventory, steps, best_steps)
        return path

    except TimeoutError:
        return None


def item_set_id_to_type(item_set_ids: set[int]):
    return set(id_to_item(i) for i in item_set_ids)


def find_free_inventory_slot(
    inventory: dict, from_slot: int, from_item_type=None, from_item_quantity=None
) -> int:
    # find a free slot in the inventory for the item in from_slot
    if from_item_type is None:
        from_item_type = inventory[from_slot]["type"]
    if from_item_quantity is None:
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


def find_item_in_inventory(
    target: str, inventory: dict, min_quantity_needed: set = set()
) -> int:
    for slot, item in inventory.items():
        if item["type"] == target and item["quantity"] > 0:
            # if we don't need to keep an item in the slot, we can use it
            if slot not in min_quantity_needed:
                return slot
            # if we need to keep an item in the slot, we can only use it quantity>1
            elif item["quantity"] > 1:
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
    from_item = new_inventory[slot_from]
    if slot_to not in new_inventory:
        new_inventory[slot_to] = {"type": from_item["type"], "quantity": quantity}
    else:
        new_inventory[slot_to]["quantity"] += quantity

    new_inventory[slot_from]["quantity"] -= quantity
    if new_inventory[slot_from]["quantity"] <= 0:
        del new_inventory[slot_from]
    return new_inventory


def get_plan(observation: dict):
    # this simply recovering the target item to craft
    inventory_counter = get_inventory_counter(observation["inventory"])
    return optimal_planner(target=observation["target"], inventory=inventory_counter)


def decompose_subgoal(
    current_inventory: dict, plan_recipe, new_inventory: dict[str, int]
) -> list[str]:
    """
    For a given plan_recipe and inventory, output the list of action to craft recipe
    """
    subplan = []
    new_inventory_counter = Counter(new_inventory)
    current_inventory_counter = get_inventory_counter(current_inventory)
    items_to_use_counter = current_inventory_counter - new_inventory_counter
    new_items = new_inventory_counter - current_inventory_counter
    assert len(new_items) == 1

    # if plan_recipe is a smelting recipe
    if isinstance(plan_recipe, SmeltingRecipe):
        assert len(items_to_use_counter) == 1, "smelting only supports one item"
        for item, quantity in items_to_use_counter.items():
            from_slot = find_item_in_inventory(item, current_inventory)
            out_item_type, out_item_quantity = (
                plan_recipe.result.item,
                plan_recipe.result.count,
            )
            # find a free slot in the inventory for the item that will be smelted
            free_slot = find_free_inventory_slot(
                current_inventory,
                from_slot=from_slot,
                from_item_type=out_item_type,
                from_item_quantity=out_item_quantity,
            )
            action = SmeltAction(
                slot_from=from_slot, slot_to=free_slot, quantity=quantity
            )
            subplan.append(str(action))

            # update inventory to decrement quantity of item in from_slot and increment quantity of item in free_slot
            current_inventory = dict(current_inventory)
            if free_slot not in current_inventory:
                current_inventory[free_slot] = {
                    "type": out_item_type,
                    "quantity": out_item_quantity,
                }
            else:
                current_inventory[free_slot]["quantity"] += quantity

            current_inventory[from_slot]["quantity"] -= quantity
            if current_inventory[from_slot]["quantity"] <= 0:
                del current_inventory[from_slot]

            return subplan, current_inventory

    elif isinstance(plan_recipe, ShapelessRecipe):
        crafting_slot = 1
        min_quantity_needed = set()
        while crafting_slot < 10:
            # if something is already in the crafting slot
            if (
                crafting_slot in current_inventory
                and current_inventory[crafting_slot]["quantity"] > 0
            ):
                # check it is a desired item
                crafting_slot_item = current_inventory[crafting_slot]["type"]
                # if it is a desired item, skip it
                if (
                    crafting_slot_item in items_to_use_counter
                    and items_to_use_counter[crafting_slot_item] > 0
                ):
                    items_to_use_counter[crafting_slot_item] -= 1
                    if items_to_use_counter[crafting_slot_item] == 0:
                        del items_to_use_counter[crafting_slot_item]
                    min_quantity_needed.add(crafting_slot)
                    crafting_slot += 1
                    continue
                # if it is not a desired item, move it to a free slot
                else:
                    free_slot = find_free_inventory_slot(
                        current_inventory, from_slot=crafting_slot
                    )
                    action = MoveAction(
                        slot_from=crafting_slot,
                        slot_to=free_slot,
                        quantity=current_inventory[crafting_slot]["quantity"],
                    )
                    subplan.append(str(action))
                    current_inventory = update_inventory(
                        current_inventory,
                        crafting_slot,
                        free_slot,
                        current_inventory[crafting_slot]["quantity"],
                    )

            # if there are still items to add
            if len(items_to_use_counter) != 0:
                item = next(iter(items_to_use_counter))
                from_slot = find_item_in_inventory(
                    item, current_inventory, min_quantity_needed
                )
                action = MoveAction(
                    slot_from=from_slot, slot_to=crafting_slot, quantity=1
                )
                subplan.append(str(action))
                current_inventory = update_inventory(
                    current_inventory, from_slot, crafting_slot, 1
                )
                items_to_use_counter[item] -= 1
                if items_to_use_counter[item] == 0:
                    del items_to_use_counter[item]

            # update state of inventory
            crafting_slot += 1

    # if plan_recipe is a shaped recipe
    elif isinstance(plan_recipe, ShapedRecipe):
        min_quantity_needed = set()
        seen_kernel = set()
        for i, row in enumerate(plan_recipe.kernel):
            for j, item_set in enumerate(row):
                inventory_position = (i * 3) + j + 1
                seen_kernel.add(inventory_position)
                valid_items = item_set_id_to_type(item_set)

                # if the inventory position is needed to be empty
                if (
                    valid_items == {None}
                    and inventory_position in current_inventory
                    and current_inventory[inventory_position]["quantity"] > 0
                ):
                    free_slot = find_free_inventory_slot(
                        current_inventory, from_slot=inventory_position
                    )
                    action = MoveAction(
                        slot_from=inventory_position,
                        slot_to=free_slot,
                        quantity=current_inventory[inventory_position]["quantity"],
                    )
                    current_inventory = update_inventory(
                        current_inventory,
                        inventory_position,
                        free_slot,
                        current_inventory[inventory_position]["quantity"],
                    )
                    subplan.append(str(action))
                    continue

                # if the inventory position is needed to be filled
                added_item = False
                for item in valid_items:
                    # check if item is already added
                    if added_item:
                        break

                    # if item is already in the correct position, skip
                    if (
                        inventory_position in current_inventory
                        and current_inventory[inventory_position]["type"] == item
                    ) and current_inventory[inventory_position]["quantity"] > 0:
                        min_quantity_needed.add(inventory_position)
                        # decrement the quantity of the items to use
                        items_to_use_counter[item] -= 1
                        if items_to_use_counter[item] == 0:
                            del items_to_use_counter[item]
                        break

                    if items_to_use_counter[item] > 0:
                        # check and remove any item in the inventory position
                        if (
                            inventory_position in current_inventory
                            and current_inventory[inventory_position]["quantity"] > 0
                            and current_inventory[inventory_position]["type"] != item
                        ):
                            free_slot = find_free_inventory_slot(
                                current_inventory, from_slot=inventory_position
                            )
                            action = MoveAction(
                                slot_from=inventory_position,
                                slot_to=free_slot,
                                quantity=current_inventory[inventory_position][
                                    "quantity"
                                ],
                            )
                            current_inventory = update_inventory(
                                current_inventory,
                                inventory_position,
                                free_slot,
                                current_inventory[inventory_position]["quantity"],
                            )
                            subplan.append(str(action))

                        # move item to correct position
                        from_slot = find_item_in_inventory(
                            item, current_inventory, min_quantity_needed
                        )
                        action = MoveAction(
                            slot_from=from_slot,
                            slot_to=inventory_position,
                            quantity=1,
                        )
                        items_to_use_counter[item] -= 1
                        added_item = True
                        # update state of inventory
                        current_inventory = update_inventory(
                            current_inventory, from_slot, inventory_position, 1
                        )
                        subplan.append(str(action))

        # ensure all other items are removed
        leftover_kernel = set(range(1, 10)) - seen_kernel
        for slot in leftover_kernel:
            if slot in current_inventory and current_inventory[slot]["quantity"] > 0:
                free_slot = find_free_inventory_slot(current_inventory, from_slot=slot)
                action = MoveAction(
                    slot_from=slot,
                    slot_to=free_slot,
                    quantity=current_inventory[slot]["quantity"],
                )
                current_inventory = update_inventory(
                    current_inventory,
                    slot,
                    free_slot,
                    current_inventory[slot]["quantity"],
                )
                subplan.append(str(action))
    else:
        raise NotImplementedError(f"Recipe type {type(plan_recipe)} not supported")

    # move crafted item to free slot
    current_inventory[0] = {
        "type": plan_recipe.result.item,
        "quantity": plan_recipe.result.count,
    }
    free_slot = find_free_inventory_slot(current_inventory, from_slot=0)
    subplan.append(
        str(
            MoveAction(
                slot_from=0, slot_to=free_slot, quantity=plan_recipe.result.count
            )
        )
    )
    current_inventory = update_inventory(
        current_inventory, 0, free_slot, plan_recipe.result.count
    )
    # decrement all quantities of items present in crafting slots
    for i in range(1, 10):
        if i in current_inventory:
            current_inventory[i]["quantity"] -= 1
            if current_inventory[i]["quantity"] <= 0:
                del current_inventory[i]

    return subplan, current_inventory


def get_subplans(observation: dict) -> tuple[list[list[str]], list]:
    current_inventory = copy.deepcopy(observation["inventory"])
    plan = get_plan(observation)
    # get action
    if plan is None or len(plan) == 0:
        return [[str(StopAction())]], []
    # plan_recipe, new_inventory = plan[0]
    subplans = []
    # Calculate the subplans for each step in the plan
    for plan_recipe, new_inventory in plan:
        subplan, current_inventory = decompose_subgoal(
            current_inventory, plan_recipe, new_inventory
        )
        subplans.append(subplan)
    return subplans, plan
