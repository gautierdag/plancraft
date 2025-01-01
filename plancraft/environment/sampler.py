import math
import random
from collections import Counter

import numpy as np
from loguru import logger

from plancraft.environment.items import ALL_ITEMS, all_data
from plancraft.environment.planner import get_ancestors, optimal_planner
from plancraft.environment.recipes import RECIPES

MAX_STACK_SIZE = {}
for data_item in all_data["items"]:
    if data_item["stackable"]:
        MAX_STACK_SIZE[data_item["type"]] = data_item["stackSize"]
    else:
        MAX_STACK_SIZE[data_item["type"]] = 1


def sample_distractors(
    exclude_set: set = None, num_distractors: int = 16
) -> dict[str, int]:
    distractors = {}
    while len(distractors) < num_distractors:
        item = random.choice(ALL_ITEMS)
        if exclude_set is not None and item in exclude_set:
            continue
        count = random.randint(1, MAX_STACK_SIZE[item])
        distractors[item] = count
    return distractors


def assign_to_slots(inventory: dict[str, int]) -> list[dict]:
    # slots available outside of crafting interface
    available_slots = list(range(10, 46))
    random.shuffle(available_slots)
    inventory_dict = {}

    for item, total_count in inventory.items():
        while total_count > 0:
            if len(available_slots) == 0:
                logger.info("Not enough slots available")
                break
            slot = available_slots.pop()
            count_in_slot = min(total_count, MAX_STACK_SIZE[item])
            inventory_dict[slot] = {"type": item, "quantity": count_in_slot}
            total_count -= count_in_slot

    return inventory_dict


def sample_recipes(
    target: str,
    overall_exclude_set: set,
    target_count: int = 1,
    current_depth=0,
    max_depth=20,
) -> tuple[set, set]:
    # stop if the depth is too high
    if current_depth > max_depth:
        return {}, overall_exclude_set

    # get all the recipes that can craft the target
    overall_exclude_set.update([target])
    local_exclude_set = set()
    random_recipes = []
    for r in RECIPES[target]:
        recipe_inputs, exclude_set = r.sample_inputs()
        # if inputs are already in the exclude set, skip this recipe (ensures no cycle)
        if exclude_set.intersection(overall_exclude_set):
            return {}, overall_exclude_set
        local_exclude_set.update(exclude_set)
        random_recipes.append((r, recipe_inputs))

    overall_exclude_set |= local_exclude_set

    # no recipes found
    if len(random_recipes) == 0:
        return {}, overall_exclude_set

    # sample a random recipe
    random_recipe = random.choice(random_recipes)
    recipe, start_inputs = random_recipe

    # recipe will not produce enough
    if recipe.result.count < target_count:
        # must do recipe X times
        recipe_multiplier = math.ceil(target_count / recipe.result.count)
        start_inputs = {k: v * recipe_multiplier for k, v in start_inputs.items()}

    for input_item in list(start_inputs.keys()):
        # randomize depth first search to end early
        if random.choice([True, False]):
            continue

        children_recipe_inputs, updated_exclude_set = sample_recipes(
            target=input_item,
            overall_exclude_set=overall_exclude_set,
            target_count=start_inputs[input_item],
            current_depth=current_depth + 1,
        )
        if len(children_recipe_inputs) == 0:
            continue

        overall_exclude_set.update(updated_exclude_set)

        # remove recipe input item since we are crafting it
        start_inputs[input_item] = 0

        # add the children recipe inputs
        for item, count in children_recipe_inputs.items():
            start_inputs[item] = start_inputs.get(item, 0) + count

    overall_exclude_set = overall_exclude_set - {None}
    start_inputs = {k: v for k, v in start_inputs.items() if v > 0}

    return start_inputs, overall_exclude_set


def remove_ancestor_items(target: str, inventory: dict[str, int]) -> dict[str, int]:
    ancestors = set(get_ancestors(target))
    possible_items = set(inventory.keys())
    items_to_remove = list(ancestors.intersection(possible_items))
    num_items = random.randint(1, len(items_to_remove))
    for item in random.sample(items_to_remove, num_items):
        count_to_remove = random.randint(1, inventory[item])
        inventory[item] -= count_to_remove
        if inventory[item] == 0:
            del inventory[item]
    return inventory


def construct_example(
    target: str,
    num_distractors: 16,
    impossible=False,
) -> list[dict]:
    """
    For a given target object, number of distractors, and impossible flag
    Return a dictionary with the start inventory for the crafting task

    The crafting task should be to craft the target, the inventory should contain
    the resources required for the recipe to be crafted.

    The number of distractors are how many random items should be added to the inventory.

    If impossible is True, the target item should not be craftable with the given inventory.
    """

    # sample the recipe
    inventory, overall_exclude_set = sample_recipes(target, set())
    if impossible:
        # if impossible then remove one or more items from the inventory
        inventory = remove_ancestor_items(
            target,
            inventory,
        )

    # add distractors to the inventory
    distractors = sample_distractors(overall_exclude_set, num_distractors)
    inventory.update(distractors)

    optimal_path = optimal_planner(target, inventory)
    # @TODO this is a hack to ensure that we don't have impossible examples
    while optimal_path is not None and impossible:
        inventory = remove_ancestor_items(target, inventory)
        optimal_path = optimal_planner(target, inventory)

    # assign to slots
    inventory_dict = assign_to_slots(inventory)
    example = {
        "inventory": inventory,
        "slotted_inventory": inventory_dict,
        "target": target,
        "num_distractors": num_distractors,
        "impossible": impossible,
    }
    # either impossible and no path or not impossible and path exists
    assert (impossible and optimal_path is None) or (
        not impossible and optimal_path is not None
    )

    if not impossible:
        example["optimal_path_length"] = len(optimal_path)
        example["optimal_path"] = [r.result.item for (r, i) in optimal_path]
        example["inventory_trace"] = [i for (r, i) in optimal_path]
        items_used, unique_items_used = calculate_stats_from_inventory_trace(
            [example["inventory"]] + example["inventory_trace"]
        )
        example["items_used"] = items_used
        example["unique_items_used"] = unique_items_used

    return example


def calculate_stats_from_inventory_trace(
    inventory_trace: list[dict],
) -> tuple[int, int]:
    total_items_used = 0
    total_unique_items_used = 0

    for a, b in zip(inventory_trace[:-1], inventory_trace[1:]):
        diff = Counter(a) - Counter(b)
        total_items_used += sum(diff.values())
        total_unique_items_used += len(diff)

    return total_items_used, total_unique_items_used


def generate_dataset(seed=2024, distractors=[4, 8, 16], num_examples=10):
    random.seed(seed)
    np.random.seed(seed)

    dataset = []
    for recipe_target in list(RECIPES.keys()):
        if len(RECIPES[recipe_target]) == 0:
            continue
        for num_distractors in distractors:
            for _ in range(num_examples):
                example = construct_example(
                    target=recipe_target, num_distractors=num_distractors
                )
                dataset.append(example)

    return dataset
