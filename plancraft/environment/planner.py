import time

import networkx as nx

from plancraft.environment.recipes import RECIPES, BaseRecipe

RECIPE_GRAPH = nx.DiGraph()

for item, recipes in RECIPES.items():
    for recipe in recipes:
        RECIPE_GRAPH.add_node(recipe.result.item)
        for ingredient in recipe.inputs:
            RECIPE_GRAPH.add_node(ingredient)
            RECIPE_GRAPH.add_edge(ingredient, recipe.result.item)


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
