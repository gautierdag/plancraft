from plancraft.environments.recipes import RECIPES


def gold_search_recipe(recipe_name: str) -> str:
    """
    Gold search recipe for the given observation and action
    """
    if recipe_name not in RECIPES:
        return "Could not find a recipe by that name."

    out_string = f"Recipes to craft {recipe_name}:\n"
    for r in RECIPES[recipe_name]:
        # @TODO change
        out_string += r.__prompt_repr__() + "\n"
    return out_string
