from plancraft.environment.actions import convert_from_slot_index
from plancraft.environment.recipes import RECIPES


def gold_search_recipe(recipe_name: str) -> str:
    """
    Gold search recipe for the given observation and action
    """
    if recipe_name not in RECIPES:
        return "Could not find a recipe by that name."

    out_string = f"Recipes to craft {recipe_name}:\n"
    for i, r in enumerate(RECIPES[recipe_name]):
        if r.recipe_type != "smelting":
            # sample a valid input grid (note that this is not guaranteed to be the only valid grid)
            input_crafting_grid = r.sample_input_crafting_grid()
            recipe_instructions = ""
            for item in input_crafting_grid:
                recipe_instructions += (
                    f"{item['type']} at {convert_from_slot_index(item['slot'])}\n"
                )
        else:
            # smelting recipe
            recipe_instructions = f"smelt {r.ingredient}\n"
        out_string += f"recipe {i+1}:\n{recipe_instructions}"
    return out_string
