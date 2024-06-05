import logging

from gym import Env
from minerl.herobraine.hero import spaces

from plancraft.environments.actions import SymbolicAction
from plancraft.environments.recipes import (
    RECIPES,
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    convert_ingredients_to_table,
)
from plancraft.environments.sampler import MAX_STACK_SIZE

logger = logging.getLogger(__name__)


class SymbolicPlancraft(Env):
    def __init__(self, inventory: list[dict] = [], recipes=RECIPES, **kwargs):
        self.action_space = spaces.Dict(
            {
                "inventory_command": spaces.Tuple(
                    (
                        spaces.Discrete(46),
                        spaces.Discrete(46),
                        spaces.Discrete(64),
                    )
                ),
                "smelt": spaces.Tuple(
                    (
                        spaces.Discrete(46),
                        spaces.Discrete(46),
                        spaces.Discrete(64),
                    )
                ),
            }
        )

        self.inventory = inventory
        self.reset_state()
        self.table_indexes = list(range(1, 10))
        self.output_index = 0

        self.recipes = recipes

        self.smelting_recipes = []

        self.crafting_recipes = []

        for recipe_list in recipes.values():
            for recipe in recipe_list:
                if isinstance(recipe, SmeltingRecipe):
                    self.smelting_recipes.append(recipe)
                elif isinstance(recipe, (ShapelessRecipe, ShapedRecipe)):
                    self.crafting_recipes.append(recipe)

    def reset_state(self):
        self.state = {i: {"type": "air", "quantity": 0} for i in range(46)}
        # initialise inventory
        for item in self.inventory:
            self.state[item["slot"]] = {
                "type": item["type"],
                "quantity": item["quantity"],
            }

    def step(self, action: SymbolicAction | dict):
        # action_dict = action.to_action_dict()
        if not isinstance(action, dict):
            action = action.to_action_dict()

        if "inventory_command" in action:
            # do inventory command (move)
            slot, slot_to, quantity = action["inventory_command"]
            self.move_item(slot, slot_to, quantity)
        elif "smelt" in action:
            # do smelt
            slot, slot_to, quantity = action["smelt"]
            self.smelt_item(slot, slot_to, quantity)
        else:
            raise ValueError("Invalid action")
            # logger.warn("Cannot parse action for Symbolic action")

        self.clean_state()

        # convert to list for same format as minerl
        state_list = [
            {"type": item["type"], "quantity": item["quantity"], "index": idx}
            for idx, item in self.state.items()
        ]

        return {"inventory": state_list}, 0, False, {}

    def clean_state(self):
        # reset slot type if quantity is 0
        for i in self.state.keys():
            if self.state[i]["quantity"] == 0:
                self.state[i]["type"] = "air"

    def move_item(self, slot_from: int, slot_to: int, quantity: int):
        if slot_from == slot_to or quantity < 1 or slot_to == 0:
            return
        # slot outside of inventory
        if slot_from not in self.state or slot_to not in self.state:
            return
        # not enough
        if self.state[slot_from]["quantity"] < quantity:
            return

        item = self.state[slot_from]

        # slot to is not empty or is the same type as item
        if (self.state[slot_to]["type"] == "air") or (
            self.state[slot_to]["quantity"] <= 0
        ):
            self.state[slot_to] = {"type": item["type"], "quantity": quantity}
            self.state[slot_from]["quantity"] -= quantity
        elif self.state[slot_to]["type"] == item["type"] and (
            MAX_STACK_SIZE[item["type"]] >= self.state[slot_to]["quantity"] + quantity
        ):
            # check if the quantity exceeds the max stack size
            self.state[slot_to]["quantity"] += quantity
            self.state[slot_from]["quantity"] -= quantity
        else:
            return

        # reset slot if quantity is 0
        if self.state[slot_from]["quantity"] == 0:
            self.state[slot_from] = {"type": "air", "quantity": 0}

        # use up ingredients
        if slot_from == 0:
            self.use_ingredients()

        # populate craft slot if ingredients in crafting table have changed
        if slot_to < 10 or slot_from < 10:
            self.populate_craft_slot_craft_item()

    def smelt_item(self, slot_from: int, slot_to: int, quantity: int):
        if quantity < 1 or slot_to == 0 or slot_from == slot_to or slot_from == 0:
            return  # skip if quantity is less than 1

        if slot_from not in self.state or slot_to not in self.state:
            return  # handle slot out of bounds or invalid slot numbers

        item = self.state[slot_from]
        if item["quantity"] < quantity or item["type"] == "air":
            return  # skip if the slot from is empty or does not have enough items

        for recipe in self.smelting_recipes:
            if output := recipe.smelt(item["type"]):
                output_type = output.item
                # Check if the destination slot is empty or has the same type of item as the output
                if self.state[slot_to]["type"] == "air":
                    self.state[slot_to] = {"type": output_type, "quantity": quantity}
                    self.state[slot_from]["quantity"] -= quantity
                    break
                elif self.state[slot_to]["type"] == output_type and (
                    MAX_STACK_SIZE[output_type]
                    >= self.state[slot_to]["quantity"] + quantity
                ):  # assuming max stack size is 64
                    self.state[slot_to]["quantity"] += quantity
                    self.state[slot_from]["quantity"] -= quantity
                    break
                else:
                    return  # No space or type mismatch in slot_to

        # Clean up if the source slot is depleted
        if self.state[slot_from] == 0:
            self.state[slot_from] = {"type": "air", "quantity": 0}

        if slot_to < 10 or slot_from < 10:
            self.populate_craft_slot_craft_item()

    def populate_craft_slot_craft_item(self):
        # get ingredients from crafting table
        ingredients = []
        for i in self.table_indexes:
            if self.state[i]["type"] != "air" and self.state[i]["quantity"] > 0:
                ingredients.append(self.state[i]["type"])
            else:
                ingredients.append(None)
        table = convert_ingredients_to_table(ingredients)

        # check if any of the crafting recipes match the ingredients
        for recipe in self.crafting_recipes:
            if result := recipe.craft(table):
                result, indexes = result
                self.ingredients_idxs = indexes
                self.state[self.output_index] = {
                    "type": result.item,
                    "quantity": result.count,
                }
                return

        self.ingredients_idxs = []
        self.state[self.output_index] = {"type": "air", "quantity": 0}

    def use_ingredients(self):
        # remove used ingredients from crafting table
        for idx in self.ingredients_idxs:
            self.state[idx + 1]["quantity"] -= 1
            if self.state[idx + 1]["quantity"] <= 0:
                self.state[idx + 1] = {"type": "air", "quantity": 0}
        self.ingredients_idxs = []

    def reset(self):
        self.reset_state()
        return self.state

    def fast_reset(self, new_inventory: list[dict]):
        self.inventory = new_inventory
        self.reset_state()
        return self.state

    def render(self):
        print(f"state: {self.state}")
