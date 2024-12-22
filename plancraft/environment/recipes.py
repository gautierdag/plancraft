import glob
import json
import os
import random
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from plancraft.environment.items import ALL_ITEMS


def clean_item_name(item: str) -> str:
    return item.replace("minecraft:", "")


# find directory of file
dir_path = os.path.dirname(os.path.realpath(__file__))
TAG_TO_ITEMS: dict[str, list[str]] = {}
for tag_file in glob.glob(f"{dir_path}/tags/*.json"):
    with open(tag_file) as file:
        tag = json.load(file)
        tag_name = tag_file.split("/")[-1].split(".")[0]
        TAG_TO_ITEMS[tag_name] = [clean_item_name(v) for v in tag["values"]]


def item_to_id(item: str) -> int:
    if item is None:
        return len(ALL_ITEMS)
    return ALL_ITEMS.index(clean_item_name(item))


def id_to_item(item_id: int) -> str:
    if item_id == len(ALL_ITEMS):
        return None
    return ALL_ITEMS[item_id]


def convert_ingredients_to_table(ingredients: list[str]) -> np.array:
    assert len(ingredients) == 9, "Crafting table must have 9 slots"
    table = np.zeros((3, 3), dtype=int)
    for index, item in enumerate(ingredients):
        x, y = divmod(index, 3)
        table[x, y] = item_to_id(item)
    return table


def get_item(item):
    """
    Iterator over the possible items in a recipe object
    """
    if isinstance(item, list):
        for i in item:
            yield from get_item(i)
    if isinstance(item, str):
        if item.startswith("#"):
            tag_items = TAG_TO_ITEMS[clean_item_name(item.replace("#", ""))]
            yield from get_item(tag_items)
        else:
            yield clean_item_name(item)
    if isinstance(item, dict):
        if "item" in item:
            yield clean_item_name(item["item"])
        else:
            tag_items = TAG_TO_ITEMS[clean_item_name(item["tag"])]
            yield from get_item(tag_items)


@dataclass
class RecipeResult:
    item: str
    count: int = 1


class BaseRecipe:
    @abstractmethod
    def craft(self, table: np.array) -> RecipeResult:
        pass

    @abstractmethod
    def smelt(self, ingredient: str) -> RecipeResult:
        pass

    @abstractmethod
    def sample_inputs(self) -> tuple[dict[str, int], set]:
        pass

    @abstractmethod
    def can_craft_from_inventory(self, inventory: dict[str, int]) -> bool:
        pass

    @abstractmethod
    def craft_from_inventory(self, inventory: dict[str, int]) -> dict[str, int]:
        pass

    @property
    def inputs(self) -> set:
        raise NotImplementedError()

    @property
    def recipe_type(self) -> str:
        raise NotImplementedError()

    @property
    def num_slots(self) -> int:
        return NotImplementedError()

    def __repr__(self) -> str:
        pass


class ShapelessRecipe(BaseRecipe):
    def __init__(self, recipe):
        # list of counters that represent the different valid inputs
        self.ingredients: list[dict[str, int]] = []
        self.add_ingredient(recipe["ingredients"], 0, {})

        self.ingredients_arr = np.stack(
            [self.convert_ingredients_counter_to_arr(ing) for ing in self.ingredients],
            axis=0,
        )

        result_item_name = clean_item_name(recipe["result"]["item"])
        self.result = RecipeResult(result_item_name, recipe["result"].get("count", 1))

    def add_ingredient(
        self, ingredients_list: list[dict], index: int, current_counter: dict[str, int]
    ):
        # Base case: all ingredients are processed, add current variant to the list
        if index == len(ingredients_list):
            self.ingredients.append(current_counter)
            return

        ingredient_names = list(get_item(ingredients_list[index]))
        if len(ingredient_names) > 1:
            # If the ingredient has alternatives, recurse for each one
            # This is the case for fire_charge with coal/charcoal
            for item_name in ingredient_names:
                new_counter = deepcopy(current_counter)
                new_counter[item_name] = new_counter.get(item_name, 0) + 1
                self.add_ingredient(ingredients_list, index + 1, new_counter)
        # single acceptable ingredient
        elif len(ingredient_names) == 1:
            item_name = ingredient_names[0]
            current_counter[item_name] = current_counter.get(item_name, 0) + 1
            self.add_ingredient(ingredients_list, index + 1, current_counter)
        else:
            raise ValueError("No item found in ingredient")

    @staticmethod
    def convert_ingredients_counter_to_arr(
        ingredients_counter: dict[str, int],
    ) -> np.array:
        arr = np.zeros(len(ALL_ITEMS) + 1)
        total_slots = 0
        for item, count in ingredients_counter.items():
            arr[item_to_id(item)] = count
            total_slots += count
        # account for empty slots
        arr[len(ALL_ITEMS)] = 9 - total_slots
        return arr

    def craft(self, table: np.array) -> tuple[RecipeResult, list[int]]:
        assert table.shape == (3, 3), "Crafting table must have 3x3 shape"
        table_arr = np.bincount(table.flatten(), minlength=len(ALL_ITEMS) + 1)
        if (table_arr == self.ingredients_arr).all(axis=1).any():
            indexes_to_decrement = []
            for idx, item_id in enumerate(table.flatten()):
                if item_id != len(ALL_ITEMS):
                    indexes_to_decrement.append(idx)
            return self.result, indexes_to_decrement

        return None

    def smelt(self, ingredient: str):
        return None

    def sample_inputs(self) -> tuple[dict[str, int], set]:
        exclude_set = set()
        for ingredient_counter in self.ingredients:
            for ingredient in ingredient_counter.keys():
                if ingredient is not None:
                    exclude_set.add(ingredient)

        # sample a random ingredients set from the list of possible
        return deepcopy(random.choice(self.ingredients)), deepcopy(exclude_set)

    @property
    def inputs(self) -> set:
        all_inputs = set()
        for ingredient_counter in self.ingredients:
            for ingredient in ingredient_counter.keys():
                if ingredient is not None:
                    all_inputs.add(ingredient)
        return all_inputs

    @property
    def recipe_type(self) -> str:
        return "shapeless"

    def can_craft_from_inventory(self, inventory: dict[str, int]) -> bool:
        for ingredient_counter in self.ingredients:
            temp_inventory = deepcopy(inventory)
            for ingredient, count in ingredient_counter.items():
                if (
                    ingredient not in temp_inventory
                    or temp_inventory[ingredient] < count
                ):
                    break
                temp_inventory[ingredient] -= count
            else:
                return True
        return False

    def craft_from_inventory(self, inventory: dict[str, int]) -> dict[str, int]:
        assert self.can_craft_from_inventory(inventory), "Cannot craft from inventory"
        for ingredient_counter in self.ingredients:
            temp_inventory = deepcopy(inventory)
            for ingredient, count in ingredient_counter.items():
                if temp_inventory.get(ingredient, 0) < count:
                    break
                temp_inventory[ingredient] -= count
            else:
                new_inventory = deepcopy(inventory)
                for ingredient, count in ingredient_counter.items():
                    new_inventory[ingredient] -= count
                    if new_inventory[ingredient] == 0:
                        del new_inventory[ingredient]
                new_inventory[self.result.item] = (
                    new_inventory.get(self.result.item, 0) + self.result.count
                )
                return new_inventory

    def __repr__(self):
        return f"ShapelessRecipe({self.ingredients}, {self.result})"

    def __prompt_repr__(self) -> str:
        # use to get a simple representation of the recipe for prompting
        out = []
        for ingredients in self.ingredients:
            ingredients_string = ", ".join(
                [f"{count} {i}" for i, count in ingredients.items()]
            )
            out.append(
                f"{ingredients_string} -> {self.result.count} {self.result.item}"
            )
        return "\n".join(out)

    def sample_input_crafting_grid(self) -> list[dict[str, int]]:
        # sample a random ingredient crafting arrangement to craft item
        ingredients = deepcopy(random.choice(self.ingredients))

        num_inputs = sum(ingredients.values())
        crafting_table_slots = random.sample(range(1, 10), num_inputs)

        ingredients_list = []
        for i, count in ingredients.items():
            ingredients_list += [i] * count

        crafting_table = []
        for ingredient, slot in zip(ingredients_list, crafting_table_slots):
            if ingredient is not None:
                crafting_table.append({"type": ingredient, "slot": slot, "quantity": 1})

        return crafting_table


class ShapedRecipe(BaseRecipe):
    def __init__(self, recipe):
        self.kernel = self.extract_kernel(recipe)

        self.kernel_height = len(self.kernel)
        self.kernel_width = len(self.kernel[0])
        self.kernel_size = self.kernel_height * self.kernel_width

        result_item_name = clean_item_name(recipe["result"]["item"])
        self.result = RecipeResult(result_item_name, recipe["result"].get("count", 1))

    def possible_kernel_positions(self):
        return [
            (row, col)
            for row in range(3 - self.kernel_height + 1)
            for col in range(3 - self.kernel_width + 1)
        ]

    def extract_kernel(self, recipe) -> np.array:
        patterns = recipe["pattern"]
        keys = recipe["key"]

        # Convert pattern symbols to corresponding items ids
        def symbol_to_items(symbol):
            if symbol == " ":
                return set([item_to_id(None)])
            return set([item_to_id(item) for item in get_item(keys[symbol])])

        # Convert each row of the pattern to a list of possible item lists
        kernel = [[symbol_to_items(symbol) for symbol in row] for row in patterns]
        return kernel

    def craft(self, table: np.array) -> tuple[RecipeResult, list[int]]:
        assert table.shape == (3, 3), "Crafting table must have 3x3 shape"

        count_empty = (table == len(ALL_ITEMS)).sum()
        should_be_empty_count = 9 - self.kernel_size
        if count_empty < should_be_empty_count:
            return None

        for start_row, start_col in self.possible_kernel_positions():
            # count number of empty slots in kernel
            count_empty_in_kernel = (
                table[
                    start_row : start_row + self.kernel_height,
                    start_col : start_col + self.kernel_width,
                ]
                == len(ALL_ITEMS)
            ).sum()
            # count number of empty slots outside
            count_empty_outside_kernel = count_empty - count_empty_in_kernel
            # check if the number of empty slots outside is correct
            if count_empty_outside_kernel != should_be_empty_count:
                continue

            # check that all items in kernel match the table
            indexes_to_decrement = []
            for row in range(self.kernel_height):
                for col in range(self.kernel_width):
                    if (
                        table[start_row + row, start_col + col]
                        not in self.kernel[row][col]
                    ):
                        break
                    # add to indexes to decrement if not empty slot
                    if table[start_row + row, start_col + col] != len(ALL_ITEMS):
                        idx = (start_row + row) * 3 + start_col + col
                        indexes_to_decrement.append(idx)
                else:
                    continue
                break
            else:
                return self.result, indexes_to_decrement
        return None

    def smelt(self, ingredient: str):
        return None

    def sample_inputs(self) -> tuple[dict[str, int], set]:
        input_counter = defaultdict(int)
        exclude_set = set()
        for row in self.kernel:
            for item_set in row:
                # sample a random item from the set
                if len(item_set) > 1:
                    item_id = random.choice(list(item_set))
                    # add all items to exclude set
                    exclude_set.update(item_set)
                # single acceptable ingredient
                else:
                    item_id = list(item_set)[0]
                    exclude_set.add(item_id)

                # exclude empty slot
                if id_to_item(item_id) is not None:
                    input_counter[id_to_item(item_id)] += 1

        # convert exclude set to item names
        exclude_set = {
            id_to_item(item_id) for item_id in exclude_set if id_to_item(item_id)
        }
        return dict(input_counter), exclude_set

    @property
    def inputs(self) -> set:
        all_inputs = set()
        for row in self.kernel:
            for item_set in row:
                all_inputs.update(item_set)
        all_inputs = {
            id_to_item(item_id) for item_id in all_inputs if id_to_item(item_id)
        }
        return all_inputs

    @property
    def recipe_type(self) -> str:
        return "shaped"

    def can_craft_from_inventory(self, inventory: dict[str, int]) -> bool:
        temp_inventory = deepcopy(inventory)
        for row in self.kernel:
            for item_set in row:
                for inventory_item in item_set:
                    item_name = id_to_item(inventory_item)
                    if item_name is None:
                        break
                    elif item_name in temp_inventory and temp_inventory[item_name] > 0:
                        temp_inventory[item_name] -= 1
                        break
                else:
                    return False
        return True

    def craft_from_inventory(self, inventory: dict[str, int]) -> dict[str, int]:
        assert self.can_craft_from_inventory(inventory), "Cannot craft from inventory"
        new_inventory = deepcopy(inventory)
        for row in self.kernel:
            for item_set in row:
                for inventory_item in item_set:
                    item_name = id_to_item(inventory_item)
                    if (
                        item_name
                        and item_name in new_inventory
                        and new_inventory[item_name] > 0
                    ):
                        new_inventory[item_name] -= 1
                        if new_inventory[item_name] == 0:
                            del new_inventory[item_name]
                        break
        # add result to inventory
        new_inventory[self.result.item] = (
            new_inventory.get(self.result.item, 0) + self.result.count
        )
        return new_inventory

    def __repr__(self) -> str:
        return f"ShapedRecipe({self.kernel}, {self.result})"

    def __prompt_repr__(self) -> str:
        string_kernel = []
        for row in self.kernel:
            row_col = []
            for col in row:
                valid_items = [str(id_to_item(i)) for i in col]
                if valid_items[0] == "None":
                    valid_items[0] = "empty"
                row_col.append("|".join(valid_items))
            string_kernel.append("\t".join(row_col))
        result_row = len(self.kernel) // 2
        string_kernel[result_row] = (
            string_kernel[result_row] + f" -> {self.result.count} {self.result.item}"
        )
        return "\n".join(string_kernel)

    def sample_input_crafting_grid(self) -> list[dict[str, int]]:
        # sample a random ingredient crafting arrangement to craft item
        crafting_table = []

        start_row, start_col = random.choice(self.possible_kernel_positions())
        for x in range(0, len(self.kernel)):
            for y in range(0, len(self.kernel[0])):
                i = random.choice(list(self.kernel[x][y]))
                object_type = id_to_item(i)
                if object_type is None:
                    continue
                location_x = start_row + x
                location_y = start_col + y
                crafting_table += [
                    {
                        "type": object_type,
                        "slot": location_x * 3 + location_y + 1,
                        "quantity": 1,
                    }
                ]
        return crafting_table


class SmeltingRecipe(BaseRecipe):
    def __init__(self, recipe):
        self.ingredient = set(get_item(recipe["ingredient"]))
        result_item_name = clean_item_name(recipe["result"])
        self.result = RecipeResult(result_item_name, 1)

    def smelt(self, ingredient: str) -> RecipeResult:
        if ingredient in self.ingredient:
            return self.result
        return None

    def craft(self, table: np.array):
        return None

    def can_craft_from_inventory(self, inventory: dict[str, int]) -> bool:
        return len(set(inventory.keys()).intersection(self.ingredient)) > 0

    def craft_from_inventory(
        self, inventory: dict[str, int], quantity=1
    ) -> dict[str, int]:
        assert self.can_craft_from_inventory(inventory), "Cannot craft from inventory"

        new_inventory = deepcopy(inventory)
        for ingredient in self.ingredient:
            if ingredient in new_inventory and new_inventory[ingredient] >= quantity:
                new_inventory[ingredient] -= quantity
                if new_inventory[ingredient] == 0:
                    del new_inventory[ingredient]
                new_inventory[self.result.item] = (
                    new_inventory.get(self.result.item, 0) + quantity  # count
                )
                return new_inventory
        return None

    def sample_inputs(self) -> tuple[dict[str, int], set]:
        # sample a random ingredient from the list of possible
        # return a dict with the ingredient and its count
        # also return the set of all possible ingredients
        return {random.choice(deepcopy(list(self.ingredient))): 1}, deepcopy(
            self.ingredient
        )

    @property
    def inputs(self) -> set:
        return deepcopy(self.ingredient)

    @property
    def recipe_type(self) -> str:
        return "smelting"

    def __repr__(self) -> str:
        return f"SmeltingRecipe({self.ingredient}, {self.result})"

    def __prompt_repr__(self) -> str:
        # use to get a simple representation of the recipe for prompting
        out = []
        for i in self.ingredient:
            out.append(f"1 {i} -> {self.result.count} {self.result.item}")
        return "\n".join(out)


def recipe_factory(recipe: dict) -> BaseRecipe:
    if recipe["type"] == "minecraft:crafting_shapeless":
        return ShapelessRecipe(recipe)
    if recipe["type"] == "minecraft:crafting_shaped":
        return ShapedRecipe(recipe)
    if recipe["type"] == "minecraft:smelting":
        return SmeltingRecipe(recipe)


RECIPES: dict[str, list[BaseRecipe]] = defaultdict(list)
for f in glob.glob(f"{dir_path}/recipes/*.json"):
    with open(f) as file:
        recipe = json.load(file)
        if r := recipe_factory(recipe):
            RECIPES[r.result.item].append(r)
