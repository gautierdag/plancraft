import pytest
from plancraft.environment.recipes import (
    RECIPES,
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    convert_ingredients_to_table,
    id_to_item,
)


@pytest.fixture
def shaped_recipe_json():
    return {
        "type": "minecraft:crafting_shaped",
        "group": "boat",
        "pattern": ["# #", "###"],
        "key": {"#": {"item": "minecraft:acacia_planks"}},
        "result": {"item": "minecraft:acacia_boat"},
    }


@pytest.fixture
def shapeless_recipe_json():
    return {
        "type": "minecraft:crafting_shapeless",
        "ingredients": [
            {"item": "minecraft:bowl"},
            {"item": "minecraft:beetroot"},
            {"item": "minecraft:beetroot"},
            {"item": "minecraft:beetroot"},
            {"item": "minecraft:beetroot"},
            {"item": "minecraft:beetroot"},
            {"item": "minecraft:beetroot"},
        ],
        "result": {"item": "minecraft:beetroot_soup"},
    }


@pytest.fixture
def smelting_recipe_json():
    return {
        "type": "minecraft:smelting",
        "ingredient": {"item": "minecraft:quartz_block"},
        "result": "minecraft:smooth_quartz",
        "experience": 0.1,
        "cookingtime": 200,
    }


def test_convert_ingredients_to_table():
    ingredients = [None] * 9
    ingredients[4] = "acacia_planks"
    table = convert_ingredients_to_table(ingredients)
    assert table.shape == (3, 3)
    assert id_to_item(table[0, 1]) is None
    assert id_to_item(table[1, 1]) == "acacia_planks"


def test_shaped_recipe(shaped_recipe_json):
    recipe = ShapedRecipe(shaped_recipe_json)
    assert recipe.result.item == "acacia_boat"
    assert recipe.result.count == 1
    assert recipe.inputs == {"acacia_planks"}


def test_shapeless_recipe(shapeless_recipe_json):
    recipe = ShapelessRecipe(shapeless_recipe_json)
    assert recipe.result.item == "beetroot_soup"
    assert recipe.result.count == 1
    assert recipe.inputs == {"bowl", "beetroot"}


def test_smelting_recipe(smelting_recipe_json):
    recipe = SmeltingRecipe(smelting_recipe_json)
    assert recipe.result.item == "smooth_quartz"
    assert recipe.result.count == 1
    assert recipe.inputs == {"quartz_block"}


def test_craft_from_inventory_shaped():
    recipe = RECIPES["acacia_boat"][0]
    inventory = {"acacia_planks": 6}
    new_inventory = recipe.craft_from_inventory(inventory)
    assert new_inventory == {"acacia_planks": 1, "acacia_boat": 1}


def test_craft_from_table_shaped():
    recipe = RECIPES["acacia_boat"][0]
    ingredients = [None] * 9
    table = convert_ingredients_to_table(ingredients)
    result = recipe.craft(table)
    assert result is None

    ingredients[0] = "acacia_planks"
    ingredients[2] = "acacia_planks"
    ingredients[3] = "acacia_planks"
    ingredients[4] = "acacia_planks"
    ingredients[5] = "acacia_planks"
    table = convert_ingredients_to_table(ingredients)
    result, indexes = recipe.craft(table)
    assert result.item == "acacia_boat"
    assert result.count == 1
    assert indexes == [0, 2, 3, 4, 5]


def test_sample_inputs_shaped():
    recipe = RECIPES["acacia_boat"][0]
    inputs, exclude_set = recipe.sample_inputs()
    assert inputs == {"acacia_planks": 5}
    assert exclude_set == {"acacia_planks"}


def test_craft_from_inventory_shapeless():
    recipe = RECIPES["beetroot_soup"][0]
    inventory = {"bowl": 1, "beetroot": 7}
    new_inventory = recipe.craft_from_inventory(inventory)
    assert new_inventory == {"beetroot": 1, "beetroot_soup": 1}


def test_craft_from_table_shapeless():
    recipe = RECIPES["beetroot_soup"][0]
    ingredients = [None] * 9
    table = convert_ingredients_to_table(ingredients)
    result = recipe.craft(table)
    assert result is None

    ingredients[0] = "bowl"
    ingredients[1] = "beetroot"
    ingredients[2] = "beetroot"
    ingredients[3] = "beetroot"
    ingredients[4] = "beetroot"
    ingredients[5] = "beetroot"
    ingredients[6] = "beetroot"

    table = convert_ingredients_to_table(ingredients)
    result, indexes = recipe.craft(table)
    assert result.item == "beetroot_soup"
    assert result.count == 1
    assert indexes == [0, 1, 2, 3, 4, 5, 6]

    ingredients[7] = "beetroot"
    table = convert_ingredients_to_table(ingredients)
    result = recipe.craft(table)
    assert result is None


def test_sample_inputs_shapeless():
    recipe = RECIPES["beetroot_soup"][0]
    inputs, exclude_set = recipe.sample_inputs()
    assert inputs == {"bowl": 1, "beetroot": 6}
    assert exclude_set == {"bowl", "beetroot"}


def test_craft_from_inventory_smelting():
    recipe = RECIPES["smooth_quartz"][0]
    inventory = {"quartz_block": 1}
    new_inventory = recipe.craft_from_inventory(inventory)
    assert new_inventory == {"smooth_quartz": 1}


def test_smelt():
    recipe = RECIPES["smooth_quartz"][0]
    result = recipe.smelt("")
    assert result is None
    result = recipe.smelt("quartz_block")
    assert result.item == "smooth_quartz"
    assert result.count == 1


def test_sample_inputs_smelting():
    recipe = RECIPES["smooth_quartz"][0]
    inputs, exclude_set = recipe.sample_inputs()
    assert inputs == {"quartz_block": 1}
    assert exclude_set == {"quartz_block"}
