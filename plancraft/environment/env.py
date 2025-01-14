import glob
import os
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plancraft.environment.actions import SymbolicAction, convert_from_slot_index
from plancraft.environment.recipes import (
    RECIPES,
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    convert_ingredients_to_table,
)
from plancraft.environment.sampler import MAX_STACK_SIZE


def get_objective_str(target: str) -> str:
    return f"Craft an item of type: {target}"


def target_and_inventory_to_text_obs(target: str, inventory: dict) -> str:
    """
    Convert inventory dict to text observation
    """
    objective = get_objective_str(target)
    inventory_str = ""
    # sort by slot number
    for slot, item in sorted(inventory.items(), key=lambda item: item[0]):
        # skip items with quantity 0
        if item["quantity"] <= 0:
            continue
        slot_str = convert_from_slot_index(slot)
        inventory_str += f"\n - {item['type']} {slot_str} quantity {item['quantity']}"
    return f"{objective}\ninventory:{inventory_str}"


def text_obs_to_objective_and_inventory(
    text_observation: str,
) -> tuple[str, list[dict]]:
    """
    Parser for text observation back to target object and inventory dict
    """
    raise NotImplementedError()


class CraftingTableGUI:
    """
    Class to render a crafting table with items and quantities.
    """

    def __init__(
        self,
        resolution: Literal["low", "medium", "high"] = "high",
    ):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.assets_dir = f"{dir_path}/assets"
        self.table_file = f"{self.assets_dir}/table.png"
        self.resolution = resolution
        self.multiple = 4
        self.icon_size = 64
        self.calculate_crafting_slot_locations()

        # Use a context manager to load the base table
        with Image.open(self.table_file) as base_table:
            self.base_table = base_table.copy()

        # Load all overlay images safely
        self.overlay_images = {}
        for filename in glob.glob(f"{self.assets_dir}/images/*.png"):
            with Image.open(filename) as img:
                self.overlay_images[os.path.basename(filename).split(".")[0]] = (
                    img.copy()
                )

        self.font = ImageFont.truetype(f"{self.assets_dir}/minecraft_font.ttf", 40)
        self.current_table = self.base_table.copy()

    def calculate_crafting_slot_locations(self):
        """
        Calculates the coordinates of each slot in the crafting table.
        """
        self.crafting_slot_locations = [(124 * self.multiple, 35 * self.multiple)]
        start_x = 30 * self.multiple
        start_y = 17 * self.multiple
        offset = self.icon_size + (2 * self.multiple)
        for i in range(3):
            for j in range(3):
                self.crafting_slot_locations.append(
                    (
                        start_x + j * offset,
                        start_y + i * offset,
                    )
                )
        start_x = 8 * self.multiple
        start_y = 84 * self.multiple
        for i in range(3):
            for j in range(9):
                self.crafting_slot_locations.append(
                    (start_x + j * offset, start_y + i * offset)
                )
        start_y = start_y + 3 * offset + 4 * self.multiple
        for j in range(9):
            self.crafting_slot_locations.append((start_x + j * offset, start_y))

    def add_quantity(self, quantity: int, slot_index: int):
        """
        Adds a number with a shadow effect to an image at the specified coordinates.
        Parameters:
        - image: PIL.Image object
        - quantity: int, the number to overlay (1 <= number <= 64)
        - font_path: str, path to the Minecraft font (.ttf)
        - font_size: int, size of the font
        Returns:
        - image: PIL.Image object with the overlay applied
        """
        if quantity <= 1:
            return
        x, y = self.crafting_slot_locations[slot_index]
        # offset the coordinates
        x = x + 4 * self.multiple + 2
        y = y + 8 * self.multiple

        draw = ImageDraw.Draw(self.current_table)
        # Define the shadow and text color
        shadow_color = (62, 62, 62)  # Shadow color
        text_color = (252, 252, 252)  # Main text color

        if quantity < 10:
            # single digit - offset by 4 pixels
            x += 6 * self.multiple
        # Add shadow (offset by 1 pixel)
        draw.text(
            (x + 1 * self.multiple, y + 1 * self.multiple),
            str(quantity),
            font=self.font,
            fill=shadow_color,
        )
        # Add main text
        draw.text((x, y), str(quantity), font=self.font, fill=text_color)

    def overlay_image(self, overlay_image, x, y):
        # Resize overlay image to fit the cell if necessary
        overlay_image = overlay_image
        # Paste the image onto the base image
        self.current_table.paste(overlay_image, (x, y), overlay_image)

    def add_item_to_slot(self, item_name: str, slot: int, quantity: int = 1):
        if quantity < 1 or quantity > 64:
            return

        # clear the slot first
        self.remove_item_from_slot(slot)

        # overlay the item image
        x, y = self.crafting_slot_locations[slot]
        self.overlay_image(self.overlay_images[item_name], x, y)

        # overlay the quantity
        if quantity > 1:
            self.add_quantity(quantity, slot)

    def clear(self):
        self.current_table = self.base_table.copy()

    def save(self, path):
        self.current_table.save(path)

    @property
    def frame(self) -> np.ndarray:
        latest_frame = self.current_table.copy()
        # scale down if necessary
        if self.resolution == "low":
            latest_frame = latest_frame.resize(
                (
                    self.current_table.size[0] // 4,
                    self.current_table.size[1] // 4,
                ),
                Image.NEAREST,
            )
        if self.resolution == "medium":
            latest_frame = latest_frame.resize(
                (
                    self.current_table.size[0] // 2,
                    self.current_table.size[1] // 2,
                ),
                Image.NEAREST,
            )
        return np.array(latest_frame.convert("RGB"))

    def remove_item_from_slot(self, slot: int):
        """
        Removes an item from the specified slot by clearing the corresponding area on the current table.
        """
        if slot == 0:
            x, y = self.crafting_slot_locations[slot]
            icon_size = self.icon_size + self.multiple
            blank_area = Image.new("RGB", (icon_size, icon_size), (139, 139, 139))
            self.current_table.paste(blank_area, (x, y))
        else:
            # Get the coordinates of the slot
            x, y = self.crafting_slot_locations[slot]
            # Create a white rectangle to cover the slot area
            # this will cover the quantity text as well and reset the border
            # Assuming the icon size is 64x64 pixels
            icon_size = self.icon_size + self.multiple
            blank_area = Image.new("RGB", (icon_size, icon_size), (255, 255, 255))
            self.current_table.paste(blank_area, (x, y))
            # create a background color for the slot and cover item
            icon_size = self.icon_size
            blank_area = Image.new("RGB", (icon_size, icon_size), (139, 139, 139))
            # Paste the blank area over the slot
            self.current_table.paste(blank_area, (x, y))

    @property
    def size(self):
        if self.resolution == "low":
            return self.current_table.size[0] // 4, self.current_table.size[1] // 4
        elif self.resolution == "medium":
            return self.current_table.size[0] // 2, self.current_table.size[1] // 2
        return self.current_table.size


class PlancraftEnvironment:
    """
    Environment class for the Plancraft environment.
    """

    def __init__(
        self, inventory: Optional[dict] = None, recipes=RECIPES, resolution="high"
    ):
        self.table = CraftingTableGUI(resolution=resolution)
        self.state = {}

        self.reset(inventory)

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

    def add_item_to_slot(self, item_name: str, slot: int, quantity: int = 1):
        # update visual representation
        self.table.add_item_to_slot(item_name, slot, quantity)
        # add item to state
        self.state[slot] = {"type": item_name, "quantity": quantity}

    def remove_item_from_slot(self, slot: int):
        self.table.remove_item_from_slot(slot)
        if slot in self.state:
            del self.state[slot]  # Remove the slot from the dictionary

    def change_quantity_in_slot(self, slot: int, quantity: int = 1):
        if quantity == 0:
            self.remove_item_from_slot(slot)
            return

        assert 0 < quantity <= 64
        assert slot in self.state, "Slot does not exist in the inventory"

        self.state[slot]["quantity"] = quantity
        # update visual representation by removing and adding the item with the new quantity
        self.table.remove_item_from_slot(slot)
        self.table.add_item_to_slot(self.state[slot]["type"], slot, quantity)

    def reset(self, new_inventory: Optional[dict] = None):
        self.table.clear()
        self.state = {}  # Start with an empty dictionary
        # Initialise inventory
        if new_inventory:
            for slot, item in new_inventory.items():
                assert "type" in item and "quantity" in item
                assert slot >= 0 and slot < 46
                assert item["quantity"] > 0 and item["quantity"] <= 64
                self.state[slot] = item
                self.table.add_item_to_slot(item["type"], slot, item["quantity"])
        return self.state

    def step(self, action: Optional[SymbolicAction] = None) -> dict:
        # default no op action
        if action is None:
            return {"inventory": dict(self.state), "image": self.table.frame}

        if action.action_type == "move":
            # do inventory command (move)
            self.move_item(action.slot_from, action.slot_to, action.quantity)
        elif action.action_type == "smelt":
            # do smelt
            self.smelt_item(action.slot_from, action.slot_to, action.quantity)
        else:
            raise ValueError("Invalid action")

        self.clean_state()
        return {"inventory": dict(self.state), "image": self.table.frame}

    def clean_state(self):
        # reset slot type if quantity is 0
        to_remove = [slot for slot, item in self.state.items() if item["quantity"] <= 0]
        for slot in to_remove:
            self.remove_item_from_slot(slot)

    def slot_empty(self, slot: int) -> bool:
        return slot not in self.state or self.state[slot]["quantity"] <= 0

    def move_item(self, slot_from: int, slot_to: int, quantity: int):
        if slot_from == slot_to or quantity < 1:
            return
        # slot outside of inventory
        if slot_from < 0 or slot_to < 1 or slot_from >= 46 or slot_to >= 46:
            return
        # not enough
        if self.slot_empty(slot_from) or self.state[slot_from]["quantity"] < quantity:
            return
        # if craft slot - must take all
        if slot_from == 0 and self.state[slot_from]["quantity"] != quantity:
            return

        item = self.state[slot_from]

        # slot to is not empty or is the same type as item
        if self.slot_empty(slot_to):
            # add quantity to new slot
            self.add_item_to_slot(item["type"], slot_to, quantity)
            # remove quantity from old slot
            self.change_quantity_in_slot(
                slot_from, self.state[slot_from]["quantity"] - quantity
            )
        elif self.state[slot_to]["type"] == item["type"] and (
            MAX_STACK_SIZE[item["type"]] >= self.state[slot_to]["quantity"] + quantity
        ):
            # check if the quantity exceeds the max stack size
            self.change_quantity_in_slot(
                slot_to, self.state[slot_to]["quantity"] + quantity
            )
            self.change_quantity_in_slot(
                slot_from, self.state[slot_from]["quantity"] - quantity
            )
        else:
            return

        # reset slot if quantity is 0
        if self.slot_empty(slot_from):
            self.remove_item_from_slot(slot_from)

        # use up ingredients
        if slot_from == 0:
            self.use_ingredients()

        # populate craft slot if ingredients in crafting table have changed
        if slot_to < 10 or slot_from < 10:
            self.populate_craft_slot_craft_item()

    def smelt_item(self, slot_from: int, slot_to: int, quantity: int):
        if quantity < 1 or slot_from == slot_to:
            return  # skip if quantity is less than 1

        # handle slot out of bounds or invalid slot numbers
        if slot_from < 1 or slot_to < 1 or slot_from >= 46 or slot_to >= 46:
            return

        item = self.state[slot_from]
        if self.slot_empty(slot_from) or item["quantity"] < quantity:
            return  # skip if the slot from is empty or does not have enough items

        for recipe in self.smelting_recipes:
            if output := recipe.smelt(item["type"]):
                output_type = output.item
                # Check if the destination slot is empty or has the same type of item as the output
                if self.slot_empty(slot_to):
                    self.add_item_to_slot(output_type, slot_to, quantity)
                    self.change_quantity_in_slot(
                        slot_from, self.state[slot_from]["quantity"] - quantity
                    )
                    break
                elif self.state[slot_to]["type"] == output_type and (
                    MAX_STACK_SIZE[output_type]
                    >= self.state[slot_to]["quantity"] + quantity
                ):
                    self.change_quantity_in_slot(
                        slot_to, self.state[slot_to]["quantity"] + quantity
                    )
                    self.change_quantity_in_slot(
                        slot_from, self.state[slot_from]["quantity"] - quantity
                    )
                    break
                else:
                    return  # No space or type mismatch in slot_to

        # Clean up if the source slot is depleted
        if self.slot_empty(slot_from):
            self.remove_item_from_slot(slot_from)

        if slot_to < 10 or slot_from < 10:
            self.populate_craft_slot_craft_item()

    def populate_craft_slot_craft_item(self):
        # get ingredients from crafting table
        ingredients = []
        for i in self.table_indexes:
            if self.slot_empty(i):
                ingredients.append(None)
            else:
                ingredients.append(self.state[i]["type"])
        table = convert_ingredients_to_table(ingredients)

        # check if any of the crafting recipes match the ingredients
        for recipe in self.crafting_recipes:
            if result := recipe.craft(table):
                result, indexes = result
                self.ingredients_idxs = indexes
                self.add_item_to_slot(result.item, self.output_index, result.count)
                return

        self.ingredients_idxs = []
        # no match, clear the output slot
        self.remove_item_from_slot(self.output_index)

    def use_ingredients(self):
        # remove used ingredients from crafting table
        for idx in self.ingredients_idxs:
            self.state[idx + 1]["quantity"] -= 1
            if self.state[idx + 1]["quantity"] <= 0:
                self.remove_item_from_slot(idx + 1)

        self.ingredients_idxs = []
