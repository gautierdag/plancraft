import base64
import glob
import io
import pathlib
import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from plancraft.environments.actions import (
    StopAction,
    SymbolicAction,
    MoveAction,
    SmeltAction,
    convert_from_slot_index,
)
from plancraft.environments.recipes import RECIPES


def numpy_to_base64(img_array: np.ndarray, image_format: str = "PNG") -> str:
    """
    Convert a NumPy array to a base64 encoded string.

    Parameters:
    - img_array: np.ndarray - Input image array.
    - image_format: str - The format to save the image in (e.g., "PNG", "JPEG").

    Returns:
    - str - Base64 encoded string of the image.
    """
    # Convert NumPy array to image
    image = Image.fromarray(img_array)

    # Save the image to a bytes buffer
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)

    # Encode the bytes to a base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


def get_downloaded_models() -> dict:
    """
    Get the list of downloaded models on the NFS partition (EIDF).
    """
    downloaded_models = {}
    # known models on NFS partition
    if pathlib.Path("/nfs").exists():
        local_models = glob.glob("/nfs/public/hf/models/*/*")
        downloaded_models = {
            model.replace("/nfs/public/hf/models/", ""): model for model in local_models
        }
    return downloaded_models


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sequence = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.longest_sequence_length = 0

    def insert(self, sequence: list):
        node = self.root
        for num in sequence:
            if num not in node.children:
                node.children[num] = TrieNode()
            node = node.children[num]
        node.is_end_of_sequence = True

        if len(sequence) > self.longest_sequence_length:
            self.longest_sequence_length = len(sequence)

    def starts_with(self, prefix: list) -> bool:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return False
            node = node.children[num]
        return True

    def get_next(self, prefix: list) -> list:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return []
            node = node.children[num]
        return list(node.children.keys())


def tokenize(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_messages: list[list[dict]],
    start_messages_generation: list[str],
    max_tokens=256,
    images=None,
) -> dict[str, torch.Tensor]:
    """
    Tokenize a list of messages and start the response message
    """
    assert len(start_messages_generation) == len(
        batch_messages
    ), "Length of start_messages_generation should be equal to batch_messages"

    message_texts = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    # add the start of the response message for each message
    message_texts = [
        messages_text + new_message_start
        for (messages_text, new_message_start) in zip(
            message_texts, start_messages_generation
        )
    ]

    max_prompt_length = None
    # need to truncate if max_length is set
    if model.generation_config.max_length > max_tokens:
        max_prompt_length = model.generation_config.max_length - max_tokens

    if images:
        assert len(images) == len(
            batch_messages
        ), "Length of images should be equal to batch_messages"
        tokenized_messages = tokenizer(
            message_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=True,
            images=images,
        )
    else:
        tokenized_messages = tokenizer(
            message_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=True,
        )
    return tokenized_messages


def objective_and_inventory_to_str(objective: str, inventory: list[dict]) -> str:
    inventory_str = ""
    for item in inventory:
        # skip items with quantity 0
        if item["quantity"] <= 0:
            continue
        slot = item["slot"]
        if isinstance(slot, int):
            slot = convert_from_slot_index(slot)
        inventory_str += f"\n - {item['type']} {slot} quantity {item['quantity']}"
    return f"{objective}\ninventory:{inventory_str}"


def convert_observation_to_message(
    observation: dict,
    objective: str,
    bbox_model=None,
    oam_model=False,
    use_text_inventory=True,
    use_multimodal_content_format=False,
    use_images=False,
) -> str | dict:
    """
    Convert an observation to a message format

    Parameters:
    - observation: dict - The observation to convert.
    - objective: str - The objective of the observation.
    - bbox_model: Optional - The bounding box model to use.
    - oam_model: bool - Whether to use the OAM model.
    - use_text_inventory: bool - Whether to use text inventory.
    - use_multimodal_content_format: bool - Whether to use multimodal content format.
    - use_images: bool - Whether to append an image to the message content - must be used with use_multimodal_content_format.
    """
    if bbox_model is not None:
        # convert to tensor
        inventory = bbox_model.get_inventory(observation["pov"].copy())
        text_content = objective_and_inventory_to_str(
            objective, sorted(inventory, key=lambda x: x["slot"])
        )
    elif oam_model:
        text_content = f"{objective}\ninventory:\n"
    elif not use_text_inventory:
        text_content = objective
    else:
        # if not multimodal, we only have text - we just dump a JSON of the inventory
        inventory = []
        for o in observation["inventory"]:
            if o["quantity"] > 0:
                inventory.append(
                    {
                        "type": o["type"],
                        "slot": convert_from_slot_index(o["slot"]),
                        "quantity": o["quantity"],
                    }
                )
        text_content = objective_and_inventory_to_str(objective, inventory)

    if not use_multimodal_content_format:
        return text_content

    content_list = [{"type": "text", "text": text_content}]
    if use_images:
        content_list.append({"type": "image"})
    return {"content": content_list}


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


def parse_content_response(
    content: str, valid_actions: list[str] = ["smelt", "move"]
) -> str | SymbolicAction | StopAction:
    """
    Given a message and set of valid actions, parse the content to return the action
    or a message if the action is not valid/requires message response
    """

    action_match = re.search(f"({'|'.join(valid_actions)}):", content)
    if action_match:
        action = action_match.group(1)
        if action == "think":
            return "Ok"
        elif action == "impossible":
            reason = re.search(r"impossible: (.*)", content).group(1)
            return StopAction(reason=reason)
        elif action == "search":
            search_target = re.search(r"search: (\w+)", content).group(1)
            return gold_search_recipe(search_target)
        else:
            try:
                slot_from = re.search(r" from (\[[ABCI]?\d+\])", content).group(1)
                slot_to = re.search(r" to (\[[ABCI]?\d+\])", content).group(1)
                quantity = re.search(r"with quantity (\d+)", content).group(1)
                if action == "move":
                    action = MoveAction(
                        slot_from=slot_from,
                        slot_to=slot_to,
                        quantity=quantity,
                    )
                else:
                    action = SmeltAction(
                        slot_from=slot_from,
                        slot_to=slot_to,
                        quantity=quantity,
                    )
                return action
            except AttributeError as e:
                return f"Format Error: {e}"
    return f"Only select actions from the following: {', '.join(valid_actions)}"
