import base64
import glob
import io
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


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
