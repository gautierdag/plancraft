import glob
import pathlib
from collections import Counter
from copy import copy
from typing import Optional

import torch
from loguru import logger

from plancraft.environment.actions import ActionHandlerBase
from plancraft.environment.prompts import (
    get_prompt_example,
    get_system_prompt,
    load_prompt_images,
)


class History:
    """
    History class to keep track of dialogue, actions, inventory and images
    Args:
        valid_actions: list of valid actions names
        initial_dialogue: list of dialogue messages
        use_multimodal_content_format: whether to use multimodal content format (list of content with types)
    """

    def __init__(
        self,
        actions: list[ActionHandlerBase] = [],
        use_multimodal_content_format=False,
        few_shot=False,
        use_images=False,
        use_text_inventory=False,
        resolution="high",
        system_prompt: Optional[dict] = None,
        prompt_examples: list[dict] = [],
        prompt_images: list[str] = [],
    ):
        self.action_handlers = actions
        self.use_multimodal_content_format = use_multimodal_content_format
        self.few_shot = few_shot
        self.use_images = use_images
        self.use_text_inventory = use_text_inventory
        self.resolution = resolution  # low, medium, high

        self.inventory_history = []
        self.tokens_used = 0

        # use system prompt if provided
        if system_prompt:
            self.system_prompt_dialogue = system_prompt
        else:
            # generate system prompt
            self.system_prompt_dialogue = get_system_prompt(
                handlers=self.action_handlers,
                use_multimodal_content_format=self.use_multimodal_content_format,
            )

        # set up dialogue history with few-shot prompt
        self.prompt_examples = prompt_examples
        self.prompt_images = prompt_images
        self.set_up_few_shot_prompt()

        self.dialogue_history = copy(self.prompt_examples)
        self.images = copy(self.prompt_images)
        self.initial_dialogue_length = len(self.dialogue_history)

    def set_up_few_shot_prompt(self):
        # if either prompt_examples or prompt_images are provided, skip
        if self.prompt_examples or self.prompt_images:
            return
        # if few-shot is not enabled, skip
        if self.few_shot:
            self.prompt_examples = get_prompt_example(
                self.action_handlers,
                use_text_inventory=self.use_text_inventory,
                use_multimodal_content_format=self.use_multimodal_content_format,
                use_images=self.use_images,
            )
            if self.use_images:
                self.prompt_images = load_prompt_images(resolution=self.resolution)

    def add_message_to_history(self, content: str | dict, role="user"):
        if role == "assistant":
            logger.info(content)

        if isinstance(content, dict):
            assert "content" in content, "content key not found in message"
            content["role"] = role
            self.dialogue_history.append(content)
        else:
            # fix for listed content type
            if self.use_multimodal_content_format:
                return self.add_message_to_history(
                    content={
                        "content": [{"type": "text", "text": content}],
                        "role": role,
                    },
                    role=role,
                )
            else:
                self.dialogue_history.append({"role": role, "content": content})

    def add_inventory_to_history(self, inventory: dict):
        self.inventory_history.append(inventory)

    def add_image_to_history(self, image):
        self.images.append(image)

    def add_observation_to_history(self, observation: dict):
        if observation is None:
            return
        if "inventory" in observation:
            self.add_inventory_to_history(observation["inventory"])
        if "image" in observation:
            self.add_image_to_history(observation["image"])

    def __str__(self):
        return str(self.dialogue_history)

    def reset(self):
        # reset dialogue history to few-shot prompt
        self.dialogue_history = copy(self.prompt_examples)
        self.images = copy(self.prompt_images)
        self.initial_dialogue_length = len(self.dialogue_history)

        self.inventory_history = []

        self.tokens_used = 0

    def trace(self):
        return {
            "dialogue_history": copy(
                self.dialogue_history[self.initial_dialogue_length :]
            ),
            "inventory_history": copy(self.inventory_history),
            "tokens_used": copy(self.tokens_used),
        }

    @property
    def num_steps(self):
        return (len(self.dialogue_history) - self.initial_dialogue_length) // 2


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


def get_torch_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.info(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            device = torch.device("mps")
    return device
