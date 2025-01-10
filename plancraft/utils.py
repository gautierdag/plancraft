import glob
import pathlib
from collections import Counter
from copy import copy

import torch
from loguru import logger

from plancraft.environment.actions import (
    ActionHandlerBase,
    MoveAction,
    SmeltAction,
)
from plancraft.environment.prompts import (
    get_system_prompt,
    get_prompt_example,
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
    ):
        self.action_handlers = actions
        self.use_multimodal_content_format = use_multimodal_content_format
        self.few_shot = few_shot
        self.use_images = use_images
        self.use_text_inventory = use_text_inventory
        self.resolution = resolution  # low, medium, high

        self.action_history = []
        self.inventory_history = []
        self.inventory_counters = []

        self.tokens_used = 0

        # set up dialogue history with few-shot prompt
        self.set_up_few_shot_prompt()
        self.system_prompt_dialogue = self.system_prompt()

        self.dialogue_history = copy(self.prompt_examples)
        self.images = copy(self.prompt_images)
        self.initial_dialogue_length = len(self.dialogue_history)

    def system_prompt(self):
        # kept separate from dialogue history because certain models deal with system prompt differently
        system_prompt_text = get_system_prompt(handlers=self.action_handlers)
        if self.use_multimodal_content_format:
            return {
                "role": "system",
                "content": [{"text": system_prompt_text, "type": "text"}],
            }
        return {
            "role": "system",
            "content": system_prompt_text,
        }

    def set_up_few_shot_prompt(self):
        self.prompt_examples = []
        self.prompt_images = []

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

    def add_action_to_history(self, action: SmeltAction | MoveAction):
        if isinstance(action, SmeltAction) or isinstance(action, MoveAction):
            self.action_history.append(action.model_dump())

    def add_inventory_to_history(self, inventory: dict):
        self.inventory_history.append(inventory)
        # count inventory
        counter = Counter()
        for slot, item in inventory.items():
            # ignore slot 0
            if slot == 0:
                continue
            counter[item["type"]] += item["quantity"]
        self.inventory_counters.append(counter)

    def add_image_to_history(self, image):
        self.images.append(image)

    def add_observation_to_history(self, observation: dict):
        if observation is None:
            return
        if "inventory" in observation:
            # clean_inv = []
            # remove empty slots
            # for slot, item in observation["inventory"].items():
            #     if item["quantity"] > 0:
            #         clean_inv.append(item)
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

        self.action_history = []
        self.inventory_history = []
        self.inventory_counters = []

        self.tokens_used = 0

    def trace(self):
        return {
            "dialogue_history": copy(
                self.dialogue_history[self.initial_dialogue_length :]
            ),
            "action_history": copy(self.action_history),
            "inventory_history": copy(self.inventory_history),
            "tokens_used": copy(self.tokens_used),
        }

    @property
    def num_steps(self):
        return len(self.action_history)

    def check_stuck(self, max_steps_no_change: int = 10) -> bool:
        """
        If inventory content does not change for max_steps_no_change steps
        the agent is considered stuck.

        With N=10, the oracle solver can still solve 100% of the examples
        """
        if len(self.inventory_counters) <= max_steps_no_change:
            return False

        return all(
            c == self.inventory_counters[-max_steps_no_change - 1]
            for c in self.inventory_counters[-max_steps_no_change - 1 :]
        )


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
