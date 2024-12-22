import glob
import pathlib
from collections import Counter
from copy import copy

import torch
from loguru import logger

from plancraft.environment.actions import (
    MoveAction,
    SmeltAction,
)


class History:
    def __init__(
        self,
        initial_dialogue: list[dict] = [],
        use_multimodal_content_format=False,
    ):
        self.dialogue_history = initial_dialogue
        self.initial_dialogue_length = len(initial_dialogue)
        self.action_history = []
        self.inventory_history = []
        self.inventory_counters = []
        self.images = []
        self.tokens_used = 0
        self.use_multimodal_content_format = use_multimodal_content_format

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
        if action is None:
            return
        self.action_history.append(action.model_dump())

    def add_inventory_to_history(self, inventory: list[dict[str, int]]):
        self.inventory_history.append(inventory)

        # count inventory
        counter = Counter()
        for item in inventory:
            # ignore slot 0
            if "slot" in item and item["slot"] == 0:
                continue
            counter[item["type"]] += item["quantity"]

        self.inventory_counters.append(counter)

    def add_image_to_history(self, image):
        self.images.append(image)

    def add_observation_to_history(self, observation: dict):
        if observation is None:
            return
        if "inventory" in observation:
            clean_inv = []
            # remove empty slots
            for item in observation["inventory"]:
                if item["quantity"] > 0:
                    clean_inv.append(item)
            self.add_inventory_to_history(clean_inv)
        if "image" in observation:
            self.add_image_to_history(observation["image"])

    def __str__(self):
        return str(self.dialogue_history)

    def reset(self, objective: str = "", initial_dialogue: list[dict] = []):
        self.dialogue_history = initial_dialogue
        self.action_history = []
        self.inventory_history = []
        self.inventory_counters = []
        self.images = []

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
