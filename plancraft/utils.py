import abc
from copy import copy
from dataclasses import dataclass, field
from typing import Optional

from plancraft.environment.actions import ActionHandlerBase
from plancraft.environment.prompts import (
    get_prompt_example,
    get_system_prompt,
    load_prompt_images,
)


@dataclass
class HistoryConfig:
    """Configuration for History instances"""

    few_shot: bool = True
    system_prompt: Optional[dict] = None
    prompt_examples: list[dict] = field(default_factory=list)
    prompt_images: list[str] = field(default_factory=list)


class HistoryBase(abc.ABC):
    """Abstract base class defining the interface required by the Evaluator"""

    @property
    @abc.abstractmethod
    def num_steps(self) -> int:
        """Return the number of interaction steps taken"""
        pass

    @abc.abstractmethod
    def add_message_to_history(
        self, content: str | dict, role: str = "user", **kwargs
    ) -> None:
        """Add a message to the dialogue history"""
        pass

    @abc.abstractmethod
    def add_observation_to_history(self, observation: dict, **kwargs) -> None:
        """Add an observation (inventory, image) to history"""
        pass

    @abc.abstractmethod
    def trace(self) -> dict:
        """Return a traceable history of the interaction"""
        pass

    @property
    @abc.abstractmethod
    def images(self) -> list:
        """Return list of images"""
        pass

    @images.setter
    @abc.abstractmethod
    def images(self, value: list) -> None:
        """Set list of images"""
        pass


class History(HistoryBase):
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
        config: HistoryConfig = HistoryConfig(),
        resolution: str = "high",
        use_multimodal_content_format: bool = False,
        use_images: bool = False,
        use_text_inventory: bool = True,
    ):
        self.action_handlers = actions
        self.use_multimodal_content_format = use_multimodal_content_format

        self.use_images = use_images
        self.use_text_inventory = use_text_inventory
        self.resolution = resolution

        self.inventory_history = []
        self.tokens_used = 0

        # use system prompt if provided
        if config.system_prompt:
            self.system_prompt_dialogue = config.system_prompt
        else:
            # generate system prompt
            self.system_prompt_dialogue = get_system_prompt(
                handlers=self.action_handlers,
                use_multimodal_content_format=self.use_multimodal_content_format,
            )
        self.few_shot = config.few_shot

        # set up dialogue history with few-shot prompt
        self.prompt_examples = config.prompt_examples
        self.prompt_images = config.prompt_images
        self.set_up_few_shot_prompt()

        self.dialogue_history = copy(self.prompt_examples)
        self._images = copy(self.prompt_images)
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

    def add_message_to_history(self, content: str | dict, role="user", **kwargs):
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
        self._images.append(image)

    def add_observation_to_history(self, observation: dict, **kwargs):
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
        self._images = copy(self.prompt_images)
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

    def trace_images(self):
        # return only the images added after the initial dialogue
        return self._images[len(self.prompt_images) :]

    @property
    def num_steps(self):
        return (len(self.dialogue_history) - self.initial_dialogue_length) // 2

    @property
    def images(self) -> list:
        return self._images

    @images.setter
    def images(self, value: list) -> None:
        self._images = value
