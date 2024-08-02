import copy
import json
import logging
import re


from dotenv import load_dotenv
from openai import OpenAI


from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    SymbolicAction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
)
from plancraft.models.base import ABCModel, History

load_dotenv()


THINK = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Think about the answer before answering",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "thought about the question",
                }
            },
            "required": ["thought"],
        },
    },
}
PLAN = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": "Allows you to plan the next few actions",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "plan of steps to take",
                }
            },
            "required": ["plan"],
        },
    },
}

SMELT = {
    "type": "function",
    "function": {
        "name": "smelt",
        "description": "Smelts an item from the inventory",
        "parameters": {
            "type": "object",
            "properties": {
                "from_slot": {
                    "type": "integer",
                    "description": "slot to smelt from",
                },
                "to_slot": {
                    "type": "integer",
                    "description": "slot to smelt to",
                },
                "quantity": {
                    "type": "integer",
                    "description": "quantity to smelt",
                },
            },
            "required": ["from_slot", "to_slot", "quantity"],
        },
    },
}

MOVE = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Moves an item from one slot to another",
        "parameters": {
            "type": "object",
            "properties": {
                "from_slot": {
                    "type": "integer",
                    "description": "slot to move from",
                },
                "to_slot": {
                    "type": "integer",
                    "description": "slot to move to",
                },
                "quantity": {
                    "type": "integer",
                    "description": "quantity to move",
                },
            },
            "required": ["from_slot", "to_slot", "quantity"],
        },
    },
}


class OpenAIToolsGenerator:
    def __init__(
        self,
        is_multimodal=False,
        model_name="gpt-4o-mini",
    ):
        self.client = OpenAI()
        self.is_multimodal = is_multimodal

        assert not is_multimodal, "Multimodal not supported w/ tools"
        self.model_name = model_name
        self.tools = [SMELT, MOVE]

    def reset(self):
        pass

    def prepare_messages(
        self,
        history: History,
        max_messages_window: int,
        system_prompt: dict = None,
        prompt_images: list = [],
    ) -> tuple[list[dict], list]:
        """
        Prepare the image messages for the model
        """
        message_window = history.dialogue_history[-max_messages_window:]
        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]
        # add the system prompt if the first message is not a system message
        if message_window[0]["role"] != "system" and system_prompt is not None:
            message_window = [system_prompt] + message_window

        return message_window, []

    def generate_next(
        self,
        batch_messages: list[list[dict]],
        max_tokens=256,
        temperature=1.0,
        **kwargs,
    ) -> tuple[list[str], int]:
        responses = []
        tokens_used = 0
        for messages in batch_messages:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"],
                tools=self.tools,
            )
            content = response.choices[0].message.content
            tokens_used += response.usage.total_tokens
            responses.append(content)
        return responses, tokens_used


ACT_TOOLS_SYSTEM_PROMPT = """
You are crafting in Minecraft. Actions are tools.

The first 10 slots in the inventory are reserved for crafting and correspond to the minecraft crafting table. 

[1, 2, 3] 
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move or smelt items into output slot 0.
The remaining slots (10-45) are for storing items.
"""

ACT_TOOLS_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """move(from_slot=27, to_slot=4, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """move(from_slot=39, to_slot=5, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "andesite", "slot": 0,  "quantity": 1},{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 5, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """move(from_slot=0, to_slot=15, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """smelt(from_slot=45, to_slot=44, quantity=1)""",
    },
]

logger = logging.getLogger(__name__)

load_dotenv()


class ToolsAgentsModel(ABCModel):
    """
    Model that treats every action as a tool
    """

    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Real action space unsupported"

        self.is_multimodal = not cfg.plancraft.environment.symbolic
        self.few_shot = cfg.plancraft.few_shot
        self.use_system_prompt = cfg.plancraft.system_prompt

        # underlying language model
        # if "gpt-4o" in cfg.plancraft.model:
        self.llm = OpenAIToolsGenerator(model_name=cfg.plancraft.model)
        # model is transformers based
        # else:
        #     self.llm = TransformersGenerator(
        #         model_name=cfg.plancraft.model,
        #         tokenizer_name=cfg.plancraft.tokenizer,
        #         quantize=cfg.plancraft.quantize,
        #         is_multimodal=self.is_multimodal,
        #         use_hot_cache=cfg.plancraft.hot_cache,
        #     )

        self.batch_size = cfg.plancraft.batch_size
        self.prompt_images = []

        if self.is_multimodal:
            raise NotImplementedError("Multimodal not supported")

        examples = copy.deepcopy(ACT_TOOLS_EXAMPLE)
        self.system_prompt = {
            "role": "system",
            "content": copy.deepcopy(ACT_TOOLS_SYSTEM_PROMPT),
        }

        if not self.few_shot:
            examples = []
        if not self.use_system_prompt:
            self.system_prompt = None

        self.histories = [
            History(
                initial_dialogue=examples,
                is_multimodal=self.is_multimodal,
            )
            for _ in range(self.batch_size)
        ]

        self.max_messages_window = cfg.plancraft.max_message_window
        self.kv_cache = None

    def reset_history(
        self,
        history_idx: int,
        objective: str,
    ):
        examples = []
        if self.few_shot:
            examples = copy.deepcopy(ACT_TOOLS_EXAMPLE)

        self.histories[history_idx].reset(
            objective=objective, initial_dialogue=examples
        )
        self.llm.reset()

    def convert_observation_to_message(
        self, observation: dict, objective: str
    ) -> str | dict:
        inventory = []
        for o in observation["inventory"]:
            if o["quantity"] > 0:
                inventory.append(
                    {
                        "type": o["type"],
                        "slot": o["index"],
                        "quantity": o["quantity"],
                    }
                )
        return f"{objective}\ninventory={json.dumps(inventory)}"

    def step(self, observations: list[dict]) -> list[SymbolicAction]:
        assert len(observations) == self.batch_size == len(self.histories)

        # filter out None observations
        real_obs = []
        real_obs_idx = []

        for idx, (observation, history) in enumerate(zip(observations, self.histories)):
            # add observation to history
            if observation is not None:
                # note if image is present this adds the image to the history
                history.add_observation_to_history(observation)
                real_obs.append(observation)
                real_obs_idx.append(idx)

        if len(real_obs) == 0:
            return [None] * len(observations)

        action_messages_windows = []
        action_images_windows = []
        # collect dialogue histories
        for observation, history_idx in zip(real_obs, real_obs_idx):
            # add observation to history
            observation_message = self.convert_observation_to_message(
                observation, objective=self.histories[history_idx].objective
            )
            self.histories[history_idx].add_message_to_history(
                content=observation_message, role="user"
            )
            message_window, image_window = self.llm.prepare_messages(
                history=self.histories[history_idx],
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            action_messages_windows.append(message_window)
            action_images_windows.append(image_window)

        actions, action_messages, action_token_used = self.llm.generate_actions(
            batch_messages=action_messages_windows, images=action_images_windows
        )

        for action_message, history_idx in zip(action_messages, real_obs_idx):
            self.histories[history_idx].add_message_to_history(
                content=action_message, role="assistant"
            )
            self.histories[history_idx].tokens_used += action_token_used

        # re-map actions to the correct index in the batch
        out_actions = [None] * len(observations)
        for idx, history_idx in enumerate(real_obs_idx):
            out_actions[history_idx] = actions[idx]
            # add to action history
            self.histories[history_idx].add_action_to_history(actions[idx])

        return out_actions
