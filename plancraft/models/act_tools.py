import copy
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    SymbolicMoveAction,
    SymbolicSmeltAction,
    ThinkAction,
)
from plancraft.models.base import ABCModel, History

load_dotenv()
logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = {
    "ThinkAction": ThinkAction,
    "SymbolicMoveAction": SymbolicMoveAction,
    "SymbolicSmeltAction": SymbolicSmeltAction,
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
        self.action_tools = [
            pydantic_function_tool(SymbolicMoveAction),
            pydantic_function_tool(SymbolicSmeltAction),
        ]
        self.thinking_tools = [pydantic_function_tool(ThinkAction)]

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
        think=False,
        **kwargs,
    ) -> tuple[
        list[SymbolicSmeltAction | SymbolicMoveAction | ThinkAction],
        list[str],
        int,
    ]:
        actions = []
        action_messages = []
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
                tools=self.thinking_tools if think else self.action_tools,
                tool_choice="required",
            )

            # if response.choices[0].message.tool_calls is not None:
            # response contains tool usage
            func = response.choices[0].message.tool_calls[0].function
            args = json.loads(func.arguments)
            content = f"{func.name}({', '.join([f'{k}={v}' for k, v in args.items()])})"
            logger.info(f"Generated action: {content}")
            action = AVAILABLE_TOOLS[func.name](**args)

            tokens_used += response.usage.total_tokens
            actions.append(action)
            action_messages.append(content)

        return actions, action_messages, tokens_used


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
        "content": """SymbolicMoveAction(slot_from=27, slot_to=4, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """SymbolicMoveAction(slot_from=39, slot_to=5, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "andesite", "slot": 0,  "quantity": 1},{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 5, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """SymbolicMoveAction(slot_from=0, slot_to=15, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """SymbolicSmeltAction(slot_from=45, slot_to=44, quantity=1)""",
    },
]


REACT_TOOLS_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """ThinkAction(thought="To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.")""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """SymbolicMoveAction(slot_from=27, slot_to=4, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """ThinkAction(thought="Now I need to move the cobblestone into position 5 to be right of the diorite.")""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """SymbolicMoveAction(slot_from=39, slot_to=5, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "andesite", "slot": 0,  "quantity": 1},{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 5, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """ThinkAction(thought="Now I can craft the andesite by moving it from craft slot to a free inventory slot.")""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """SymbolicMoveAction(slot_from=0, slot_to=15, quantity=1)""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """ThinkAction(thought="To craft an iron_ingot, I need to smelt iron_ore into an empty slot.")""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """SymbolicSmeltAction(slot_from=45, slot_to=44, quantity=1)""",
    },
]


logger = logging.getLogger(__name__)

load_dotenv()


class ActToolsModel(ABCModel):
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
        if "gpt-4o" in cfg.plancraft.model:
            self.llm = OpenAIToolsGenerator(
                model_name=cfg.plancraft.model,
            )

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

    def step(
        self, observations: list[dict]
    ) -> list[SymbolicSmeltAction | SymbolicMoveAction]:
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

        actions, action_messages, action_token_used = self.llm.generate_next(
            batch_messages=action_messages_windows,
            images=action_images_windows,
            think=False,
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


class ReactToolsModel(ABCModel):
    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Real action space unsupported"

        self.is_multimodal = not cfg.plancraft.environment.symbolic
        self.few_shot = cfg.plancraft.few_shot
        self.use_system_prompt = cfg.plancraft.system_prompt

        # underlying language model
        if "gpt-4o" in cfg.plancraft.model:
            self.llm = OpenAIToolsGenerator(
                model_name=cfg.plancraft.model,
            )

        self.batch_size = cfg.plancraft.batch_size
        self.prompt_images = []

        if self.is_multimodal:
            raise NotImplementedError("Multimodal not supported")

        examples = copy.deepcopy(REACT_TOOLS_EXAMPLE)
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
            examples = copy.deepcopy(REACT_TOOLS_EXAMPLE)

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

    def step(
        self, observations: list[dict]
    ) -> list[SymbolicSmeltAction | SymbolicMoveAction]:
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

        thought_messages_windows = []
        thought_images_windows = []
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
            thought_messages_windows.append(message_window)
            thought_images_windows.append(image_window)

        # generate thoughts
        _, thought_messages, thinking_token_used = self.llm.generate_next(
            batch_messages=thought_messages_windows,
            max_tokens=256,
            images=thought_images_windows,
            think=True,
        )

        action_messages_windows = []
        action_images_windows = []
        # update message window with thoughts and collect action messages
        for thought_message, history_idx in zip(thought_messages, real_obs_idx):
            # add thought message to history
            self.histories[history_idx].add_message_to_history(
                content=thought_message, role="assistant"
            )
            self.histories[history_idx].add_message_to_history(
                content="Ok", role="user"
            )
            # add token used to history
            self.histories[history_idx].tokens_used += thinking_token_used

            message_window, image_window = self.llm.prepare_messages(
                history=self.histories[history_idx],
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            action_messages_windows.append(message_window)
            action_images_windows.append(image_window)

        actions, action_messages, action_token_used = self.llm.generate_next(
            batch_messages=action_messages_windows,
            images=action_images_windows,
            think=False,
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
