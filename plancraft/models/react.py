import copy
import gc
import json
import logging
import os
import time

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from plancraft.config import Config
from plancraft.environments.actions import (
    SymbolicAction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
)
from plancraft.models.base import ABCModel
from plancraft.models.utils import decode_with_choices, get_downloaded_models, tokenize

logger = logging.getLogger(__name__)

load_dotenv()


REACT_SYSTEM_PROMPT = """You are crafting in Minecraft.
You need to decide on the next action.

You must output an action in JSON format like the following:
{
    "slot_from": 1,
    "slot_to": 1, 
    "quantity": 1 
}

There are two types of actions
- move
- smelt

To assist with planning, you first generate some thoughts before answering. For example:
{
    "thought": "To craft an acacia_fence, I first need to craft acacia_planks."
}

The first 10 slots in the inventory are reserved for crafting. 

[1, 2, 3] 
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move items into output slot 0.
The remaining slots (10-46) are for storing items.

Do not generate errors. Always generate valid JSON objects following the format above.
"""

REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory='[{"slot": 27,"type": "diorite", "quantity": 1},{"slot": 39,"type": "cobblestone", "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """think: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """act: move from slot 27 to slot 4 with quantity 1""",
    },
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory=[{"slot": 4, "type": "diorite", "quantity": 1},{"slot": 39, "type": "cobblestone", "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """think: Now I need to move the cobblestone into position 5 to be right of the diorite..""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """act: move from slot 39 to slot 5 with quantity 1""",
    },
]


class TransformersGenerator:
    def __init__(self, model_name: str, quantize=False, **kwargs):
        self.model_name = model_name
        model_name, model_kwargs = self.build_model_kwargs(
            model_name, quantize=quantize
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HF_TOKEN"), trust_remote_code=True
        )
        self.fix_tokenizer_system_prompt(model_name, self.tokenizer)

        time_now = time.time()
        logger.info("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **model_kwargs,
        )
        logger.info(f"Model loaded in {time.time() - time_now:.2f} seconds")

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.truncation_side = "left"

    @staticmethod
    def fix_tokenizer_system_prompt(model_name: str, tokenizer):
        """
        Returns True if the model supports a system role
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if "mistral" in model_name.lower():
            # get directory of current file
            chat_template = open(
                current_dir + "/templates/mistral-instruct.jinja"
            ).read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            # set the chat template
            tokenizer.chat_template = chat_template
        elif "gemma" in model_name.lower():
            # get directory of current file
            chat_template = open(current_dir + "/templates/gemma-instruct.jinja").read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            # set the chat template
            tokenizer.chat_template = chat_template
        elif "phi" in model_name.lower():
            chat_template = open(current_dir + "/templates/phi-instruct.jinja").read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            tokenizer.chat_template = chat_template

    @staticmethod
    def build_model_kwargs(model_name: str, **kwargs) -> tuple[str, dict]:
        model_kwargs = {
            "token": os.getenv("HF_TOKEN"),
            # "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        }
        quantize = kwargs.get("quantize", False)
        if quantize == "int4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantize == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        downloaded_models = get_downloaded_models()
        if model_name in downloaded_models:
            model_kwargs["local_files_only"] = True
            model_name = downloaded_models[model_name]
            logger.info(f"Using local model {model_name}")

        return model_name, model_kwargs

    def reset(self):
        # TODO: could use past_key_values cache with a rolling window
        # TODO: could also cache input ids / attention mask
        # TODO: could explore bfill
        # Remove cached tensors from memory
        # clear cuda cache
        torch.cuda.empty_cache()
        # Manually invoke garbage collector
        gc.collect()

    @torch.inference_mode()
    def generate_thought(
        self,
        messages: list[dict],
        temperature=1.0,
        max_tokens=256,
        **kwargs,
    ) -> tuple[str, int]:
        tokenized_messages, prompt_tokens = tokenize(
            self.model, self.tokenizer, messages, max_tokens, new_message_start="think:"
        )
        generation_output = self.model.generate(
            tokenized_messages,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
        # decode the output
        text_response = self.tokenizer.decode(
            generation_output.sequences[0, prompt_tokens:],
            skip_special_tokens=True,
        )
        text_response = "think:" + text_response
        _, total_tokens_used = generation_output.sequences.shape
        return text_response, total_tokens_used

    @torch.inference_mode()
    def generate_action(
        self,
        messages: list[dict],
        temperature=1.0,
    ) -> tuple[SymbolicAction, str, int]:
        """
        Select whether to smelt or move
        Then select the slots and quantity
        output should be constrained to something like:
            `act: move from slot 39 to slot 5 with quantity 1`
        """
        overall_message = "act:"
        action, _ = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=["move", "smelt"],
            new_message_start=overall_message,
            temperature=temperature,
        )
        overall_message += f"{action} from slot "
        slot_from, _ = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=[str(i) for i in range(47)],
            new_message_start=overall_message,
            temperature=temperature,
        )
        overall_message += f"{slot_from} to slot "
        slot_to, _ = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=[str(i) for i in range(47)],
            new_message_start=overall_message,
            temperature=temperature,
        )
        overall_message += f"{slot_to} with quantity "
        quantity, generation_output = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=[str(i) for i in range(64)],
            new_message_start=overall_message,
            temperature=temperature,
        )
        overall_message += f"{quantity}"
        if action == "smelt":
            act = SymbolicSmeltAction(
                slot_from=slot_from, slot_to=slot_to, quantity=quantity
            )
        else:
            act = SymbolicMoveAction(
                slot_from=slot_from, slot_to=slot_to, quantity=quantity
            )
        # return the action and the number of tokens used
        return act, overall_message, generation_output.sequences.shape[-1]


class ReactModel(ABCModel):
    def __init__(self, cfg: Config):
        assert cfg.plancraft.environment.symbolic_action_space

        self.llm = TransformersGenerator(
            model_name=cfg.plancraft.model,
            quantize=cfg.plancraft.quantize,
        )

        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
        self.action_history = []
        self.history = []
        self.token_used = 0
        self.max_thinking_steps = 1
        self.num_thinking_steps = 0
        self.max_messages_window = 50

    def set_objective(self, objective: str):
        self.objective = objective
        self.system_prompt["content"] = (
            self.system_prompt["content"] + f"\n\nCURRENT TASK: {objective}"
        )

    def generate(self, max_tokens=256):
        # get the N last messages from the history
        message_window = self.history[-self.max_messages_window :]

        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]

        # add the system prompt to the message window
        message_window = [self.system_prompt] + message_window

        # we force the model to think and then act
        thought_message, thinking_token_used = self.llm.generate_thought(
            messages=message_window,
            max_tokens=max_tokens,
        )
        thought_chat_message = {
            "role": "assistant",
            "content": thought_message,
        }
        # add the thought message to the history and window
        self.history.append(thought_chat_message)
        message_window.append(thought_chat_message)

        self.history.append({"role": "user", "content": "OK"})
        message_window.append({"role": "user", "content": "OK"})

        self.num_thinking_steps += 1
        action, action_message, action_token_used = self.llm.generate_action(
            messages=message_window,
        )
        action_chat_message = {
            "role": "assistant",
            "content": action_message,
        }
        self.history.append(action_chat_message)

        self.token_used += thinking_token_used + action_token_used
        logger.info(
            f"Thinking token used: {thinking_token_used}, Action token used: {action_token_used}, Total token used: {thinking_token_used+action_token_used}"
        )
        return action

    def convert_observation_to_text(self, observation: dict) -> str:
        # @TODO
        # 1. parse observations from json/image to text
        inventory = [o for o in observation["inventory"] if o["quantity"] > 0]
        return f"TASK: {self.objective}\ninventory={json.dumps(inventory)}"

    def step(self, observation: dict) -> SymbolicAction:
        observation_str = self.convert_observation_to_text(observation)
        logger.info(f"Observation: {observation_str}")
        self.history.append({"role": "user", "content": observation_str})
        action = self.generate()
        logger.info(f"History: {self.history}")
        return action

    @property
    def trace(self) -> dict:
        return {"objective": self.objective, "action_history": self.action_history}

    def reset(self) -> None:
        self.llm.reset()
        self.action_history = []
        self.history = copy.deepcopy(REACT_EXAMPLE)
        self.objective = ""
        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
