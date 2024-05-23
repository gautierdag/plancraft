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

You must output an action like the following:
act: move from slot X to slot Y with quantity Z

There are two types of actions
- move
- smelt

To assist with planning, you first generate some thoughts before answering. For example:
think: To craft an acacia_fence, I first need to craft acacia_planks so I need to move the log from ...

The first 10 slots in the inventory are reserved for crafting and correspond to the minecraft crafting table. 

[1, 2, 3] 
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move items into output slot 0.
The remaining slots (10-46) are for storing items.
"""

REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
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
        "content": """TASK: Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
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
            model_name,
            token=os.getenv("HF_TOKEN"),  # trust_remote_code=True
            padding_side="left",  # ensure that the padding is on the left
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
            # "trust_remote_code": True,
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
        # # TODO: could use past_key_values cache with a rolling window
        # # TODO: could also cache input ids / attention mask
        # # TODO: could explore bfill
        # # Remove cached tensors from memory
        # # clear cuda cache
        torch.cuda.empty_cache()
        # # Manually invoke garbage collector
        gc.collect()
        # pass

    @torch.inference_mode()
    def generate_thought(
        self,
        batch_messages: list[list[dict]],
        temperature=1.0,
        max_tokens=256,
        **kwargs,
    ) -> tuple[str, int]:
        tokenized_messages = tokenize(
            self.model,
            self.tokenizer,
            batch_messages,
            max_tokens,
            new_message_start="think:",
        )
        prompt_tokens = tokenized_messages["input_ids"].shape[-1]

        # sent to same device as model
        tokenized_messages = {
            k: v.to(self.model.device) for k, v in tokenized_messages.items()
        }

        generation_output = self.model.generate(
            input_ids=tokenized_messages["input_ids"],
            attention_mask=tokenized_messages["attention_mask"],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
        # decode the output
        text_responses = self.tokenizer.batch_decode(
            generation_output.sequences[:, prompt_tokens:],
            skip_special_tokens=True,
        )
        # text_response = "think:" + text_response
        text_responses = [f"think: {text_response}" for text_response in text_responses]
        _, total_tokens_used = generation_output.sequences.shape
        return text_responses, total_tokens_used

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
        overall_message += f" {action} from slot "
        slot_from, _ = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=[str(i) for i in range(46)],
            new_message_start=overall_message,
            temperature=temperature,
        )
        overall_message += f"{slot_from} to slot "
        slot_to, _ = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=[str(i) for i in range(1, 46)],
            new_message_start=overall_message,
            temperature=temperature,
        )
        overall_message += f"{slot_to} with quantity "
        quantity, num_tokens = decode_with_choices(
            self.model,
            self.tokenizer,
            messages,
            choices=[str(i) for i in range(1, 65)],
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
        # return the action, message, and the number of tokens used
        return act, overall_message, num_tokens


class ReactModel(ABCModel):
    def __init__(self, cfg: Config, llm: TransformersGenerator):
        assert cfg.plancraft.environment.symbolic_action_space

        self.llm = llm  # language model
        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
        self.action_history = []
        self.history = []
        self.token_used = 0
        self.num_thinking_steps = 0
        self.max_messages_window = 30

    def set_objective(self, objective: str):
        self.objective = objective
        self.system_prompt["content"] = (
            self.system_prompt["content"] + f"\nTASK: {objective}"
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
        return f"TASK: {self.objective}\ninventory={json.dumps(inventory)}"

    def step(self, observation: dict) -> SymbolicAction:
        observation_str = self.convert_observation_to_text(observation)
        logger.info(f"Observation: {observation_str}")
        self.history.append({"role": "user", "content": observation_str})
        action = self.generate()
        self.action_history.append(action.model_dump())
        logger.info(f"Action: {action.model_dump()}")
        return action

    @property
    def trace(self) -> dict:
        return {
            "objective": self.objective,
            "action_history": self.action_history,
            "history": self.history,
            "num_thinking_steps": self.num_thinking_steps,
            "token_used": self.token_used,
        }

    def reset(self) -> None:
        self.llm.reset()
        self.action_history = []
        self.history = copy.deepcopy(REACT_EXAMPLE)
        self.token_used = 0
        self.objective = ""
        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }


AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
