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
from plancraft.models.base import ABCModel, History
from plancraft.models.utils import Trie, get_downloaded_models, tokenize

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


class ValidActionsLogitsProcessor(torch.nn.Module):
    def __init__(self, choices: list[str], tokenizer: AutoTokenizer):
        super().__init__()
        self.choices = choices
        self.tree = Trie()
        self.start_idx = None
        self.eos = tokenizer.eos_token_id
        encoded_choices = tokenizer(choices, add_special_tokens=False)["input_ids"]
        for choice in encoded_choices:
            self.tree.insert(choice + [self.eos])

    def forward(self, input_ids, scores):
        if self.start_idx is None:
            # Calculate start_idx during the first forward pass
            self.start_idx = input_ids.shape[-1]

        decoded_so_far = input_ids[:, self.start_idx :]
        mask = torch.full_like(scores, float("-inf"))
        for batch_idx in range(input_ids.shape[0]):
            valid_next_tokens = self.tree.get_next(decoded_so_far[batch_idx].tolist())
            # if no choice then we allow the model to generate eos
            if len(valid_next_tokens) == 0:
                valid_next_tokens = [self.eos]

            mask[batch_idx, valid_next_tokens] = 0
        return scores + mask

    def reset(self):
        self.start_idx = None


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

        self.action_logit_processor = ValidActionsLogitsProcessor(
            [" move", " smelt"], self.tokenizer
        )
        self.slot_from_processor = ValidActionsLogitsProcessor(
            [str(i) for i in range(46)], self.tokenizer
        )
        self.slot_to_processor = ValidActionsLogitsProcessor(
            [str(i) for i in range(1, 46)], self.tokenizer
        )
        self.quantity_processor = ValidActionsLogitsProcessor(
            [str(i) for i in range(1, 65)], self.tokenizer
        )

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

    # def reset(self):
    # # TODO: could use past_key_values cache with a rolling window
    # # TODO: could also cache input ids / attention mask
    # # TODO: could explore bfill
    # # Remove cached tensors from memory
    # # clear cuda cache
    # torch.cuda.empty_cache()
    # # Manually invoke garbage collector
    # gc.collect()
    # pass

    @torch.inference_mode()
    def generate_thoughts(
        self,
        batch_messages: list[list[dict]],
        temperature=1.0,
        max_tokens=256,
        **kwargs,
    ) -> tuple[list[str], int]:
        tokenized_messages = tokenize(
            self.model,
            self.tokenizer,
            batch_messages,
            start_messages_generation=["think:"] * len(batch_messages),
            max_tokens=max_tokens,
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
        text_responses = [f"think:{text_response}" for text_response in text_responses]
        _, total_tokens_used = generation_output.sequences.shape
        return text_responses, total_tokens_used

    def generate_with_processor(
        self,
        logits_processor: ValidActionsLogitsProcessor,
        start_messages_generation: list[str],
        batch_messages: list[list[dict]],
        temperature=1.0,
    ) -> tuple[list[str], int]:
        tokenized_messages = tokenize(
            self.model,
            self.tokenizer,
            batch_messages,
            start_messages_generation=start_messages_generation,
        )

        # sent to same device as model
        tokenized_messages = {
            k: v.to(self.model.device) for k, v in tokenized_messages.items()
        }

        # number of tokens in the prompt
        prompt_tokens = tokenized_messages["input_ids"].shape[-1]

        # Generate the initial action constrained to valid action tokens
        generated_sequences = self.model.generate(
            input_ids=tokenized_messages["input_ids"],
            attention_mask=tokenized_messages["attention_mask"],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=logits_processor.tree.longest_sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            logits_processor=[logits_processor],
        )

        # reset the start index
        logits_processor.reset()

        # select only new tokens and decode the generated choices
        generated_choices = self.tokenizer.batch_decode(
            generated_sequences["sequences"][:, prompt_tokens:],
            skip_special_tokens=True,
        )
        return generated_choices, generated_sequences["sequences"].shape[-1]

    @torch.inference_mode()
    def generate_actions(
        self,
        batch_messages: list[list[dict]],
        temperature=1.0,
    ) -> tuple[list[SymbolicAction], list[str], int]:
        """
        Select whether to smelt or move
        Then select the slots and quantity
        output should be constrained to something like:
            `act: move from slot 39 to slot 5 with quantity 1`
        """
        overall_messages = ["act:"] * len(batch_messages)
        actions_selected, _ = self.generate_with_processor(
            self.action_logit_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
        )
        overall_messages = [
            f"{overall}{action} from slot "
            for (overall, action) in zip(overall_messages, actions_selected)
        ]
        # select the slot from
        slots_from_selected, _ = self.generate_with_processor(
            self.slot_from_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
        )
        overall_messages = [
            f"{overall}{slot_from} to slot "
            for (overall, slot_from) in zip(overall_messages, slots_from_selected)
        ]
        # select the slot to
        slots_to_selected, _ = self.generate_with_processor(
            self.slot_to_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
        )
        overall_messages = [
            f"{overall}{slot_to} with quantity "
            for (overall, slot_to) in zip(overall_messages, slots_to_selected)
        ]
        # select the quantity
        quantities_selected, num_tokens = self.generate_with_processor(
            self.quantity_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
        )
        overall_messages = [
            f"{overall}{quantity}"
            for (overall, quantity) in zip(overall_messages, quantities_selected)
        ]

        # parse the actions
        actions = []
        for action, slot_from, slot_to, quantity in zip(
            actions_selected,
            slots_from_selected,
            slots_to_selected,
            quantities_selected,
        ):
            if action == "smelt":
                act = SymbolicSmeltAction(
                    slot_from=int(slot_from),
                    slot_to=int(slot_to),
                    quantity=int(quantity),
                )
            else:
                act = SymbolicMoveAction(
                    slot_from=int(slot_from),
                    slot_to=int(slot_to),
                    quantity=int(quantity),
                )
            actions.append(act)
        return actions, overall_messages, num_tokens


class ReactModel(ABCModel):
    def __init__(self, cfg: Config, llm: TransformersGenerator):
        assert cfg.plancraft.environment.symbolic_action_space

        # underlying language model
        self.llm = llm

        self.batch_size = cfg.plancraft.batch_size
        self.histories = [
            History(initial_dialogue=copy.deepcopy(REACT_EXAMPLE))
            for _ in range(self.batch_size)
        ]

        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
        self.tokens_used = 0
        self.max_messages_window = 30

    def reset_history(
        self,
        history_idx: int,
        objective: str,
    ):
        self.histories[history_idx].reset(
            objective=objective, initial_dialogue=copy.deepcopy(REACT_EXAMPLE)
        )

    def convert_observation_to_text(self, observation: dict, objective: str) -> str:
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
        return f"TASK: {objective}\ninventory={json.dumps(inventory)}"

    def step(self, observations: list[dict]) -> list[SymbolicAction]:
        thought_messages_windows = []

        # collect dialogue histories
        for observation, history in zip(observations, self.histories):
            # add observation to history
            observation_str = self.convert_observation_to_text(
                observation, objective=history.objective
            )
            history.add_message_to_history(content=observation_str, role="user")

            message_window = history.dialogue_history[-self.max_messages_window :]
            # remove the first assistant message if it is present
            if len(message_window) > 0 and message_window[0]["role"] == "assistant":
                message_window = message_window[1:]
            # add the system prompt if the first message is not a system message
            if message_window[0]["role"] != "system":
                message_window = [self.system_prompt] + message_window

            thought_messages_windows.append(message_window)

        # generate thoughts
        thought_messages, thinking_token_used = self.llm.generate_thoughts(
            batch_messages=thought_messages_windows,
            max_tokens=256,
        )

        action_messages_windows = []
        # update message window with thoughts and collect action messages
        for thought_message, history in zip(thought_messages, self.histories):
            # add thought message to history
            history.add_message_to_history(content=thought_message, role="assistant")
            history.add_message_to_history(content="OK", role="user")

            message_window = history.dialogue_history[-self.max_messages_window :]
            # remove the first assistant message if it is present
            if len(message_window) > 0 and message_window[0]["role"] == "assistant":
                message_window = message_window[1:]
            # add the system prompt if the first message is not a system message
            if message_window[0]["role"] != "system":
                message_window = [self.system_prompt] + message_window

            action_messages_windows.append(message_window)

        actions, action_messages, action_token_used = self.llm.generate_actions(
            batch_messages=action_messages_windows,
        )

        for action_message, history in zip(action_messages, self.histories):
            history.add_message_to_history(content=action_message, role="assistant")

        # NOTE: this overestimates the token used as it some batches might use less tokens
        self.tokens_used += (thinking_token_used + action_token_used) * self.batch_size

        return actions
