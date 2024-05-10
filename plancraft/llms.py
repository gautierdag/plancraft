import gc
import time
import os

from enum import Enum

import outlines
from outlines.samplers import MultinomialSampler
from pydantic import BaseModel
from typing import Optional, Union, Literal

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
)

from plancraft.utils import get_downloaded_models

load_dotenv()

MINECRAFT_ITEMS = Literal[
    "anvil",
    "armor_stand",
    "bed",
    "beef",
    "boat",
    "bowl",
    "bucket",
    "carpet",
    "cauldron",
    "chest",
    "chest_minecart",
    "coal",
    "cobblestone",
    "cobblestone_wall",
    "cooked_beef",
    "cooked_mutton",
    "cooked_porkchop",
    "crafting_table",
    "diamond",
    "diamond_shovel",
    "fence",
    "fence_gate",
    "furnace",
    "furnace_minecart",
    "heavy_weighted_pressure_plate",
    "hopper",
    "hopper_minecart",
    "iron_axe",
    "iron_bars",
    "iron_block",
    "iron_boots",
    "iron_chestplate",
    "iron_door",
    "iron_helmet",
    "iron_hoe",
    "iron_ingot",
    "iron_leggings",
    "iron_nugget",
    "iron_ore",
    "iron_pickaxe",
    "iron_shovel",
    "iron_sword",
    "iron_trapdoor",
    "item_frame",
    "jukebox",
    "leather",
    "leather_boots",
    "leather_chestplate",
    "leather_helmet",
    "leather_leggings",
    "lever",
    "log",
    "minecart",
    "mutton",
    "oak_stairs",
    "painting",
    "planks",
    "porkchop",
    "quartz_block",
    "rail",
    "shears",
    "shield",
    "sign",
    "stick",
    "stone",
    "stone_axe",
    "stone_brick_stairs",
    "stone_button",
    "stone_hoe",
    "stone_pickaxe",
    "stone_pressure_plate",
    "stone_shovel",
    "stone_slab",
    "stone_stairs",
    "stone_sword",
    "stonebrick",
    "torch",
    "trapdoor",
    "tripwire_hook",
    "wooden_axe",
    "wooden_button",
    "wooden_hoe",
    "wooden_pickaxe",
    "wooden_pressure_plate",
    "wooden_shovel",
    "wooden_slab",
    "wooden_sword",
    "wool",
]


class ActionType(str, Enum):
    craft = "craft"
    mine = "mine"
    smelt = "smelt"


# class Action(BaseModel):
#     type: ActionType
#     output: MINECRAFT_ITEMS
#     quantity: int
#     tool: Optional[MINECRAFT_ITEMS] = None
#     materials: list[MINECRAFT_ITEMS] = []


class Thought(BaseModel):
    thought: str


class Plan(BaseModel):
    actions: list[Union[Action, Thought]]


class JSONStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria for stopping generation on a list of strings.
    """

    def __init__(self, tokenizer, device=torch.device("cpu")):
        super().__init__()
        # Preparing a tensor of token IDs for tokens that contain "}"
        self.stop_ids = torch.tensor(
            [v for k, v in tokenizer.vocab.items() if "}" in k],
            dtype=torch.long,
            device=device,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        This function is called at each generation step to determine if generation should stop.
        Stops if the last tokens in the input_ids are in the stop_ids.
        """
        # Returns a boolean tensor of shape (batch_size,)
        # NOTE: this is not foolproof if generating multiple sequences
        # at the same time with different lengths
        return (input_ids[:, -1].unsqueeze(-1) == self.stop_ids).any(dim=(1))


class LLMGeneratorBase:
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ):
        self.model_name = model_name

    @staticmethod
    def fix_tokenizer_system_prompt(model_name: str, tokenizer) -> bool:
        """
        Returns True if the model supports a system role
        """
        if "mistral" in model_name.lower():
            # get directory of current file
            current_dir = os.path.dirname(os.path.realpath(__file__))
            chat_template = open(
                current_dir + "/templates/mistral-instruct.jinja"
            ).read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            # set the chat template
            tokenizer.chat_template = chat_template
        if "gemma" in model_name.lower():
            # get directory of current file
            current_dir = os.path.dirname(os.path.realpath(__file__))
            chat_template = open(current_dir + "/templates/gemma-instruct.jinja").read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            # set the chat template
            tokenizer.chat_template = chat_template

    @staticmethod
    def build_model_kwargs(model_name: str, **kwargs) -> tuple[str, dict]:
        model_kwargs = {
            "token": os.getenv("HF_TOKEN"),
            "attn_implementation": "flash_attention_2",
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
            print(f"Using local model {model_name}")

        return model_name, model_kwargs

    def reset(self):
        pass

    def generate(
        self,
        messages: list[dict],
        max_tokens=256,
        **kwargs,
    ) -> tuple[str, int]:
        raise NotImplementedError()


class OpenAIGenerator(LLMGeneratorBase):
    def __init__(self, model_name="gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        messages: list[dict],
        temperature=1.0,
        max_tokens=256,
        enforce_json=False,
        **kwargs,
    ) -> tuple[str, int]:
        kwargs = {}
        if enforce_json:
            kwargs = {"response_format": {"type": "json_object"}}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        text_response = response.choices[0].message.content.strip()
        token_used = response.usage.total_tokens
        return text_response, token_used


class TransformersGenerator(LLMGeneratorBase):
    def __init__(self, model_name: str, quantize=False, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        model_name, model_kwargs = self.build_model_kwargs(
            model_name, quantize=quantize
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HF_TOKEN")
        )
        self.fix_tokenizer_system_prompt(model_name, self.tokenizer)

        time_now = time.time()
        print("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **model_kwargs,
        )

        self.stopping_criteria = JSONStoppingCriteria(
            self.tokenizer, device=self.model.device
        )

        print(f"Model loaded in {time.time() - time_now:.2f} seconds")
        time_now = time.time()
        print("Compiling model to reduce overhead")
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        print(f"Model compiled in {time.time() - time_now:.2f} seconds")

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.truncation_side = "left"

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
    def generate(
        self,
        messages: list[dict],
        temperature=1.0,
        max_tokens=256,
        enforce_json: Union[bool, str] = False,
        **kwargs,
    ) -> tuple[str, int]:
        message_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if enforce_json and isinstance(enforce_json, str):
            # start json object in prompt text
            message_text += f"{enforce_json}"

        max_prompt_length = None
        # need to truncate if max_length is set
        if self.model.generation_config.max_length > max_tokens:
            max_prompt_length = self.model.generation_config.max_length - max_tokens

        tokenized_messages = self.tokenizer.encode(
            message_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        tokenized_messages = tokenized_messages.to(self.model.device)
        _, prompt_tokens = tokenized_messages.shape
        generation_output = self.model.generate(
            tokenized_messages,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            stopping_criteria=[self.stopping_criteria] if enforce_json else None,
        )

        text_response = self.tokenizer.decode(
            generation_output.sequences[0, prompt_tokens:],
            skip_special_tokens=True,
        )
        if enforce_json:
            text_response = enforce_json + text_response

        _, total_tokens_used = generation_output.sequences.shape
        return text_response, total_tokens_used


class GuidanceGenerator(LLMGeneratorBase):
    def __init__(self, model_name: str, quantize=False, temperature=1.0, **kwargs):
        super().__init__(model_name, **kwargs)
        model_name, model_kwargs = self.build_model_kwargs(
            model_name, quantize=quantize
        )
        time_now = time.time()
        print("Loading model")
        self.model = outlines.models.transformers(
            model_name,
            device="auto",
            model_kwargs=model_kwargs,
            tokenizer_kwargs={
                "token": os.getenv("HF_TOKEN"),
            },
        )
        # fix system prompt
        self.fix_tokenizer_system_prompt(model_name, self.model.tokenizer.tokenizer)

        print(f"Model loaded in {time.time() - time_now:.2f} seconds")
        self.bos_token = self.model.tokenizer.tokenizer.bos_token

        self.temperature = temperature
        sampler = MultinomialSampler(temperature=temperature)

        # react mode
        self.action_generator = outlines.generate.json(
            self.model, Action, sampler=sampler
        )
        self.thought_generator = outlines.generate.json(
            self.model, Thought, sampler=sampler
        )
        # full plan mode
        self.plan_generator = outlines.generate.json(self.model, Plan, sampler=sampler)

    @torch.inference_mode()
    def generate(
        self, messages: list[dict], max_tokens=128, mode="think", **kwargs
    ) -> tuple[Union[Action | Thought | Plan], int]:
        message_text = self.model.tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).replace(self.bos_token, "")

        prompt_tokens = len(
            self.model.tokenizer.tokenizer.encode(
                message_text, add_special_tokens=False
            )
        )
        result = None
        while not result:
            try:
                if mode == "think":
                    result = self.thought_generator(message_text, max_tokens=max_tokens)
                elif mode == "action":
                    result = self.action_generator(message_text, max_tokens=max_tokens)
                elif mode == "plan":
                    result = self.plan_generator(message_text, max_tokens=max_tokens)
                else:
                    raise ValueError(f"Mode {mode} not supported")
            except Exception as e:
                # sometimes json generation fails due to wrongly escaped characters
                print("Constrained generation error", e)
                continue

        total_tokens_used = (
            len(
                self.model.tokenizer.tokenizer.encode(
                    result.json(), add_special_tokens=False
                )
            )
            + prompt_tokens
        )
        print(result)
        return result, total_tokens_used


def get_llm_generator(
    model_name: str, guidance=False, quantize=False
) -> LLMGeneratorBase:
    """
    Returns a generator object for the specified model
    """
    if model_name in ["gpt-3.5-turbo", "gpt-4.0-turbo-preview"]:
        return OpenAIGenerator(model_name)
    if quantize:
        assert quantize in ["int4", "int8"], "Quantization must be int4 or int8"
    if guidance:
        return GuidanceGenerator(model_name, quantize=quantize)
    return TransformersGenerator(
        model_name,
        quantize=quantize,
    )
