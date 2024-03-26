from copy import deepcopy
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_downloaded_models

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


class Action(BaseModel):
    type: ActionType
    output: MINECRAFT_ITEMS
    quantity: int
    tool: Optional[MINECRAFT_ITEMS] = None
    materials: list[MINECRAFT_ITEMS] = []


class Thought(BaseModel):
    thought: str


class Plan(BaseModel):
    actions: list[Union[Action, Thought]]


class LLMGeneratorBase:
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.supports_system_prompt = self.supports_system_prompt(model_name)

    @staticmethod
    def supports_system_prompt(model_name: str) -> bool:
        """
        Returns True if the model supports a system role
        """
        if "mistral" in model_name.lower() or "gemma" in model_name.lower():
            return False
        return True

    def create_initial_history(
        self, system_prompt: str, chat_example: list[dict[str, str]] = []
    ) -> list[str]:
        if self.supports_system_prompt:
            # default implementation supports system prompt
            system_prompt_message = {"role": "system", "content": system_prompt}
            if len(chat_example) == 0:
                return [system_prompt_message]
            history = [system_prompt_message, *deepcopy(chat_example)]
            return history

        # add system prompt to the first message instead
        if len(chat_example) == 0:
            return [{"role": "user", "content": system_prompt}]
        history = deepcopy(chat_example)
        history[0]["content"] = system_prompt + "\n" + history[0]["content"]
        return history

    def reset(self):
        pass

    def generate(
        self,
        messages: list[dict],
        temperature=1.0,
        max_tokens=256,
        enforce_json=False,
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
    def __init__(self, model_name: str, dtype=torch.bfloat16, **kwargs):
        super().__init__(model_name=model_name, dtype=dtype, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN")
        )
        model_kwargs = {
            "trust_remote_code": True,
            "token": os.getenv("HF_TOKEN"),
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2",
        }
        downloaded_models = get_downloaded_models()
        if model_name in downloaded_models:
            model_kwargs["local_files_only"] = True
            model_name = downloaded_models[model_name]
            print(f"Using local model {model_name}")

        time_now = time.time()
        print("Loading model")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
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
        self.past_key_values = None

    def reset(self):
        # TODO: could also cache input ids / attention mask
        # TODO: could explore bfill
        # Remove cached tensors from memory
        self.past_key_values = None
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
        enforce_json=False,
        **kwargs,
    ) -> tuple[str, int]:
        message_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if enforce_json:
            # start json object in prompt text
            message_text += '{"'

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
            past_key_values=self.past_key_values,
        )

        # update past key values for next generation
        self.past_key_values = generation_output.past_key_values

        text_response = self.tokenizer.decode(
            generation_output.sequences[0, prompt_tokens:],
            skip_special_tokens=True,
        )
        if enforce_json:
            text_response = '{"' + text_response

        _, total_tokens_used = generation_output.sequences.shape
        return text_response, total_tokens_used


class GuidanceGenerator(LLMGeneratorBase):
    def __init__(
        self, model_name: str, dtype=torch.bfloat16, temperature=1.0, **kwargs
    ):
        super().__init__(model_name, dtype, **kwargs)
        model_kwargs = {
            "trust_remote_code": True,
            "token": os.getenv("HF_TOKEN"),
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2",
        }
        downloaded_models = get_downloaded_models()
        if model_name in downloaded_models:
            model_kwargs["local_files_only"] = True
            model_name = downloaded_models[model_name]
            print(f"Using local model {model_name}")

        time_now = time.time()
        print("Loading model")
        self.model = outlines.models.transformers(
            model_name,
            device="auto",
            model_kwargs=model_kwargs,
            tokenizer_kwargs={
                "trust_remote_code": True,
                "token": os.getenv("HF_TOKEN"),
            },
        )
        print(f"Model loaded in {time.time() - time_now:.2f} seconds")
        self.bos_token = self.model.tokenizer.tokenizer.bos_token

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


def get_llm_generator(model_name: str, guidance=False) -> LLMGeneratorBase:
    if model_name in ["gpt-3.5-turbo", "gpt-4.0-turbo-preview"]:
        return OpenAIGenerator(model_name)
    if guidance:
        return GuidanceGenerator(model_name)
    return TransformersGenerator(model_name)
