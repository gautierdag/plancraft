from copy import deepcopy
import gc
import time
import os

from enum import Enum

import outlines
from outlines.samplers import MultinomialSampler
from pydantic import BaseModel
from typing import Optional, Union

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class ActionType(str, Enum):
    craft = "craft"
    mine = "mine"
    smelt = "smelt"


# TODO: test further constraint based on inventory
class Action(BaseModel):
    type: ActionType
    output: str
    quantity: int
    tool: Optional[str] = None
    materials: list[str] = []


class Thought(BaseModel):
    thought: str


class Plan(BaseModel):
    actions: list[Action]


class LLMGeneratorBase:
    def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.dtype = dtype

    @staticmethod
    def create_initial_history(
        system_prompt: str, chat_example: list[dict[str, str]]
    ) -> list[str]:
        # default implementation supports system prompt
        system_prompt_message = {"role": "system", "content": system_prompt}
        history = [system_prompt_message, *deepcopy(chat_example)]
        return history

    def reset(self):
        pass

    def generate(
        self, messages: list[dict], temperature=1.0, max_tokens=256, enforce_json=False
    ) -> tuple[str, int]:
        raise NotImplementedError()


class OpenAIGenerator(LLMGeneratorBase):
    def __init__(self, model_name="gpt-3.5-turbo", **kwargs):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self, messages: list[dict], temperature=1.0, max_tokens=256, enforce_json=False
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
    supports_system_prompt = True

    def __init__(self, model_name: str, dtype=torch.bfloat16):
        super().__init__(model_name, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN")
        )
        time_now = time.time()
        print("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            attn_implementation="flash_attention_2",
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
        self, messages: list[dict], temperature=1.0, max_tokens=256, enforce_json=False
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
        print("Prompt tokens", prompt_tokens)
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
        print("Total tokens used", total_tokens_used)
        return text_response, total_tokens_used


class MistralGenerator(TransformersGenerator):
    supports_system_prompt = False

    @staticmethod
    def create_initial_history(
        system_prompt: str, chat_example: list[dict[str, str]]
    ) -> list[str]:
        # Mistral does not support the system prompt
        history = deepcopy(chat_example)
        history[0]["content"] = system_prompt + "\n" + history[0]["content"]
        return history


class GemmaGenerator(MistralGenerator):
    pass


class LLama2Generator(TransformersGenerator):
    pass


class GuidanceGenerator(LLMGeneratorBase):
    supports_system_prompt = True

    def __init__(self, model_name: str, dtype=torch.bfloat16, temperature=1.0):
        super().__init__(model_name, dtype)
        time_now = time.time()
        print("Loading model")
        self.model = outlines.models.transformers(
            model_name,
            device="auto",
            model_kwargs={
                "trust_remote_code": True,
                "token": os.getenv("HF_TOKEN"),
                "torch_dtype": dtype,
                "attn_implementation": "flash_attention_2",
            },
            tokenizer_kwargs={
                "trust_remote_code": True,
                "token": os.getenv("HF_TOKEN"),
            },
        )
        print(f"Model loaded in {time.time() - time_now:.2f} seconds")
        time_now = time.time()
        print("Compiling model to reduce overhead")
        self.model = torch.compile(
            self.model.model, mode="reduce-overhead", fullgraph=True
        )
        print(f"Model compiled in {time.time() - time_now:.2f} seconds")
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
        self.plan_generator = outlines.generate.json(
            self.model, list[Action], sampler=sampler
        )

    @torch.inference_mode()
    def generate(
        self, messages: list[dict], max_tokens=128, mode="think", **kwargs
    ) -> tuple[Union[Action | Thought | Plan], int]:
        message_text = self.model.tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).replace(self.bos_token, "")

        if mode == "think":
            result = self.thought_generator(message_text, max_tokens=max_tokens)
        elif mode == "action":
            result = self.action_generator(message_text, max_tokens=max_tokens)
        elif mode == "plan":
            result = self.plan_generator(message_text, max_tokens=max_tokens)
        else:
            raise ValueError(f"Mode {mode} not supported")

        total_tokens_used = len(
            self.model.tokenizer.tokenizer.encode(
                result.json(), add_special_tokens=False
            )
        )
        print("Total tokens used", total_tokens_used)
        return result, total_tokens_used


def get_llm_generator(model_name: str, guidance=False) -> LLMGeneratorBase:
    if model_name in ["gpt-3.5-turbo", "gpt-4.0-turbo-preview"]:
        return OpenAIGenerator(model_name)
    if guidance:
        print("Using Guidance")
        return GuidanceGenerator(model_name)
    elif "Llama-2" in model_name:
        return LLama2Generator(model_name)
    elif "Mistral" in model_name:
        return MistralGenerator(model_name)
    elif "gemma" in model_name:
        return GemmaGenerator(model_name)
    else:
        raise ValueError(f"LLM {model_name} not supported")
