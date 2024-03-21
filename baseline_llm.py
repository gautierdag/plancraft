import json
import re
from copy import deepcopy
import gc
import time
import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

ONE_SHOT_SYSTEM_PROMPT = """You are a helper AI agent in Minecraft.
You need to generate the sequences of sub-goals (actions) for a planning task in Minecraft.

You have the choice to use the following commands:
- mine
- craft
- smelt

You must answer with code.

Example Response:

```python
def obtain_iron_ingot(inventory):
    # first I need to mine 3 logs using the diamond_axe
    mine({'log': 3}, {'diamond_axe': 1})
    # then I can craft 11 planks using the logs
    craft({'planks': 11}, {'log': 3})
    # I will also need sticks, so I can craft 6 sticks using the planks
    craft({'stick': 6}, {'planks': 2})
    # I will then need a crafting table to craft the wooden_pickaxe
    craft({'crafting_table': 1}, {'planks': 4})
    # I can now craft a wooden_pickaxe using the planks, sticks and crafting_table
    craft({'wooden_pickaxe': 1}, {'planks': 3, 'stick': 2, 'crafting_table': 1})
    # I need to mine 11 cobblestone using the wooden_pickaxe
    mine({'cobblestone': 11}, {'wooden_pickaxe': 1})
    # Using the cobblestone, sticks and crafting_table, I can craft a stone_pickaxe
    craft({'stone_pickaxe': 1}, {'cobblestone': 3, 'stick': 2, 'crafting_table': 1})
    # I can now mine iron_ore using the stone_pickaxe
    mine({'iron_ore': 1}, {'stone_pickaxe': 1})
    # I need to craft a furnace using the cobblestone
    craft({'furnace': 1}, {'cobblestone': 8})
    # Finally, I can smelt the iron_ore using the furnace to obtain iron_ingot
    smelt({'iron_ingot': 1}, {'iron_ore': 1, 'furnace': 1})
    return 'iron_ingot
```

The first argument of each line is a dict of items to be obtained, and the second argument is a dict of items to be used.
"""

ONE_SHOT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: How to obtain iron_ingot?\ninventory = {'diamond_axe'}""",
    },
    {
        "role": "assistant",
        "content": """def obtain_iron_ingot(inventory):
    # first I need to mine 3 logs using the diamond_axe
    mine({'log': 3}, {'diamond_axe': 1})
    # then I can craft 11 planks using the logs
    craft({'planks': 11}, {'log': 3})
    # I will also need sticks, so I can craft 6 sticks using the planks
    craft({'stick': 6}, {'planks': 2})
    # I will then need a crafting table to craft the wooden_pickaxe
    craft({'crafting_table': 1}, {'planks': 4})
    # I can now craft a wooden_pickaxe using the planks, sticks and crafting_table
    craft({'wooden_pickaxe': 1}, {'planks': 3, 'stick': 2, 'crafting_table': 1})
    # I need to mine 11 cobblestone using the wooden_pickaxe
    mine({'cobblestone': 11}, {'wooden_pickaxe': 1})
    # Using the cobblestone, sticks and crafting_table, I can craft a stone_pickaxe
    craft({'stone_pickaxe': 1}, {'cobblestone': 3, 'stick': 2, 'crafting_table': 1})
    # I can now mine iron_ore using the stone_pickaxe
    mine({'iron_ore': 1}, {'stone_pickaxe': 1})
    # I need to craft a furnace using the cobblestone
    craft({'furnace': 1}, {'cobblestone': 8})
    # Finally, I can smelt the iron_ore using the furnace to obtain iron_ingot
    smelt({'iron_ingot': 1}, {'iron_ore': 1, 'furnace': 1})
    return 'iron_ingot""",
    },
]


REACT_SYSTEM_PROMPT = """You are a helper AI agent in Minecraft.
You need to decide on the next action to accomplish the planning TASK in Minecraft.

You must output a JSON object like the following:
{
"output": "iron_ingot",
"quantity": 1 # quantity to be produced
"type": "smelt",
"materials": ["iron_ore"] # optional list of materials
"tool": "furnace" # optional tool
}

There are four types of actions
- think
- mine
- craft
- smelt

If you output "think", you can output a string with your thought. For example:
{
"type": "think",
"thought": "I need to gather some logs to craft planks using the diamond_axe."
}

Do not generate errors. Always generate valid JSON objects following the format above.
"""

REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: How to obtain iron_ingot?\ninventory = {'diamond_axe': 1}""",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "To solve this task I need to smelt an iron_ingot."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "First I need to gather some logs to craft planks using the diamond_axe."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "mine",\n"output": "log",\n"quantity": 3,\n"tool": "diamond_axe"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'log': 3}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can craft planks using the logs. I need 11 planks to craft sticks and a crafting_table."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft",\n"output": "planks",\n"quantity": 11,\n"materials": ["log"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'planks': 12}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can craft sticks using the planks. I need 4 sticks to craft a wooden_pickaxe and then the stone_pickaxe."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft",\n"output": "stick",\n"quantity": 4,\n"materials": ["planks"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'planks': 10, 'stick': 4}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can craft a crafting_table using the planks."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft",\n"output": "crafting_table",\n"quantity": 1,\n"materials": ["planks"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'planks': 6, 'stick': 4, 'crafting_table': 1}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can craft a wooden_pickaxe using the planks, sticks and crafting_table."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft",\n"output": "wooden_pickaxe",\n"quantity": 1,\n"materials": ["planks", "stick"]\n"tool": "crafting_table"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'stick': 2, 'crafting_table': 1, 'wooden_pickaxe': 1}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I need to mine 11 cobblestone using the wooden_pickaxe to be able to craft a stone_pickaxe and a furnace."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "mine",\n"output": "cobblestone",\n"quantity": 11,\n"tool": "wooden_pickaxe"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'cobblestone': 11, 'stick': 2, 'crafting_table': 1, 'wooden_pickaxe': 1}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can craft a stone_pickaxe using the cobblestone, sticks and crafting_table."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft",\n"output": "stone_pickaxe",\n"quantity": 1,\n"materials": ["cobblestone", "stick"],\n"tool": "crafting_table"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'cobblestone': 8, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can mine iron_ore using the stone_pickaxe."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "mine",\n"output": "iron_ore",\n"quantity": 1,\n"tool": "stone_pickaxe"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'cobblestone': 8, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1, 'iron_ore': 3}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can craft a furnace using the cobblestone to smelt the iron_ore."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft",\n"output": "furnace",\n"quantity": 1,\n"materials": ["cobblestone"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1, 'furnace': 1}",
    },
    {
        "role": "assistant",
        "content": """{"type": "think",\n"thought": "Now I can smelt the iron_ore using the furnace to obtain iron_ingot."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "smelt",\n"output": "iron_ingot",\n"quantity": 1,\n"materials": ["iron_ore"],\n"tool": "furnace"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1, 'furnace': 1, 'iron_ingot': 1}",
    },
    {
        "role": "assistant",
        "content": """DONE""",
    },
]


class LLMGeneratorBase:
    def __init__(self, model_name: str, dtype: torch.dtype):
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
    def __init__(self, model_name="gpt-3.5-turbo", dtype=torch.bfloat16):
        super().__init__(model_name, dtype)
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


class TransformerGenerator(LLMGeneratorBase):
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
        self.model = torch.compile(self.model, mode="reduce-overhead")
        print(f"Model compiled in {time.time() - time_now:.2f} seconds")

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.past_key_values = None

    def reset(self):
        # Move tensors no longer needed to CPU and delete them
        self.past_key_values = None
        # clear cuda cache
        torch.cuda.empty_cache()
        # Manually invoke garbage collector
        gc.collect()

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

        with torch.no_grad():
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


class MistralGenerator(TransformerGenerator):
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


class LLama2Generator(TransformerGenerator):
    pass


def get_llm_generator(model_name: str) -> LLMGeneratorBase:
    if model_name in ["gpt-3.5-turbo", "gpt-4.0-turbo-preview"]:
        return OpenAIGenerator(model_name)
    elif "Llama-2" in model_name:
        return LLama2Generator(model_name)
    elif "Mistral" in model_name:
        return MistralGenerator(model_name)
    elif "gemma" in model_name:
        return GemmaGenerator(model_name)
    else:
        raise ValueError(f"LLM {model_name} not supported")


@dataclass
class ActionStep:
    output: str  # item to be produced
    quantity_needed: int  # quantity to be produced
    type: str  # type of action
    tool: dict[str, int]  # tools/materials needed


class OneShotOpenAILLM:
    def __init__(self, model: LLMGeneratorBase):
        self.model = model
        self.token_used = 0
        self.history = self.model.create_initial_history(
            ONE_SHOT_SYSTEM_PROMPT, ONE_SHOT_EXAMPLE
        )

    @staticmethod
    def code_regex(text: str) -> list[str]:
        matches_with_parentheses = re.findall(
            r"\s*((?:mine|craft|smelt)\(\{.*\},\s*\{.*\}\))",
            text,
        )
        return matches_with_parentheses

    def parse_generated_plan(self, generated_plan: str) -> list[ActionStep]:
        # select the python code
        lines = self.code_regex(generated_plan)
        parsed = []
        for item in lines:
            try:
                action, parts = item.split("(")
                # Split the string by parentheses and commas
                produce, materials_and_tools = eval("(" + parts)
                assert len(produce) == 1, "Only one item can be produced at a time"
                parsed.append(
                    ActionStep(
                        **{
                            "output": list(produce.keys())[0],
                            "type": action,
                            "tool": materials_and_tools,
                            "quantity_needed": list(produce.values())[0],
                        }
                    )
                )
            except Exception:
                # ignore the error and continue
                print("Error parsing", item)
                continue
        return parsed

    def generate(self, question: str, temperature=1.0, max_tokens=256) -> str:
        self.model.reset()
        messages = [
            *self.history,
            {
                "role": "user",
                "content": f"{ONE_SHOT_SYSTEM_PROMPT}\nTASK: {question}\ninventory = {'diamond_axe'}",
            },
        ]
        response, token_used = self.model.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enforce_json=False,
        )

        self.token_used += token_used
        return response


class ReactOpenAILLM:
    def __init__(self, model: LLMGeneratorBase):
        self.model = model
        self.history = self.model.create_initial_history(
            REACT_SYSTEM_PROMPT, REACT_EXAMPLE
        )
        self.token_used = 0
        self.max_thinking_steps = 3
        self.num_thinking_steps = 0

    @staticmethod
    def is_thinking(step: str) -> bool:
        return "think" in step

    @staticmethod
    def json_regex(text: str) -> str:
        # find the first json like object and return it
        regex = r"\{\"[^{}]*\"\}"
        match = re.search(regex, text)
        if match:
            return match.group(0).replace("\_", "_")

    def parse_step(self, step: str) -> ActionStep:
        try:
            action_step = json.loads(self.json_regex(step))
            step = {
                "output": action_step["output"],
                "quantity_needed": action_step["quantity"],
                "type": action_step["type"],
            }
            materials_and_tools = {}
            if "materials" in action_step and isinstance(
                action_step["materials"], list
            ):
                materials_and_tools = {m: 1 for m in action_step["materials"]}
            if "tool" in action_step:
                materials_and_tools[action_step["tool"]] = 1

            step["tool"] = materials_and_tools
            return ActionStep(**step)
        except Exception:
            print("Error parsing", step)
            return {}

    def generate_initial_step(
        self, question: str, temperature=1.0, max_tokens=256
    ) -> str:
        self.model.reset()
        initial_message = {
            "role": "user",
            "content": f"{question}\ninventory = {'diamond_axe'}",
        }
        self.history.append(initial_message)

        thinking_step = 0
        # iterate while model is thinking:
        while thinking_step < self.max_thinking_steps:
            out_message, token_used = self.model.generate(
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
                enforce_json=True,
            )
            self.token_used += token_used

            if self.is_thinking(out_message):
                self.history.append({"role": "assistant", "content": out_message})
                self.history.append({"role": "user", "content": "OK"})
                thinking_step += 1
                self.num_thinking_steps += 1
                continue
            break

        self.history.append({"role": "assistant", "content": out_message})
        return out_message

    def generate_step(self, observation: str, temperature=1.0, max_tokens=256) -> str:
        self.history.append({"role": "user", "content": observation})
        thinking_step = 0
        # iterate while model is thinking:
        while thinking_step < self.max_thinking_steps:
            out_message, token_used = self.model.generate(
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
                enforce_json=True,
            )
            self.token_used += token_used

            if self.is_thinking(out_message):
                self.history.append({"role": "assistant", "content": out_message})
                self.history.append({"role": "user", "content": "OK"})
                thinking_step += 1
                self.num_thinking_steps += 1
                continue
            break

        self.history.append({"role": "assistant", "content": out_message})
        return out_message
