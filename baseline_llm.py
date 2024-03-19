import json
import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class LLMClient:
    def __init__(self, model_name="gpt-3.5-turbo", dtype=torch.bfloat16):
        self.model_name = model_name
        # openAI models
        if model_name in ["gpt-3.5-turbo", "gpt-4.0-turbo-preview"]:
            self.client_type = "openai"
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            self.client_type = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )

    def generate(
        self, messages: list[dict], temperature=1.0, max_tokens=512, enforce_json=False
    ) -> tuple[str, int]:
        kwargs = {}
        if self.client_type == "openai":
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

        elif self.client_type == "transformers":
            message_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if enforce_json:
                # start json object in prompt text
                message_text += '{"'

            tokenized_messages = self.tokenizer.encode(
                message_text,
                return_tensors="pt",
            )

            tokenized_messages = tokenized_messages.to(self.model.device)
            _, prompt_tokens = tokenized_messages.shape
            response = self.model.generate(
                tokenized_messages,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )

            text_response = self.tokenizer.decode(
                response[0, prompt_tokens:],
                skip_special_tokens=True,
            )
            _, total_tokens = response.shape
            return text_response, total_tokens
        else:
            raise ValueError(f"Client type {self.client_type} not supported")


@dataclass
class ActionStep:
    output: str  # item to be produced
    quantity_needed: int  # quantity to be produced
    type: str  # type of action
    tool: dict[str, int]  # tools/materials needed


class OneShotOpenAILLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.system_prompt = {
            "role": "system",
            "content": """You are a helper AI agent in Minecraft.

You need to generate the sequences of sub-goals (actions) for a planning task in Minecraft.

You have the choice to use the following commands:
- mine
- craft
- smelt

The first argument is a dict of items to be obtained, and the second argument is a dict of items to be used.
""",
        }
        self.example = [
            {
                "role": "user",
                "content": """How to obtain iron_ingot?\ninventory = {'diamond_axe'}""",
            },
            {
                "role": "assistant",
                "content": """def obtain_iron_ingot(inventory):
\tmine({'log': 3}, {'diamond_axe': 1})
\tcraft({'planks': 11}, {'log': 3})
\tcraft({'stick': 6}, {'planks': 2})
\tcraft({'crafting_table': 1}, {'planks': 4})
\tcraft({'wooden_pickaxe': 1}, {'planks': 3, 'stick': 2, 'crafting_table': 1})
\tmine({'cobblestone': 11}, {'wooden_pickaxe': 1})
\tcraft({'stone_pickaxe': 1}, {'cobblestone': 3, 'stick': 2, 'crafting_table': 1})
\tmine({'iron_ore': 1}, {'stone_pickaxe': 1})
\tcraft({'furnace': 1}, {'cobblestone': 8})
\tsmelt({'iron_ingot': 1}, {'iron_ore': 1, 'furnace': 1})
\treturn 'iron_ingot'""",
            },
        ]
        self.model = LLMClient(model_name=model_name)
        self.token_used = 0

    @staticmethod
    def parse_generated_plan(generated_plan: str) -> list[ActionStep]:
        # select the python code
        if "```python" in generated_plan:
            generated_plan = generated_plan.split("```python")[1].split("```")[0]

        lines = generated_plan.split("\n")

        parsed = []
        for item in lines:
            try:
                # ignore the function definition and return statement
                if "def " in item or "return " in item:
                    continue
                # ignore empty lines
                if item.strip(" ") == "":
                    continue
                # ignore comments
                if "#" in item:
                    item = item.split("#")[0]
                item = item.strip(" ").replace("\t", "").split("(")
                # ignore the line if it can't be split into two parts
                if len(item) == 1:
                    continue

                action, parts = item

                if action not in ["mine", "craft", "smelt"]:
                    continue

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

    def generate(self, question: str, temperature=1.0, max_tokens=512) -> str:
        messages = [
            self.system_prompt,
            *self.example,
            {
                "role": "user",
                "content": f"{question}\ninventory = {'diamond_axe'}",
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
    def __init__(self, model="gpt-3.5-turbo"):
        self.system_prompt = [
            {
                "role": "system",
                "content": """You are a helper AI agent in Minecraft.

You need to decide on the next action to accomplish the planning TASK in Minecraft.

You must output a JSON object with the following:
{
"output": "iron_pickaxe",
"quantity": 1 # quantity to be produced
"type": "craft",
"materials": ["iron_ingot", "stick"] # optional list of materials
"tool": "crafting_table" # optional tool
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
""",
            },
        ]

        self.example = [
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
        self.model = LLMClient(model_name=model)

        self.history = self.system_prompt + self.example

        self.token_used = 0
        self.max_thinking_steps = 3
        self.num_thinking_steps = 0

    @staticmethod
    def is_thinking(step: str) -> bool:
        return "think" in step

    @staticmethod
    def parse_step(step: str) -> ActionStep:
        try:
            action_step = json.loads(step)
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
        self, question: str, temperature=1.0, max_tokens=512
    ) -> str:
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

    def generate_step(self, observation: str, temperature=1.0, max_tokens=512) -> str:
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
