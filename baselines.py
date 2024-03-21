import json
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from prompts import (
    ONE_SHOT_EXAMPLE,
    ONE_SHOT_SYSTEM_PROMPT,
    REACT_EXAMPLE,
    REACT_SYSTEM_PROMPT,
)

from llms import LLMGeneratorBase

load_dotenv()


@dataclass
class ActionStep:
    output: str  # item to be produced
    quantity_needed: int  # quantity to be produced
    type: str  # type of action
    tool: dict[str, int]  # tools/materials needed


class OneShotOpenAILLM:
    def __init__(self, model: LLMGeneratorBase, guidance=False):
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
        return "thought" in step

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
