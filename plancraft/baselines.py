import re
from dataclasses import dataclass
from typing import Union

from dotenv import load_dotenv

from plancraft.prompts import (
    ONE_SHOT_SYSTEM_PROMPT,
    REACT_EXAMPLE,
    REACT_SYSTEM_PROMPT,
)

from plancraft.llms import LLMGeneratorBase, Action, Plan, Thought

load_dotenv()


@dataclass
class ActionStep:
    output: str  # item to be produced
    quantity_needed: int  # quantity to be produced
    type: str  # type of action
    tools_and_materials: list[str]  # tools/materials needed


JSON_REGEX = re.compile(r"\s*(\{.*?\})", re.DOTALL)


def parse_step(action_step: Union[str, Action, Thought]) -> ActionStep:
    try:
        # ignore thoughts
        if isinstance(action_step, Thought):
            return

        if isinstance(action_step, Action):
            tools_and_materials = []
            for material in action_step.materials:
                tools_and_materials.append(material.strip())
            if action_step.tool:
                tools_and_materials.append(action_step.tool.strip())
            return ActionStep(
                output=action_step.output,
                quantity_needed=action_step.quantity,
                type=action_step.type,
                tools_and_materials=tools_and_materials,
            )

        if "thought" in action_step:
            return

        # parse the json object if it is a json string
        action_step = eval(action_step)
        step = {
            "output": action_step["output"].strip(),
            "quantity_needed": int(action_step["quantity"]),
            "type": action_step["type"].strip(),
        }
        materials_and_tools = []
        if "materials" in action_step and isinstance(action_step["materials"], list):
            for material in action_step["materials"]:
                materials_and_tools.append(material.strip())
        if "tool" in action_step:
            materials_and_tools.append(action_step["tool"].strip())
        return ActionStep(
            output=step["output"],
            quantity_needed=step["quantity_needed"],
            type=step["type"],
            tools_and_materials=materials_and_tools,
        )
    except Exception:
        print("Error parsing", action_step)
        return


class OneShotLLM:
    def __init__(self, model: LLMGeneratorBase):
        self.model = model
        self.token_used = 0

    @staticmethod
    def parse_generation(
        generated_plan: Union[str, Plan],
    ) -> list[ActionStep]:
        parsed = []
        # check if we used constrained decoding
        if isinstance(generated_plan, Plan):
            for action in generated_plan.actions:
                parsed_action = parse_step(action)
                if parsed_action:
                    parsed.append(parsed_action)
            return parsed

        # parse the json objects from the generated plan string
        for item in JSON_REGEX.findall(generated_plan):
            action = parse_step(item)
            if action:
                parsed.append(action)
        return parsed

    def generate(self, question: str, temperature=1.0, max_tokens=256) -> str:
        self.model.reset()
        messages = [
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
            mode="plan",  # used for guidance
        )

        self.token_used += token_used
        return response


class ReactLLM:
    def __init__(self, model: LLMGeneratorBase, guidance=False):
        self.model = model
        self.history = self.model.create_initial_history(
            REACT_SYSTEM_PROMPT, REACT_EXAMPLE
        )
        self.token_used = 0
        self.max_thinking_steps = 1
        self.num_thinking_steps = 0
        self.guidance = guidance

    @staticmethod
    def is_thinking(step: str) -> bool:
        return "thought" in step

    @staticmethod
    def parse_generation(step: Union[str, Action, Thought]) -> ActionStep:
        action = parse_step(step)
        if action is None:
            print("Error parsing", step)
            return {}
        return action

    def generate(self, temperature=1.0, max_tokens=256):
        # if guidance is enabled, we force the model to think and then act
        if self.guidance:
            thought_message, token_used = self.model.generate(
                messages=self.history,
                max_tokens=max_tokens,
                mode="think",
            )
            self.history.append(
                {"role": "assistant", "content": thought_message.json()}
            )
            self.history.append({"role": "user", "content": "OK"})
            self.token_used += token_used
            self.num_thinking_steps += 1
            action_message, token_used = self.model.generate(
                messages=self.history,
                max_tokens=max_tokens,
                mode="action",
            )
            self.history.append({"role": "assistant", "content": action_message.json()})
            self.token_used += token_used
            return action_message

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

    def generate_initial_step(
        self, question: str, temperature=1.0, max_tokens=256
    ) -> str:
        self.model.reset()
        initial_message = {
            "role": "user",
            "content": f"{question}\ninventory = {'diamond_axe'}",
        }
        self.history.append(initial_message)
        return self.generate(temperature=temperature, max_tokens=max_tokens)

    def generate_step(self, observation: str, temperature=1.0, max_tokens=256) -> str:
        self.history.append({"role": "user", "content": observation})
        return self.generate(temperature=temperature, max_tokens=max_tokens)
