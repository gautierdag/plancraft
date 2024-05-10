import re
from dataclasses import dataclass
from typing import Union
from copy import deepcopy
import json

from plancraft.prompts import (
    ONE_SHOT_SYSTEM_PROMPT,
    REACT_EXAMPLE,
    REACT_SYSTEM_PROMPT,
)

from plancraft.llms import LLMGeneratorBase, Action, Plan, Thought


@dataclass
class ActionStep:
    output: str  # item to be produced
    quantity_needed: int  # quantity to be produced
    type: str  # type of action
    tools_and_materials: list[str]  # tools/materials needed


JSON_REGEX = re.compile(r"\s*(\{.*?\})", re.DOTALL)


def get_dict_from_text(step: str) -> dict:
    try:
        return json.loads(step)
    except json.JSONDecodeError:
        pass
    try:
        return eval(step)
    except Exception:
        pass
    return {}


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

        # clean some escape characters
        action_step = action_step.replace(r"\_", "_")
        # parse into a dict
        action_step_dict = get_dict_from_text(action_step)

        step = {
            "output": action_step_dict["output"].strip(),
            "quantity_needed": int(action_step_dict["quantity"]),
            "type": action_step_dict["type"].strip(),
        }
        materials_and_tools = []
        if "materials" in action_step_dict and isinstance(
            action_step_dict["materials"], list
        ):
            for material in action_step_dict["materials"]:
                materials_and_tools.append(material.strip())
        if "tool" in action_step_dict:
            materials_and_tools.append(action_step_dict["tool"].strip())
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
            mode="plan",  # used for guidance
        )

        self.token_used += token_used
        return response


class ReactLLM:
    def __init__(self, model: LLMGeneratorBase, guidance=False):
        self.model = model
        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
        self.history = deepcopy(REACT_EXAMPLE)

        self.token_used = 0
        self.max_thinking_steps = 1
        self.num_thinking_steps = 0
        self.guidance = guidance
        self.max_messages_window = 50

    @staticmethod
    def parse_generation(step: Union[str, Action, Thought]) -> ActionStep:
        if isinstance(step, str):
            match = JSON_REGEX.findall(step)
            if len(match) < 1:
                print("Error parsing", step)
                return {}
            action = parse_step(match[0])
        else:
            action = parse_step(step)

        if action is None:
            print("Error parsing", step)
            return {}
        return action

    def generate(self, max_tokens=256):
        # get the N last messages from the history
        message_window = self.history[-self.max_messages_window :]
        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]

        # add the system prompt to the message window
        message_window = [self.system_prompt] + message_window

        # we force the model to think and then act
        thought_message, thinking_token_used = self.model.generate(
            messages=message_window,
            max_tokens=max_tokens,
            mode="think",  # used for guidance
            enforce_json='{"thought":',
        )
        if self.guidance:
            thought_chat_message = {
                "role": "assistant",
                "content": thought_message.json(),
            }
        else:
            thought_chat_message = {"role": "assistant", "content": thought_message}

        # add the thought message to the history and window
        self.history.append(thought_chat_message)
        message_window.append(thought_chat_message)
        self.history.append({"role": "user", "content": "OK"})
        message_window.append({"role": "user", "content": "OK"})

        self.num_thinking_steps += 1
        action_message, action_token_used = self.model.generate(
            messages=message_window,
            max_tokens=max_tokens,
            mode="action",  # used for guidance
            enforce_json='{"type":',
        )

        if self.guidance:
            action_chat_message = {
                "role": "assistant",
                "content": action_message.json(),
            }
        else:
            action_chat_message = {"role": "assistant", "content": action_message}

        self.history.append(action_chat_message)
        self.token_used += thinking_token_used + action_token_used
        print(
            f"Thinking token used: {thinking_token_used}, Action token used: {action_token_used}, Total token used: {thinking_token_used+action_token_used}"
        )
        return action_message

    def generate_initial_step(self, question: str, max_tokens=128) -> str:
        self.model.reset()
        initial_message = {
            "role": "user",
            "content": f"TASK: {question}\ninventory = {'diamond_axe'}",
        }
        self.history.append(initial_message)

        # add current task to the system prompt
        self.system_prompt["content"] = (
            self.system_prompt["content"] + f"\n\nCURRENT TASK: {question}"
        )

        return self.generate(max_tokens=max_tokens)

    def generate_step(self, observation: str, max_tokens=128) -> str:
        self.history.append({"role": "user", "content": observation})
        return self.generate(max_tokens=max_tokens)


# import re
# from dataclasses import dataclass
# from typing import Union
# from copy import deepcopy
# import json

# from plancraft.prompts import (
#     REACT_EXAMPLE,
#     REACT_SYSTEM_PROMPT,
# )

# class ReactLLM(BaseModel):
#     def __init__(self, model, symbolic):
#         self.model = model
#         self.system_prompt = {
#             "role": "system",
#             "content": REACT_SYSTEM_PROMPT,
#         }
#         self.history = deepcopy(REACT_EXAMPLE)

#         self.token_used = 0
#         self.max_thinking_steps = 1
#         self.num_thinking_steps = 0
#         self.max_messages_window = 50

#     def generate(self, max_tokens=256):
#         # get the N last messages from the history
#         message_window = self.history[-self.max_messages_window :]
#         # remove the first assistant message if it is present
#         if len(message_window) > 0 and message_window[0]["role"] == "assistant":
#             message_window = message_window[1:]

#         # add the system prompt to the message window
#         message_window = [self.system_prompt] + message_window

#         # we force the model to think and then act
#         thought_message, thinking_token_used = self.model.generate(
#             messages=message_window,
#             max_tokens=max_tokens,
#             mode="think",  # used for guidance
#             enforce_json='{"thought":',
#         )
#         if self.guidance:
#             thought_chat_message = {
#                 "role": "assistant",
#                 "content": thought_message.json(),
#             }
#         else:
#             thought_chat_message = {"role": "assistant", "content": thought_message}

#         # add the thought message to the history and window
#         self.history.append(thought_chat_message)
#         message_window.append(thought_chat_message)
#         self.history.append({"role": "user", "content": "OK"})
#         message_window.append({"role": "user", "content": "OK"})

#         self.num_thinking_steps += 1
#         action_message, action_token_used = self.model.generate(
#             messages=message_window,
#             max_tokens=max_tokens,
#             mode="action",  # used for guidance
#             enforce_json='{"type":',
#         )

#         if self.guidance:
#             action_chat_message = {
#                 "role": "assistant",
#                 "content": action_message.json(),
#             }
#         else:
#             action_chat_message = {"role": "assistant", "content": action_message}

#         self.history.append(action_chat_message)
#         self.token_used += thinking_token_used + action_token_used
#         print(
#             f"Thinking token used: {thinking_token_used}, Action token used: {action_token_used}, Total token used: {thinking_token_used+action_token_used}"
#         )
#         return action_message

#     def generate_initial_step(self, question: str, max_tokens=128) -> str:
#         self.model.reset()
#         initial_message = {
#             "role": "user",
#             "content": f"TASK: {question}\ninventory = {'diamond_axe'}",
#         }
#         self.history.append(initial_message)

#         # add current task to the system prompt
#         self.system_prompt["content"] = (
#             self.system_prompt["content"] + f"\n\nCURRENT TASK: {question}"
#         )

#         return self.generate(max_tokens=max_tokens)

#     def generate_step(self, observation: str, max_tokens=128) -> str:
#         self.history.append({"role": "user", "content": observation})
#         return self.generate(max_tokens=max_tokens)
