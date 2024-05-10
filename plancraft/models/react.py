from plancraft.models.base import ABCModel

from plancraft.environments.actions import SymbolicAction

from plancraft.config import Config
from plancraft.models.prompts import (
    # REACT_EXAMPLE,
    REACT_SYSTEM_PROMPT,
)

from plancraft.llms import get_llm_generator


class ReactModel(ABCModel):
    def __init__(self, cfg: Config):
        assert cfg.plancraft.environment.symbolic_action_space

        self.llm = get_llm_generator(
            model_name=cfg.plancraft.model,
            guidance=cfg.plancraft.guidance,
            quantize=cfg.plancraft.quantize,
        )

        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
        self.action_history = []
        self.history = []
        self.token_used = 0
        self.max_thinking_steps = 1
        self.num_thinking_steps = 0
        self.max_messages_window = 50
        self.guidance = cfg.plancraft.guidance

    def set_objective(self, objective: str):
        self.objective = objective
        self.system_prompt["content"] = (
            self.system_prompt["content"] + f"\n\nCURRENT TASK: {objective}"
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
        thought_message, thinking_token_used = self.llm.generate(
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
        action_message, action_token_used = self.llm.generate(
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

    def step(self, observation: dict) -> SymbolicAction:
        # @TODO
        # 1. parse observations from json/image to text
        # 2. set up message history
        # 3. set up react example actions taken from train
        #
        return self.generate(observation)

    @property
    def trace(self) -> dict:
        return {"objective": self.objective, "action_history": self.action_history}

    def reset(self) -> None:
        self.llm.reset()
        self.action_history = []
        self.objective = ""
        self.system_prompt = {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT,
        }
