import copy
import json
import logging

from dotenv import load_dotenv

from plancraft.config import EvalConfig
from plancraft.environments.actions import SymbolicAction
from plancraft.models.base import ABCModel, History
from plancraft.models.react import OpenAIGenerator, TransformersGenerator
from plancraft.models.few_shot_images import load_prompt_images
from plancraft.models.react_prompts import (
    ACT_EXAMPLE,
    ACT_EXAMPLE_IMGS,
    ACT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

load_dotenv()


class ActModel(ABCModel):
    """
    Model that does action without thinking step
    """

    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Real action space unsupported"

        self.is_multimodal = not cfg.plancraft.environment.symbolic

        # underlying language model
        if "gpt-4o" in cfg.plancraft.model:
            self.llm = OpenAIGenerator(is_multimodal=self.is_multimodal)
        # model is transformers based
        else:
            self.llm = TransformersGenerator(
                model_name=cfg.plancraft.model,
                tokenizer_name=cfg.plancraft.tokenizer,
                quantize=cfg.plancraft.quantize,
                is_multimodal=self.is_multimodal,
                use_hot_cache=cfg.plancraft.hot_cache,
            )

        self.batch_size = cfg.plancraft.batch_size
        self.prompt_images = []

        examples = copy.deepcopy(ACT_EXAMPLE)
        self.system_prompt = {
            "role": "system",
            "content": copy.deepcopy(ACT_SYSTEM_PROMPT),
        }

        if self.is_multimodal:
            examples = copy.deepcopy(ACT_EXAMPLE_IMGS)
            self.prompt_images = load_prompt_images()
            self.system_prompt = {
                "role": "system",
                "content": [{"text": copy.deepcopy(ACT_SYSTEM_PROMPT), "type": "text"}],
            }

        self.histories = [
            History(
                initial_dialogue=examples,
                is_multimodal=self.is_multimodal,
            )
            for _ in range(self.batch_size)
        ]

        self.max_messages_window = cfg.plancraft.max_message_window
        self.kv_cache = None

    def reset_history(
        self,
        history_idx: int,
        objective: str,
    ):
        examples = copy.deepcopy(ACT_EXAMPLE)
        if self.is_multimodal:
            examples = copy.deepcopy(ACT_EXAMPLE_IMGS)

        self.histories[history_idx].reset(
            objective=objective, initial_dialogue=examples
        )
        self.llm.reset()

    def convert_observation_to_message(
        self, observation: dict, objective: str
    ) -> str | dict:
        if self.is_multimodal:
            content_message = {
                "content": [
                    {"type": "text", "text": f"{objective}"},
                    {"type": "image"},
                ]
            }
            return content_message
        else:
            # if not multimodal, we only have text - we just dump a JSON of the inventory
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
            return f"{objective}\ninventory={json.dumps(inventory)}"

    def step(self, observations: list[dict]) -> list[SymbolicAction]:
        assert len(observations) == self.batch_size == len(self.histories)

        # filter out None observations
        real_obs = []
        real_obs_idx = []

        for idx, (observation, history) in enumerate(zip(observations, self.histories)):
            # add observation to history
            if observation is not None:
                # note if image is present this adds the image to the history
                history.add_observation_to_history(observation)
                real_obs.append(observation)
                real_obs_idx.append(idx)

        if len(real_obs) == 0:
            return [None] * len(observations)

        action_messages_windows = []
        action_images_windows = []
        # collect dialogue histories
        for observation, history_idx in zip(real_obs, real_obs_idx):
            # add observation to history
            observation_message = self.convert_observation_to_message(
                observation, objective=self.histories[history_idx].objective
            )
            self.histories[history_idx].add_message_to_history(
                content=observation_message, role="user"
            )
            message_window, image_window = self.llm.prepare_messages(
                history=self.histories[history_idx],
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            action_messages_windows.append(message_window)
            action_images_windows.append(image_window)

        actions, action_messages, action_token_used = self.llm.generate_actions(
            batch_messages=action_messages_windows, images=action_images_windows
        )

        for action_message, history_idx in zip(action_messages, real_obs_idx):
            self.histories[history_idx].add_message_to_history(
                content=action_message, role="assistant"
            )
            self.histories[history_idx].tokens_used += action_token_used

        # re-map actions to the correct index in the batch
        out_actions = [None] * len(observations)
        for idx, history_idx in enumerate(real_obs_idx):
            out_actions[history_idx] = actions[idx]
            # add to action history
            self.histories[history_idx].add_action_to_history(actions[idx])

        return out_actions
