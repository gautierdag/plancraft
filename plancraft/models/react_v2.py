import copy
import logging

from dotenv import load_dotenv

from plancraft.config import EvalConfig
from plancraft.environments.actions import SymbolicAction
from plancraft.models.base import ABCModel, History
from plancraft.models.few_shot_images import load_prompt_images
from plancraft.models.generators import (
    OpenAIGenerator,
    TransformersGenerator,
)
from plancraft.models.prompts import (
    REACT_EXAMPLE,
    REACT_EXAMPLE_IMGS,
    SYSTEM_PROMPT,
)
from plancraft.models.utils import convert_observation_to_message

logger = logging.getLogger(__name__)

load_dotenv()


class ReactModel(ABCModel):
    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Real action space unsupported"

        self.is_multimodal = not cfg.plancraft.environment.symbolic
        self.few_shot = cfg.plancraft.few_shot
        self.use_system_prompt = cfg.plancraft.system_prompt

        # underlying language model
        if "gpt-4o" in cfg.plancraft.model:
            self.llm = OpenAIGenerator(
                is_multimodal=self.is_multimodal, model_name=cfg.plancraft.model
            )
        # model is transformers based
        else:
            self.llm = TransformersGenerator(
                model_name=cfg.plancraft.model,
                tokenizer_name=cfg.plancraft.tokenizer,
                quantize=cfg.plancraft.quantize,
                is_multimodal=self.is_multimodal,
                use_hot_cache=cfg.plancraft.hot_cache,
                adapter_name=cfg.plancraft.adapter,
            )

        self.batch_size = cfg.plancraft.batch_size
        self.prompt_images = []

        if self.is_multimodal:
            examples = copy.deepcopy(REACT_EXAMPLE_IMGS)
            self.prompt_images = load_prompt_images()
            self.system_prompt = {
                "role": "system",
                "content": [{"text": copy.deepcopy(SYSTEM_PROMPT), "type": "text"}],
            }
        else:
            examples = copy.deepcopy(REACT_EXAMPLE)
            self.system_prompt = {
                "role": "system",
                "content": copy.deepcopy(SYSTEM_PROMPT),
            }

        if not self.few_shot:
            examples = []
        if not self.use_system_prompt:
            self.system_prompt = None

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
        examples = []
        if self.few_shot:
            if self.is_multimodal:
                examples = copy.deepcopy(REACT_EXAMPLE_IMGS)
            else:
                examples = copy.deepcopy(REACT_EXAMPLE)
        self.histories[history_idx].reset(
            objective=objective, initial_dialogue=examples
        )
        self.llm.reset()

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

        # thought_messages_windows = []
        # thought_images_windows = []
        # collect dialogue histories
        for observation, history_idx in zip(real_obs, real_obs_idx):
            # add observation to history
            observation_message = convert_observation_to_message(
                observation,
                objective=self.histories[history_idx].objective,
                is_multimodal=self.is_multimodal,
            )
            self.histories[history_idx].add_message_to_history(
                content=observation_message, role="user"
            )

            # @TODO iterate until action is either smelt or move
            message_window, image_window = self.llm.prepare_messages(
                history=self.histories[history_idx],
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            thought_messages_windows.append(message_window)
            thought_images_windows.append(image_window)

            # generate thoughts
            # thought_messages, thinking_token_used = self.llm.generate_thoughts(
            #     batch_messages=thought_messages_windows,
            #     max_tokens=256,
            #     images=thought_images_windows,
            # )

            # action_messages_windows = []
            # action_images_windows = []
            # update message window with thoughts and collect action messages
            for thought_message, history_idx in zip(thought_messages, real_obs_idx):
                print(thought_message)
                # add thought message to history
                self.histories[history_idx].add_message_to_history(
                    content=thought_message, role="assistant"
                )
                self.histories[history_idx].add_message_to_history(
                    content="Ok", role="user"
                )
                # add token used to history
                self.histories[history_idx].tokens_used += thinking_token_used

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
                print(action_message)
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
