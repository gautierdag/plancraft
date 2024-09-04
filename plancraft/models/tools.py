import copy
import logging
import re

from dotenv import load_dotenv

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    SymbolicAction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
    StopAction,
)
from plancraft.models.base import ABCModel, History

# from plancraft.models.few_shot_images import load_prompt_images
from plancraft.models.generators import (
    OpenAIGenerator,
    TransformersGenerator,
)
from plancraft.models.prompts import (
    TOOLS_EXAMPLE,
    # REACT_EXAMPLE_IMGS,
    TOOLS_SYSTEM_PROMPT,
)
from plancraft.models.search import gold_search_recipe
from plancraft.models.utils import convert_observation_to_message, convert_to_slot_index

logger = logging.getLogger(__name__)

load_dotenv()


class ToolsModel(ABCModel):
    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Real action space unsupported"

        self.is_multimodal = not cfg.plancraft.environment.symbolic
        self.few_shot = cfg.plancraft.few_shot
        self.use_system_prompt = cfg.plancraft.system_prompt
        self.max_non_action_tools = 3

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
            assert False, "Multimodal tools not supported"
            # examples = copy.deepcopy(REACT_EXAMPLE_IMGS)
            # self.prompt_images = load_prompt_images()
            # self.system_prompt = {
            #     "role": "system",
            #     "content": [
            #         {"text": copy.deepcopy(TOOLS_SYSTEM_PROMPT), "type": "text"}
            #     ],
            # }
        else:
            examples = copy.deepcopy(TOOLS_EXAMPLE)
            self.system_prompt = {
                "role": "system",
                "content": copy.deepcopy(TOOLS_SYSTEM_PROMPT),
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
            # if self.is_multimodal:
            # examples = copy.deepcopy(REACT_EXAMPLE_IMGS)
            # else:
            examples = copy.deepcopy(TOOLS_EXAMPLE)
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

        actions = []
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

            # Iterate until action is either smelt or move
            i = 0
            done = False
            while i < self.max_non_action_tools and not done:
                message_window, image_window = self.llm.prepare_messages(
                    history=self.histories[history_idx],
                    max_messages_window=self.max_messages_window,
                    system_prompt=self.system_prompt,
                    prompt_images=self.prompt_images,
                )
                text_response, tokens_used = self.llm.generate_unconstrained(
                    batch_messages=[message_window],
                    max_tokens=256,
                    images=[image_window],
                )
                content = text_response[0].split("\n")[0].strip()

                print(content)

                # increment token used
                self.histories[history_idx].tokens_used += tokens_used

                # add message to history
                self.histories[history_idx].add_message_to_history(
                    content=content, role="assistant"
                )

                # for tool in self.tools:
                tool_match = re.search(
                    r"^(smelt|move|search|think|stop):", content.strip()
                )
                if tool_match:
                    tool = tool_match.group(1)
                    if tool == "think":
                        self.histories[history_idx].add_message_to_history(
                            content="Ok", role="user"
                        )
                    elif tool == "stop":
                        reason = re.search(r"stop: (.*)", content).group(1)
                        actions.append(StopAction(reason=reason))
                        done = True
                    elif tool == "search":
                        search_target = re.search(r"search: (\w+)", content).group(1)
                        search_response = gold_search_recipe(search_target)
                        self.histories[history_idx].add_message_to_history(
                            content=search_response, role="user"
                        )
                    else:
                        try:
                            slot_from = re.search(
                                r" from (\[[ABCI]?\d+\])", content
                            ).group(1)
                            slot_to = re.search(r" to (\[[ABCI]?\d+\])", content).group(
                                1
                            )
                            slot_from = convert_to_slot_index(slot_from)
                            slot_to = convert_to_slot_index(slot_to)

                            quantity = re.search(r"with quantity (\d+)", content).group(
                                1
                            )
                            if tool == "move":
                                action = SymbolicMoveAction(
                                    slot_from=slot_from,
                                    slot_to=slot_to,
                                    quantity=quantity,
                                )
                            else:
                                action = SymbolicSmeltAction(
                                    slot_from=slot_from,
                                    slot_to=slot_to,
                                    quantity=quantity,
                                )
                            actions.append(action)
                            done = True
                        except AttributeError as e:
                            error_message = f"Could not parse action: {e}"
                            self.histories[history_idx].add_message_to_history(
                                content=error_message, role="user"
                            )
                else:
                    self.histories[history_idx].add_message_to_history(
                        content="Please start your message with a valid action from the following: smelt, move, search, think",
                    )
                i += 1
                if not done and i >= self.max_non_action_tools:
                    actions.append(
                        SymbolicMoveAction(
                            slot_from=0,
                            slot_to=1,
                            quantity=1,
                        )
                    )

            assert len(actions) == len(
                real_obs_idx
            ), "Mismatch in number of actions and observations"

            # re-map actions to the correct index in the batch
            out_actions = [None] * len(observations)
            for idx, history_idx in enumerate(real_obs_idx):
                out_actions[history_idx] = actions[idx]
                # add to action history
                self.histories[history_idx].add_action_to_history(actions[idx])

        return out_actions
