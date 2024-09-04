import copy
import logging

from dotenv import load_dotenv

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    StopAction,
    SymbolicAction,
    SymbolicMoveAction,
)
from plancraft.models.base import ABCModel, History

# from plancraft.models.few_shot_images import load_prompt_images
from plancraft.models.generators import (
    OpenAIGenerator,
    TransformersGenerator,
)
from plancraft.models.prompts import (
    get_prompt_example,
    get_system_prompt,
)
from plancraft.models.utils import (
    convert_observation_to_message,
    parse_content_response,
)

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
        self.prompt_images = []

        self.valid_actions = cfg.plancraft.valid_actions
        self.system_prompt_text = get_system_prompt(self.valid_actions)
        examples = get_prompt_example(self.valid_actions, self.is_multimodal)
        if self.is_multimodal:
            assert False, "Multimodal tools not supported"
            # examples = copy.deepcopy(REACT_EXAMPLE_IMGS)
            # self.prompt_images = load_prompt_images()
            # self.system_prompt = {
            #     "role": "system",
            #     "content": [
            #         {"text": copy.deepcopy(self.system_prompt_text), "type": "text"}
            #     ],
            # }
        else:
            self.system_prompt = {
                "role": "system",
                "content": copy.deepcopy(self.system_prompt_text),
            }

        if not self.few_shot:
            examples = []
        if not self.use_system_prompt:
            self.system_prompt = None

        self.history = History(
            initial_dialogue=examples,
            is_multimodal=self.is_multimodal,
        )

        self.max_messages_window = cfg.plancraft.max_message_window
        self.kv_cache = None

    def reset_history(
        self,
        objective: str,
    ):
        examples = []
        if self.few_shot:
            examples = get_prompt_example(self.valid_actions, self.is_multimodal)
        self.history.reset(objective=objective, initial_dialogue=examples)
        self.llm.reset()

    def step(self, observation: dict) -> SymbolicAction | StopAction:
        self.history.add_observation_to_history(observation)

        # add observation to history
        observation_message = convert_observation_to_message(
            observation,
            objective=self.history.objective,
            is_multimodal=self.is_multimodal,
        )
        self.history.add_message_to_history(content=observation_message, role="user")

        # Iterate until action is either smelt or move
        i = 0
        while i < self.max_non_action_tools:
            message_window, image_window = self.llm.prepare_messages(
                history=self.history,
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
            self.history.tokens_used += tokens_used

            # add message to history
            self.history.add_message_to_history(content=content, role="assistant")

            response = parse_content_response(content, valid_actions=self.valid_actions)
            if not isinstance(response, str):
                # valid action
                self.history.add_action_to_history(response)
                return response

            self.history.add_message_to_history(
                content=response,
            )
            i += 1

        # if no action is found after max_non_action_tools, default to (mostly) useless move action
        return SymbolicMoveAction(
            slot_from=0,
            slot_to=1,
            quantity=1,
        )
