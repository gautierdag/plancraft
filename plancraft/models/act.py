import copy
import torch
from dotenv import load_dotenv

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    NoOp,
    StopAction,
    SymbolicAction,
)
from plancraft.models.base import ABCModel, History
from plancraft.models.bbox_model import IntegratedBoundingBoxModel
from plancraft.models.few_shot_images import load_prompt_images
from plancraft.models.generators import (
    OAMGenerator,
    OpenAIGenerator,
    TransformersGenerator,
)
from plancraft.models.prompts import get_prompt_example, get_system_prompt
from plancraft.models.utils import (
    convert_observation_to_message,
    parse_content_response,
)


load_dotenv()


class ActModel(ABCModel):
    """
    Model that does action without thinking step
    """

    def __init__(self, cfg: EvalConfig):
        assert (
            cfg.plancraft.environment.symbolic_action_space
        ), "Real action space unsupported"
        self.cfg = cfg
        self.env_is_multimodal = not cfg.plancraft.environment.symbolic
        self.use_maskrcnn = cfg.plancraft.use_maskrcnn
        self.use_multimodal_content_format = cfg.plancraft.use_multimodal_content_format
        self.use_text_inventory = cfg.plancraft.use_text_inventory
        self.use_images = cfg.plancraft.use_images

        self.bbox_model = None
        if self.use_maskrcnn:
            assert self.env_is_multimodal, "MaskRCNN only supported in multimodal mode"
            self.bbox_model = IntegratedBoundingBoxModel.from_pretrained(
                "gautierdag/plancraft-maskrcnn"
            )
            self.bbox_model.eval()
            if torch.cuda.is_available():
                self.bbox_model.cuda()
            # MaskRCNN is not multimodal model but a separate model

        self.few_shot = cfg.plancraft.few_shot
        self.use_system_prompt = cfg.plancraft.system_prompt
        self.max_invalid_actions = 3

        # underlying language model
        if "gpt-4o" in cfg.plancraft.model:
            self.use_multimodal_content_format = True
            self.llm = OpenAIGenerator(
                use_images=self.use_images, model_name=cfg.plancraft.model
            )
        elif "oam" in cfg.plancraft.model:
            self.llm = OAMGenerator(model_name=cfg.plancraft.model)
        else:
            # model is transformers based
            self.llm = TransformersGenerator(
                model_name=cfg.plancraft.model,
                tokenizer_name=cfg.plancraft.tokenizer,
                quantize=cfg.plancraft.quantize,
                use_hot_cache=cfg.plancraft.hot_cache,
                adapter_name=cfg.plancraft.adapter,
            )

        self.prompt_images = []

        self.valid_actions = cfg.plancraft.valid_actions
        self.system_prompt_text = get_system_prompt(self.valid_actions)

        examples = get_prompt_example(
            self.valid_actions,
            use_text_inventory=self.use_text_inventory,
            use_multimodal_content_format=self.use_multimodal_content_format,
            use_images=self.use_images,
        )
        if self.env_is_multimodal and self.use_images:
            self.prompt_images = load_prompt_images()

        if self.use_multimodal_content_format:
            self.system_prompt = {
                "role": "system",
                "content": [
                    {"text": copy.deepcopy(self.system_prompt_text), "type": "text"}
                ],
            }
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
            use_multimodal_content_format=self.use_multimodal_content_format,
        )

        self.max_messages_window = cfg.plancraft.max_message_window
        self.kv_cache = None

    def reset_history(
        self,
        objective: str,
    ):
        examples = []
        if self.few_shot:
            examples = get_prompt_example(
                self.valid_actions,
                use_text_inventory=self.use_text_inventory,
                use_multimodal_content_format=self.use_multimodal_content_format,
                use_images=self.use_images,
            )

        self.history.reset(objective=objective, initial_dialogue=examples)
        self.llm.reset()

    def step(self, observation: dict) -> SymbolicAction | StopAction:
        self.history.add_observation_to_history(observation)

        # add observation to history
        observation_message = convert_observation_to_message(
            observation,
            objective=self.history.objective,
            bbox_model=self.bbox_model,
            oam_model="oam" in self.llm.model_name,
            use_text_inventory=self.use_text_inventory,
            use_multimodal_content_format=self.use_multimodal_content_format,
            use_images=self.use_images,
        )
        self.history.add_message_to_history(content=observation_message, role="user")

        # Iterate until valid action
        i = 0
        while i < self.max_invalid_actions:
            # add observation to history
            message_window, image_window = self.llm.prepare_messages(
                history=self.history,
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            action_messages, action_token_used = self.llm.generate_unconstrained(
                batch_messages=[message_window],
                images=[image_window],
            )
            self.history.tokens_used += action_token_used

            action_message = action_messages[0].split("\n")[0].strip()

            self.history.add_message_to_history(
                content=action_message, role="assistant"
            )
            response = parse_content_response(
                action_message, valid_actions=self.valid_actions
            )
            if not isinstance(response, str):
                # valid action
                self.history.add_action_to_history(response)
                return response

            self.history.add_message_to_history(
                content=response,
            )
            i += 1

        # if no action is found after max_invalid_actions, default to useless move action
        return NoOp()
