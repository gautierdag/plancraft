import torch

from plancraft.config import EvalConfig
from plancraft.models.base import PlancraftBaseModel
from plancraft.models.bbox_model import IntegratedBoundingBoxModel
from plancraft.models.generators import (
    OpenAIGenerator,
    TransformersGenerator,
)

from plancraft.utils import History


class ActModel(PlancraftBaseModel):
    """
    Model that does action without thinking step
    """

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.use_fasterrcnn = cfg.plancraft.use_fasterrcnn
        self.use_multimodal_content_format = cfg.plancraft.use_multimodal_content_format
        self.use_text_inventory = cfg.plancraft.use_text_inventory
        self.use_images = cfg.plancraft.use_images

        self.bbox_model = None
        if self.use_fasterrcnn:
            # fasterrcnn is not multimodal model but a separate model
            self.bbox_model = IntegratedBoundingBoxModel.from_pretrained(
                "gautierdag/plancraft-fasterrcnn"
            )
            self.bbox_model.eval()
            if torch.cuda.is_available():
                self.bbox_model.cuda()

        # underlying language model
        if "gpt-4o" in cfg.plancraft.model:
            self.use_multimodal_content_format = True
            self.llm = OpenAIGenerator(
                use_images=self.use_images,
                model_name=cfg.plancraft.model,
                api_key=cfg.env_variables.openai_api_key,
            )
        else:
            # model is transformers based
            self.llm = TransformersGenerator(
                model_name=cfg.plancraft.model,
                tokenizer_name=cfg.plancraft.tokenizer,
                quantize=cfg.plancraft.quantize,
                use_hot_cache=cfg.plancraft.hot_cache,
                adapter_name=cfg.plancraft.adapter,
                hf_token=cfg.env_variables.hf_token,
            )
        self.max_messages_window = cfg.plancraft.max_message_window
        self.kv_cache = None

    def reset(self):
        self.llm.reset()

    def step(self, observation: dict, dialogue_history: History) -> str:
        # get message window
        message_window, image_window = self.llm.prepare_messages(
            history=dialogue_history,
            max_messages_window=self.max_messages_window,
        )
        # generate next action
        action_messages, action_token_used = self.llm.generate_unconstrained(
            batch_messages=[message_window],
            images=[image_window],
        )
        # update tokens used
        dialogue_history.tokens_used += action_token_used
        # return raw action message
        return action_messages[0].split("\n")[0].strip()
