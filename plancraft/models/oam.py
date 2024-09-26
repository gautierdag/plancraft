import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from plancraft.models.bbox_model import IntegratedBoundingBoxModel

logger = logging.getLogger(__name__)


class PlancraftOAMConfig(PretrainedConfig):
    model_type = "plancraft-aom"
    is_composition = True

    def __init__(
        self,
        use_cache=True,
        from_llama=False,
        **kwargs,
    ):
        self.use_cache = use_cache
        self.from_llama = from_llama
        super().__init__(**kwargs)


class PlancraftOAM(PreTrainedModel):
    config_class = PlancraftOAMConfig

    def __init__(self, config: PlancraftOAMConfig):
        super().__init__(config)

        self.config = config
        # load text model
        if self.config.from_llama:
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
            )
        else:
            text_model_config = AutoConfig.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
            )
            self.text_model = AutoModelForCausalLM.from_config(text_model_config)

        self.lm_head = self.text_model.get_output_embeddings()

        # load vision model
        self.vision_model = IntegratedBoundingBoxModel.from_pretrained(
            "gautierdag/plancraft-maskrcnn"
        )
        self.vision_model.eval()

        # convert vision features to text embedding
        self.vision_to_text_embedding = nn.Linear(
            1024, self.text_model.config.hidden_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            trust_remote=True,
        )
        # add special tokens
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|inventory|>",
                ]
            }
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inventory_idx = self.tokenizer.convert_tokens_to_ids("<|inventory|>")
        self.bos_token_id = self.tokenizer.eos_token_id

        # resize token embeddings
        self.text_model.resize_token_embeddings(len(self.tokenizer))
        # image transforms
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @torch.no_grad()
    def extract_bboxes(self, images: list) -> list[dict]:
        if len(images) == 0:
            return []
        img_tensors = torch.stack([self.transforms(img) for img in images])
        img_tensors = img_tensors.cuda()
        # disable gradients
        self.vision_model.freeze()
        # get bounding box predictions
        bbox_preds = self.vision_model(img_tensors)
        return bbox_preds

    def prepare_messages(self, messages: list, bboxes: list[dict]) -> str:
        # no bounding boxes
        if len(bboxes) == 0:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=not self.training, tokenize=False
            )
            text = text.replace("<|begin_of_text|>", "")
            return text

        # expand <|inventory|> tokens into N tokens (N = number of bounding boxes)
        new_messages = []
        i_pred = 0
        for m in messages:
            new_message = m.copy()
            if new_message["role"] == "user" and new_message["content"].endswith(
                "<|inventory|>"
            ):
                # add inventory tokens for each bounding box
                new_message["content"] = new_message["content"].replace(
                    "<|inventory|>",
                    "<|inventory|>" * (bboxes[i_pred]["features"].shape[0]),
                )
                i_pred += 1
            new_messages.append(new_message)
        assert i_pred == len(
            bboxes
        ), "Number of inventory tokens does not match number of bounding boxes"
        # add special tokens

        text = self.tokenizer.apply_chat_template(
            new_messages, add_generation_prompt=not self.training, tokenize=False
        )
        text = text.replace("<|begin_of_text|>", "")
        return text

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        # along batch dimension
        for i in range(len(image_hidden_states)):
            if len(image_hidden_states[i]) == 0:
                assert (
                    input_ids[i] == self.inventory_idx
                ).sum() == 0, "No images but inventory token is still present"
                continue

            # count the number of inventory tokens
            n_inventory_tokens = (input_ids[i] == self.inventory_idx).sum()
            if n_inventory_tokens != image_hidden_states[i].shape[0]:
                logger.warning(
                    f"Number of inventory tokens ({n_inventory_tokens}) does not match number of bounding boxes ({image_hidden_states[i].shape[0]}). Possible truncation."
                )
                # truncated from the start
                image_hidden_states[i] = image_hidden_states[i][-n_inventory_tokens:]

            # replace inventory tokens with bbox features
            inputs_embeds[i, input_ids[i] == self.inventory_idx] = image_hidden_states[
                i
            ]
        return inputs_embeds

    def process_inputs(
        self,
        batch_messages: list[list[dict]] = [],  # list of list of messages (untokenized)
        batch_images: list[list] = [],  # list of list of images (unprocessed)
    ) -> tuple[dict[str, torch.FloatTensor], list[torch.FloatTensor], int]:
        """
        Converts raw images and messages into model inputs
        """
        assert len(batch_images) == len(
            batch_messages
        ), "Number of images and messages should match in the batch dim"
        # initial forward pass
        texts_batch = []
        image_hidden_states = []
        total_boxes = 0
        for images, messages in zip(batch_images, batch_messages):
            # process images
            bboxes = self.extract_bboxes(images)
            if len(bboxes) > 0:
                # get bbox features
                features = torch.concat([p["features"] for p in bboxes], dim=0)
                # upscale to text embedding size
                features_embeds = self.vision_to_text_embedding(features)
                image_hidden_states.append(features_embeds)
                # count bboxes total
                total_boxes += features.shape[0]
            else:
                image_hidden_states.append([])

            # process messages
            text = self.prepare_messages(messages, bboxes)
            texts_batch.append(text)

        # tokenize text
        # @NOTE: truncation might cause issues with inventory tokens not matching number of boxes
        # in that case, we will truncate the boxes from the end, and issue a warning
        batch = self.tokenizer(
            texts_batch,
            truncation=True,
            padding=True,
            max_length=16384,
            return_tensors="pt",
        )
        return batch, image_hidden_states, total_boxes

    def forward(
        self,
        batch_messages: list[list[dict]] = [],  # list of list of messages (untokenized)
        batch_images: list[list] = [],  # list of list of images (unprocessed)
        **kwargs,
    ):
        labels = None
        batch, image_hidden_states, total_boxes = self.process_inputs(
            batch_messages, batch_images
        )
        # move to cuda
        batch = {k: v.cuda() for k, v in batch.items()}
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]
        if self.training:
            labels = input_ids.clone()
            # remove inventory tokens from labels
            labels[labels == self.inventory_idx] = -100
            # sanity check: should have same number of boxes as inventory tokens
            assert (labels == -100).sum() == total_boxes

        # get text embeddings
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        inputs_embeds = self.inputs_merger(
            input_ids, inputs_embeds, image_hidden_states
        )
        # forward pass
        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        batch_messages: list[list[dict]],
        batch_images: list[list],
        do_sample=True,
        temperature=0.6,
        max_new_tokens=32,
        return_dict_in_generate=True,
        use_cache=True,
    ):
        self.training = False
        self.tokenizer.padding_side = "left"

        batch, image_hidden_states, _ = self.process_inputs(
            batch_messages, batch_images
        )
        batch = {k: v.cuda() for k, v in batch.items()}
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        inputs_embeds = self.inputs_merger(
            input_ids, inputs_embeds, image_hidden_states
        )

        generated_sequences = self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=use_cache,
        )

        # Decode the output
        text_responses = self.tokenizer.batch_decode(
            generated_sequences,
            # generated_sequences[:, prompt_tokens:],
            skip_special_tokens=False,
        )

        # remove <|eot_id|> tokens
        text_responses = [
            text_response.replace("<|eot_id|>", "") for text_response in text_responses
        ]

        _, total_tokens_used = generated_sequences.shape
        return text_responses, total_tokens_used
