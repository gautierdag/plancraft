import time

import torch
from loguru import logger
from openai import OpenAI
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache

from plancraft.models.utils import (
    get_downloaded_models,
    numpy_to_base64,
    tokenize,
)
from plancraft.utils import History


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = "same",
        quantize=False,
        use_images=False,
        use_hot_cache=True,
        adapter_name="",
        hf_token=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.use_hot_cache = use_hot_cache
        self.hf_token = hf_token

        if tokenizer_name == "same":
            tokenizer_name = model_name

        self.use_images = use_images
        model_name, model_kwargs = self.build_model_kwargs(
            model_name, quantize=quantize
        )
        self.processor = None
        if "idefics" in model_name:
            assert use_images, "Idefics model requires multimodal input"
            self.tokenizer = AutoProcessor.from_pretrained(
                tokenizer_name,
                **model_kwargs,
            )
            self.tokenizer.eos_token_id = self.tokenizer.tokenizer.eos_token_id
            logger.info("Loading model")
            time_now = time.time()
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                device_map="auto",
                **model_kwargs,
            )
            logger.info(f"Model loaded in {time.time() - time_now:.2f} seconds")
            # set pad_token_id
            if self.tokenizer.tokenizer.pad_token_id:
                self.pad_token_id = self.tokenizer.tokenizer.pad_token_id
            else:
                self.pad_token_id = self.tokenizer.tokenizer.eos_token_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=self.hf_token,  # trust_remote_code=True
                padding_side="left",  # ensure that the padding is on the left
            )
            logger.info("Loading model")
            time_now = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                **model_kwargs,
            )
            logger.info(f"Model loaded in {time.time() - time_now:.2f} seconds")

            # load OA adapter
            if adapter_name != "":
                logger.info(f"Loading adapter and tokenizer from {adapter_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    adapter_name,
                    padding_side="left",
                )
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.load_adapter(adapter_name)

            # set pad_token_id
            if self.tokenizer.pad_token_id:
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.pad_token_id = self.tokenizer.eos_token_id

        # compile
        time_now = time.time()
        self.model = torch.compile(self.model)
        logger.info(f"Model compiled in {time.time() - time_now:.2f} seconds")

        self.model.eval()
        if self.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.truncation_side = "left"

        self.past_key_values_kwargs = {}
        self.past_token_ids = None

    def truncate_kv_cache(self, new_token_ids: torch.Tensor):
        """
        Truncate the key-value cache to the size which overlap the past_ids with the new_ids.
        Uses:
            past_ids: torch.Tensor [B, T]
            new_ids: torch.Tensor [B, T]
            kv_cache: tuple[tuple[torch.Tensor]]: tuple of key-value cache tensors

        NOTE: this essentially implements System Prompt in the worst case when using batch_size==1
        """
        if (
            self.past_token_ids is None
            or "past_key_values" not in self.past_key_values_kwargs
        ):
            return

        # caching doesn't seem to work with multimodal models
        if self.use_images:
            self.past_key_values_kwargs = {}
            return

        past_batch_size, past_seq_len = self.past_token_ids.shape
        new_batch_size, new_seq_len = new_token_ids.shape

        # If the batch size has changed, reset the cache
        if past_batch_size != new_batch_size:
            self.past_key_values_kwargs = {}
            return

        min_shape = min(past_seq_len, new_seq_len)
        compare_past = (
            self.past_token_ids[:, :min_shape] != new_token_ids[:, :min_shape]
        )

        # All tokens are the same - no need to truncate
        if not compare_past.any():
            return

        # Find the first token that is different between the past and new tokens
        seq_min = torch.argmax(compare_past.double(), dim=1).min()

        # Truncate the key-value cache to the size which overlap the past_ids with the new_ids.
        # assumes shape is [num_layers, num_heads, seq_len, hidden_size]
        self.past_key_values_kwargs["past_key_values"] = [
            [kv[:, :, :seq_min, :] for kv in kvs]
            for kvs in self.past_key_values_kwargs["past_key_values"]
        ]

    def build_model_kwargs(self, model_name: str, **kwargs) -> tuple[str, dict]:
        model_kwargs = {
            "token": self.hf_token,
            # "attn_implementation": "flash_attention_2",
            # "trust_remote_code": True,
        }
        quantize = kwargs.get("quantize", False)
        if quantize == "int4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantize == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        downloaded_models = get_downloaded_models()
        if model_name in downloaded_models:
            model_kwargs["local_files_only"] = True
            model_name = downloaded_models[model_name]
            logger.info(f"Using local model {model_name}")
        if "/plancraft/outputs" in model_name:
            model_kwargs["local_files_only"] = True
            logger.info(f"Using local model {model_name}")

        return model_name, model_kwargs

    def reset(self):
        # NOTE: past_key_values cache with a rolling window
        #  is not maximally useful as the beggining shifts over time
        #  and therefore cache is invalidated
        self.past_key_values_kwargs = {}
        self.past_token_ids = None

    def prepare_messages(
        self,
        history: History,
        max_messages_window: int,
    ) -> tuple[list[dict], list]:
        """
        Prepare the messages using a history
        """
        message_window = history.dialogue_history[-max_messages_window:]
        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]
        # add the system prompt if the first message is not a system message
        if message_window[0]["role"] != "system":
            message_window = [history.system_prompt_dialogue] + message_window

        image_window = []
        if self.use_images:
            image_count = 0
            # iterate through the messages in reverse order to assign images
            for m in message_window:
                for content in m["content"]:
                    if content["type"] == "image":
                        image_count += 1
            assert image_count <= len(history.images), "Too many images"
            image_window = history.images[-image_count:]
            image_window = [Image.fromarray(img) for img in image_window]

        return message_window, image_window

    @torch.inference_mode()
    def generate_unconstrained(
        self,
        batch_messages: list[list[dict]],
        start_messages_generation: str = "",
        max_tokens: int = 256,
        temperature=0.6,
        **kwargs,
    ) -> tuple[list[str], int]:
        """
        Generate unconstrained text based on the batch of messages.
        """
        if self.use_images:
            assert "images" in kwargs, "Images required for multimodal model"

        tokenized_messages = tokenize(
            self.model,
            self.tokenizer,
            batch_messages,
            start_messages_generation=[start_messages_generation] * len(batch_messages),
            max_tokens=max_tokens,
            images=kwargs.get("images") if self.use_images else None,
        )
        prompt_tokens = tokenized_messages["input_ids"].shape[-1]

        # Sent to the same device as model
        tokenized_messages = {
            k: v.to(self.model.device) for k, v in tokenized_messages.items()
        }

        # Truncate the key-value cache
        self.truncate_kv_cache(tokenized_messages["input_ids"])

        if (
            "past_key_values" in self.past_key_values_kwargs
            and self.past_key_values_kwargs["past_key_values"][0][0].shape[-2]
            > tokenized_messages["input_ids"].shape[-1]
        ):
            raise ValueError("Past key values are larger than the input_ids")

        past_key_values = self.past_key_values_kwargs.get("past_key_values", None)
        if past_key_values is not None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        generated_sequences = self.model.generate(
            **tokenized_messages,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=past_key_values,
            return_legacy_cache=True,
        )
        # Cache the past key values
        if self.use_hot_cache:
            self.past_key_values_kwargs["past_key_values"] = (
                generated_sequences.past_key_values
            )
        self.past_token_ids = generated_sequences.sequences

        # Decode the output
        text_responses = self.tokenizer.batch_decode(
            generated_sequences.sequences[:, prompt_tokens:],
            skip_special_tokens=False,
        )

        text_responses = [
            text_response.replace("<|eot_id|>", "") for text_response in text_responses
        ]

        _, total_tokens_used = generated_sequences.sequences.shape
        return text_responses, total_tokens_used


class OpenAIGenerator:
    def __init__(self, use_images=False, model_name="gpt-4o-mini", api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.use_images = use_images
        self.model_name = model_name

    def reset(self):
        pass

    def prepare_messages(
        self,
        history: History,
        max_messages_window: int,
    ) -> tuple[list[dict], list]:
        """
        Prepare the image messages for the model
        """
        message_window = history.dialogue_history[-max_messages_window:]
        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]
        # add the system prompt if the first message is not a system message
        if message_window[0]["role"] != "system":
            message_window = [history.system_prompt_dialogue] + message_window

        if self.use_images:
            img_idx = -1
            seen_images = 0
            # iterate through the messages in reverse order to assign images
            for i in range(len(message_window) - 1, -1, -1):
                new_content_list = []
                for content in message_window[i]["content"]:
                    if content["type"] == "text":
                        new_content_list.append(content)
                    elif content["type"] == "image":
                        base64_image = numpy_to_base64(history.images[img_idx])
                        img_idx -= 1
                        seen_images + 1
                        new_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                        new_content_list.append(new_content)
                    message_window[i]["content"] = new_content_list
            assert seen_images <= len(history.images), "Too many images"

        return message_window, []

    def generate_unconstrained(
        self,
        batch_messages: list[list[dict]],
        max_tokens=256,
        temperature=0.6,
        **kwargs,
    ) -> tuple[list[str], int]:
        contents = []
        tokens_used = 0
        for messages in batch_messages:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n", "\n\n"],
            )
            content = response.choices[0].message.content
            tokens_used += response.usage.total_tokens
            contents.append(content)
        return contents, tokens_used
