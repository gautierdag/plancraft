# import logging
import time

import torch
from loguru import logger
from openai import OpenAI
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    logger.warning("vLLM not installed. Please install vLLM to use vLLM")

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
        adapter_name="",
        hf_token=None,
        **kwargs,
    ):
        self.model_name = model_name
        # self.use_hot_cache = use_hot_cache
        self.hf_token = hf_token

        if tokenizer_name == "same":
            tokenizer_name = model_name

        self.use_images = use_images
        model_name, model_kwargs = self.build_model_kwargs(
            model_name, quantize=quantize
        )
        self.processor = None
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

        generated_sequences = self.model.generate(
            **tokenized_messages,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.pad_token_id,
            return_dict_in_generate=True,
        )

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


class VLLMGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_name="",
        **kwargs,
    ):
        self.model_name = model_name
        # Initialize vLLM model
        logger.info(f"Loading model {model_name} with vLLM")
        time_now = time.time()

        # Get downloaded models
        downloaded_models = get_downloaded_models()
        if model_name in downloaded_models:
            model_name = downloaded_models[model_name]
            logger.info(f"Using local model {model_name}")

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            enable_lora=True if adapter_name != "" else False,
        )

        # Load adapter
        self.lora_request = None
        if adapter_name != "":
            from huggingface_hub import snapshot_download

            logger.info(f"Loading adapter from {adapter_name}")
            lora_path = snapshot_download(repo_id=adapter_name)
            self.lora_request = LoRARequest(
                adapter_name,
                lora_int_id=0,
                lora_path=lora_path,
            )

        logger.info(f"Model loaded in {time.time() - time_now:.2f} seconds")

    def reset(self):
        # vLLM handles state automatically, no need to reset
        pass

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
        if len(message_window) > 0 and message_window[0]["role"] != "system":
            message_window = [history.system_prompt_dialogue] + message_window

        # vLLM doesn't use images
        return message_window, []

    @torch.inference_mode()
    def generate_unconstrained(
        self,
        batch_messages: list[list[dict]],
        max_tokens: int = 256,
        temperature=0.6,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n", "\n\n"],
        **kwargs,
    ) -> tuple[list[str], int]:
        """
        Generate unconstrained text based on the batch of messages using vLLM.
        """
        # Create sampling parameters for vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop if isinstance(stop, list) else [stop] if stop else None,
        )

        # Generate completions with vLLM
        outputs = self.llm.chat(
            batch_messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )

        # Extract responses
        text_responses = []
        total_tokens_used = 0

        for output in outputs:
            text_responses.append(output.outputs[0].text)
            # Sum prompt tokens and output tokens for the total
            total_tokens_used += len(output.prompt_token_ids) + len(
                output.outputs[0].token_ids
            )

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
