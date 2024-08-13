import copy
import json
import logging
import os
import re
import time

import torch
from dotenv import load_dotenv
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


from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    SymbolicAction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
)
from plancraft.models.base import ABCModel, History
from plancraft.models.few_shot_images import load_prompt_images
from plancraft.models.react_prompts import (
    REACT_EXAMPLE,
    REACT_EXAMPLE_IMGS,
    REACT_SYSTEM_PROMPT,
)
from plancraft.models.utils import (
    Trie,
    get_downloaded_models,
    numpy_to_base64,
    tokenize,
)

logger = logging.getLogger(__name__)

load_dotenv()


class ValidActionsLogitsProcessor(torch.nn.Module):
    def __init__(self, choices: list[str], tokenizer: AutoTokenizer):
        super().__init__()
        self.choices = choices
        self.tree = Trie()
        self.start_idx = None
        self.eos = tokenizer.eos_token_id
        encoded_choices = tokenizer(choices, add_special_tokens=False)["input_ids"]
        for choice in encoded_choices:
            self.tree.insert(choice + [self.eos])

    def forward(self, input_ids, scores):
        if self.start_idx is None:
            # Calculate start_idx during the first forward pass
            self.start_idx = input_ids.shape[-1]

        decoded_so_far = input_ids[:, self.start_idx :]
        mask = torch.full_like(scores, float("-inf"))
        for batch_idx in range(input_ids.shape[0]):
            valid_next_tokens = self.tree.get_next(decoded_so_far[batch_idx].tolist())
            # if no choice then we allow the model to generate eos
            if len(valid_next_tokens) == 0:
                valid_next_tokens = [self.eos]

            mask[batch_idx, valid_next_tokens] = 0
        return scores + mask

    def reset(self):
        self.start_idx = None


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = "same",
        quantize=False,
        is_multimodal=False,
        use_hot_cache=True,
        adapter_name="",
        **kwargs,
    ):
        self.model_name = model_name
        self.use_hot_cache = use_hot_cache

        if tokenizer_name == "same":
            tokenizer_name = model_name

        self.is_multimodal = is_multimodal
        model_name, model_kwargs = self.build_model_kwargs(
            model_name, quantize=quantize
        )
        self.processor = None
        if "idefics" in model_name:
            assert is_multimodal, "Idefics model requires multimodal input"
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
                token=os.getenv("HF_TOKEN"),  # trust_remote_code=True
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

            if adapter_name != "":
                logger.info(f"Loading adapter {adapter_name}")
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

        self.fix_tokenizer_system_prompt(model_name, self.tokenizer)

        self.model.eval()
        if self.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.truncation_side = "left"

        self.action_logit_processor = ValidActionsLogitsProcessor(
            [" move", " smelt"], self.tokenizer
        )
        self.slot_from_processor = ValidActionsLogitsProcessor(
            [str(i) for i in range(46)], self.tokenizer
        )
        self.slot_to_processor = ValidActionsLogitsProcessor(
            [str(i) for i in range(1, 46)], self.tokenizer
        )
        self.quantity_processor = ValidActionsLogitsProcessor(
            [str(i) for i in range(1, 65)], self.tokenizer
        )

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
        if self.is_multimodal:
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

    @staticmethod
    def fix_tokenizer_system_prompt(model_name: str, tokenizer):
        """
        Returns True if the model supports a system role
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if "mistral" in model_name.lower():
            # get directory of current file
            chat_template = open(
                current_dir + "/templates/mistral-instruct.jinja"
            ).read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            # set the chat template
            tokenizer.chat_template = chat_template
        elif "gemma" in model_name.lower():
            # get directory of current file
            chat_template = open(current_dir + "/templates/gemma-instruct.jinja").read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            # set the chat template
            tokenizer.chat_template = chat_template
        elif "phi" in model_name.lower():
            chat_template = open(current_dir + "/templates/phi-instruct.jinja").read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
            tokenizer.chat_template = chat_template

    @staticmethod
    def build_model_kwargs(model_name: str, **kwargs) -> tuple[str, dict]:
        model_kwargs = {
            "token": os.getenv("HF_TOKEN"),
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
        system_prompt: dict = None,
        prompt_images: list = [],
    ) -> tuple[list[dict], list]:
        """
        Prepare the messages using a history
        """
        message_window = history.dialogue_history[-max_messages_window:]
        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]
        # add the system prompt if the first message is not a system message
        if message_window[0]["role"] != "system" and system_prompt is not None:
            message_window = [system_prompt] + message_window

        image_window = []
        if self.is_multimodal:
            image_list = prompt_images + history.images
            image_count = 0
            # iterate through the messages in reverse order to assign images
            for m in message_window:
                for content in m["content"]:
                    if content["type"] == "image":
                        image_count += 1
            assert image_count <= len(image_list), "Too many images"
            image_window = image_list[-image_count:]
            image_window = [Image.fromarray(img) for img in image_window]

        return message_window, image_window

    @torch.inference_mode()
    def generate_thoughts(
        self,
        batch_messages: list[list[dict]],
        temperature=1.0,
        max_tokens=256,
        **kwargs,
    ) -> tuple[list[str], int]:
        if self.is_multimodal:
            assert "images" in kwargs, "Images required for multimodal model"

        tokenized_messages = tokenize(
            self.model,
            self.tokenizer,
            batch_messages,
            start_messages_generation=["thought:"] * len(batch_messages),
            max_tokens=max_tokens,
            images=kwargs.get("images") if self.is_multimodal else None,
        )
        prompt_tokens = tokenized_messages["input_ids"].shape[-1]

        # sent to same device as model
        tokenized_messages = {
            k: v.to(self.model.device) for k, v in tokenized_messages.items()
        }

        # truncate the key-value cache
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
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=past_key_values,
            return_legacy_cache=True,
        )
        # cache the past key values
        if self.use_hot_cache:
            self.past_key_values_kwargs["past_key_values"] = (
                generated_sequences.past_key_values
            )
        self.past_token_ids = generated_sequences.sequences

        # decode the output
        text_responses = self.tokenizer.batch_decode(
            generated_sequences.sequences[:, prompt_tokens:],
            skip_special_tokens=True,
        )
        text_responses = [
            f"thought:{text_response}" for text_response in text_responses
        ]
        _, total_tokens_used = generated_sequences.sequences.shape
        return text_responses, total_tokens_used

    def generate_with_processor(
        self,
        logits_processor: ValidActionsLogitsProcessor,
        start_messages_generation: list[str],
        batch_messages: list[list[dict]],
        temperature=1.0,
        **kwargs,
    ) -> tuple[list[str], int]:
        if self.is_multimodal:
            assert "images" in kwargs, "Images required for multimodal model"

        tokenized_messages = tokenize(
            self.model,
            self.tokenizer,
            batch_messages,
            start_messages_generation=start_messages_generation,
            images=kwargs.get("images") if self.is_multimodal else None,
        )
        # sent to same device as model
        tokenized_messages = {
            k: v.to(self.model.device) for k, v in tokenized_messages.items()
        }

        # truncate the key-value cache
        self.truncate_kv_cache(tokenized_messages["input_ids"])

        # number of tokens in the prompt
        prompt_tokens = tokenized_messages["input_ids"].shape[-1]

        if (
            "past_key_values" in self.past_key_values_kwargs
            and self.past_key_values_kwargs["past_key_values"][0][0].shape[-2]
            > tokenized_messages["input_ids"].shape[-1]
        ):
            raise ValueError("Past key values are larger than the input_ids")

        past_key_values = self.past_key_values_kwargs.get("past_key_values", None)
        if past_key_values is not None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # Generate the initial action constrained to valid action tokens
        generated_sequences = self.model.generate(
            **tokenized_messages,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=logits_processor.tree.longest_sequence_length,
            pad_token_id=self.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            logits_processor=[logits_processor],
            past_key_values=past_key_values,
            return_legacy_cache=True,
        )
        # cache the past key values
        if self.use_hot_cache:
            self.past_key_values_kwargs["past_key_values"] = (
                generated_sequences.past_key_values
            )
        self.past_token_ids = generated_sequences.sequences

        # reset the start index
        logits_processor.reset()

        # select only new tokens and decode the generated choices
        generated_choices = self.tokenizer.batch_decode(
            generated_sequences["sequences"][:, prompt_tokens:],
            skip_special_tokens=True,
        )
        return generated_choices, generated_sequences["sequences"].shape[-1]

    @torch.inference_mode()
    def generate_actions(
        self,
        batch_messages: list[list[dict]],
        temperature=1.0,
        **kwargs,
    ) -> tuple[list[SymbolicAction], list[str], int]:
        """
        Select whether to smelt or move
        Then select the slots and quantity
        output should be constrained to something like:
            `act: move from slot 39 to slot 5 with quantity 1`
        """
        overall_messages = ["act:"] * len(batch_messages)
        actions_selected, _ = self.generate_with_processor(
            self.action_logit_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
            **kwargs,
        )
        overall_messages = [
            f"{overall}{action} from slot "
            for (overall, action) in zip(overall_messages, actions_selected)
        ]
        # select the slot from
        slots_from_selected, _ = self.generate_with_processor(
            self.slot_from_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
            **kwargs,
        )
        overall_messages = [
            f"{overall}{slot_from} to slot "
            for (overall, slot_from) in zip(overall_messages, slots_from_selected)
        ]
        # select the slot to
        slots_to_selected, _ = self.generate_with_processor(
            self.slot_to_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
            **kwargs,
        )
        overall_messages = [
            f"{overall}{slot_to} with quantity "
            for (overall, slot_to) in zip(overall_messages, slots_to_selected)
        ]
        # select the quantity
        quantities_selected, num_tokens = self.generate_with_processor(
            self.quantity_processor,
            batch_messages=batch_messages,
            start_messages_generation=overall_messages,
            temperature=temperature,
            **kwargs,
        )
        overall_messages = [
            f"{overall}{quantity}"
            for (overall, quantity) in zip(overall_messages, quantities_selected)
        ]

        # parse the actions
        actions = []
        for action, slot_from, slot_to, quantity in zip(
            actions_selected,
            slots_from_selected,
            slots_to_selected,
            quantities_selected,
        ):
            if action == "smelt":
                act = SymbolicSmeltAction(
                    slot_from=int(slot_from),
                    slot_to=int(slot_to),
                    quantity=int(quantity),
                )
            else:
                act = SymbolicMoveAction(
                    slot_from=int(slot_from),
                    slot_to=int(slot_to),
                    quantity=int(quantity),
                )
            actions.append(act)
        return actions, overall_messages, num_tokens


class OpenAIGenerator:
    def __init__(self, is_multimodal=False, model_name="gpt-4o-mini"):
        self.client = OpenAI()
        self.is_multimodal = is_multimodal
        self.model_name = model_name

    def reset(self):
        pass

    def prepare_messages(
        self,
        history: History,
        max_messages_window: int,
        system_prompt: dict = None,
        prompt_images: list = [],
    ) -> tuple[list[dict], list]:
        """
        Prepare the image messages for the model
        """
        message_window = history.dialogue_history[-max_messages_window:]
        # remove the first assistant message if it is present
        if len(message_window) > 0 and message_window[0]["role"] == "assistant":
            message_window = message_window[1:]
        # add the system prompt if the first message is not a system message
        if message_window[0]["role"] != "system" and system_prompt is not None:
            message_window = [system_prompt] + message_window

        if self.is_multimodal:
            image_list = prompt_images + history.images

            img_idx = -1
            seen_images = 0
            # iterate through the messages in reverse order to assign images
            for i in range(len(message_window) - 1, -1, -1):
                new_content_list = []
                for content in message_window[i]["content"]:
                    if content["type"] == "text":
                        new_content_list.append(content)
                    elif content["type"] == "image":
                        base64_image = numpy_to_base64(image_list[img_idx])
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
            assert seen_images <= len(image_list), "Too many images"

        return message_window, []

    def generate_thoughts(
        self,
        batch_messages: list[list[dict]],
        max_tokens=256,
        temperature=1.0,
        **kwargs,
    ) -> tuple[list[str], int]:
        thoughts = []
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
                stop=["\n", "act:"],
            )
            content = response.choices[0].message.content
            tokens_used += response.usage.total_tokens
            thoughts.append(content)
        return thoughts, tokens_used

    def generate_actions(
        self,
        batch_messages: list[list[dict]],
        temperature=1.0,
        **kwargs,
    ) -> tuple[list[SymbolicAction], list[str], int]:
        """
        Select whether to smelt or move
        Then select the slots and quantity
        output should be constrained to something like:
            `act: move from slot 39 to slot 5 with quantity 1`
        """
        actions = []
        action_messages = []
        tokens_used = 0
        for messages in batch_messages:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=32,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n", "TASK:", "."],
            )
            content = response.choices[0].message.content
            tokens_used += response.usage.total_tokens

            # try to parse the actions using regex
            action = None
            slot_to = None
            slot_from = None
            quantity = None
            try:
                action = re.search(r"act: (move|smelt)", content).group(1)
                slot_from = re.search(r"from slot (\d+)", content).group(1)
                slot_to = re.search(r"to slot (\d+)", content).group(1)
                quantity = re.search(r"with quantity (\d+)", content).group(1)
            except AttributeError:
                logger.warning(f"Failed to parse action: {content}")
                action = "move"
                slot_from = 0
                slot_to = 0
                quantity = 1

            try:
                if action == "smelt":
                    act = SymbolicSmeltAction(
                        slot_from=int(slot_from),
                        slot_to=int(slot_to),
                        quantity=int(quantity),
                    )
                else:
                    act = SymbolicMoveAction(
                        slot_from=int(slot_from),
                        slot_to=int(slot_to),
                        quantity=int(quantity),
                    )
            except Exception:
                logger.warning(f"Failed to validate action: {content}")
                act = SymbolicMoveAction(
                    slot_from=0,
                    slot_to=0,
                    quantity=1,
                )

            actions.append(act)
            action_messages.append(content)

        return actions, action_messages, tokens_used


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
                "content": [
                    {"text": copy.deepcopy(REACT_SYSTEM_PROMPT), "type": "text"}
                ],
            }
        else:
            examples = copy.deepcopy(REACT_EXAMPLE)
            self.system_prompt = {
                "role": "system",
                "content": copy.deepcopy(REACT_SYSTEM_PROMPT),
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

        thought_messages_windows = []
        thought_images_windows = []
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
            thought_messages_windows.append(message_window)
            thought_images_windows.append(image_window)

        # generate thoughts
        thought_messages, thinking_token_used = self.llm.generate_thoughts(
            batch_messages=thought_messages_windows,
            max_tokens=256,
            images=thought_images_windows,
        )

        action_messages_windows = []
        action_images_windows = []
        # update message window with thoughts and collect action messages
        for thought_message, history_idx in zip(thought_messages, real_obs_idx):
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
