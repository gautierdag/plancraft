import glob
import json
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from plancraft.models.react_prompts import REACT_SYSTEM_PROMPT, ACT_SYSTEM_PROMPT

TEMPLATES = {
    "idefics2": {
        "assistant": "\nAssistant:",
        "user": "\nUser:",
    },
    "llama3": {
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n",
    },
}


class PlancraftDialogueDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/oracle",
        use_images=False,
        trace_mode="oa",
        split="train",
        max_message_window=30,
        add_system_message=True,
    ):
        super().__init__()
        self.split = split
        self.use_images = use_images
        self.trace_mode = trace_mode
        self.add_system_message = add_system_message

        assert trace_mode in ["oa", "ota"], f"Invalid trace mode {trace_mode}"

        system_message = ""
        if self.trace_mode == "oa":
            system_message = ACT_SYSTEM_PROMPT
        elif self.trace_mode == "ota":
            system_message = REACT_SYSTEM_PROMPT
        else:
            raise ValueError(f"Invalid trace mode {trace_mode}")

        print("Loading dialogue dataset")
        data = []
        for example_path in glob.glob(f"{dataset_dir}/{split}/{trace_mode}/*.json"):
            with open(example_path) as f:
                messages = json.load(f)

                if add_system_message:
                    messages = [
                        {"role": "system", "content": system_message}
                    ] + messages

                # convert to use list of content items instead of a single string
                if use_images:
                    content_messages = []
                    for message in messages:
                        # NOTE: remove the text inventory description
                        new_message = {
                            "role": message["role"],
                            "content": [
                                {
                                    "text": message["content"].split("\ninventory=[{")[
                                        0
                                    ],
                                    "type": "text",
                                }
                            ],
                        }
                        if message["role"] == "user" and message["content"] != "Ok":
                            new_message["content"] = [{"type": "image"}] + new_message[
                                "content"
                            ]
                        content_messages.append(new_message)
                    messages = content_messages

                example = {
                    "messages": messages,
                    "example_id": example_path.split("/")[-1].split(".json")[0],
                }
                data.append(example)

        if use_images:
            print("Loading images")
            # load images
            for example in data:
                example["images"] = []
                example["message_idx_to_image_idx"] = {}
                i = 0
                for message_idx, message in enumerate(example["messages"]):
                    for content in message["content"]:
                        if content["type"] == "image":
                            img_path = f"{dataset_dir}/{split}/imgs/{example['example_id']}_{i}.png"
                            img = Image.open(img_path).convert("RGB")
                            example["images"].append(img)
                            example["message_idx_to_image_idx"][message_idx] = i
                            i += 1
        self.dataset = data
        self.max_message_window = max_message_window

    def __len__(self) -> int:
        # if self.split == "val":
        # return int(len(self.dataset) * 0.5)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[dict, list]:
        example = self.dataset[idx]
        if len(example["messages"]) > self.max_message_window:
            # add system message
            if self.add_system_message:
                messages = [example["messages"][0]]
            else:
                messages = []

            # sample window
            user_messages_idxs = list(
                range(self.max_message_window, len(example["messages"]), 2)
            )
            end = random.choice(user_messages_idxs)
            start = end - self.max_message_window + 1
            assert start != 0
            # add window
            messages = messages + example["messages"][start : end + 1]
            images = []
            if "images" in example:
                for message_idx in range(start, end):
                    if message_idx in example["message_idx_to_image_idx"]:
                        image_idx = example["message_idx_to_image_idx"][message_idx]
                        images.append(example["images"][image_idx])
        else:
            messages = example["messages"]
            images = example.get("images", [])
        return messages, images


def track_assistant_response(
    batch,
    tokenizer,
    template_name: str = "llama3",
):
    """
    Mask that returns 1 for tokens in the assistant response and 0 otherwise.
    """
    assistant_template = TEMPLATES[template_name]["assistant"]
    user_template = TEMPLATES[template_name]["user"]
    start_seq = tokenizer.encode(
        assistant_template,
        add_special_tokens=False,
        return_tensors="pt",
    )[0]
    end_seq = tokenizer.encode(
        user_template,
        add_special_tokens=False,
        return_tensors="pt",
    )[0]
    encoded_label_ids = batch["labels"]
    mask = torch.zeros_like(encoded_label_ids)
    for seq_idx, seq in enumerate(encoded_label_ids):
        in_masked_response = False
        i = 0
        while i < len(seq):
            if i + len(start_seq) < len(seq) and torch.all(
                seq[i : i + len(start_seq)].eq(start_seq)
            ):
                in_masked_response = True
                i += len(start_seq)
                continue
            if i + len(end_seq) < len(seq) and torch.all(
                seq[i : i + len(end_seq)].eq(end_seq)
            ):
                in_masked_response = False
                i += len(end_seq)
                continue
            if in_masked_response:
                mask[seq_idx, i] = 1
            else:
                mask[seq_idx, i] = 0
            i += 1
    return mask


def get_collate_fn(
    tokenizer=None,
    processor=None,
    max_length=8142,
    only_assistant=False,
    template_name: str = "llama3",
    pad_token_id=None,
    image_token_id=None,
):
    assert tokenizer or processor and not (tokenizer and processor)
    # if processor then must have image_token_id
    assert not processor or image_token_id

    assert template_name in TEMPLATES

    def collate_fn(batch):
        messages_batch = []
        images_batch = []
        for messages, images in batch:
            if processor:
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                )
                images_batch.append(images)
            else:
                text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                )
            messages_batch.append(text)
        if processor:
            batch = processor(
                text=messages_batch,
                images=images_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            labels[labels == pad_token_id] = -100
            batch["labels"] = labels
        else:
            batch = tokenizer(
                messages_batch,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            # NOTE: off by one error?
            batch["labels"] = labels

        # add mask for assistant response
        if only_assistant:
            if processor:
                mask = track_assistant_response(
                    batch, processor.tokenizer, template_name=template_name
                )
            else:
                mask = track_assistant_response(
                    batch, tokenizer, template_name=template_name
                )
            labels[mask == 0] = -100

        return batch

    return collate_fn


def get_dataset_and_collate(
    template_name: str, max_length: int, max_message_window: int, trace_mode="oa"
):
    if template_name == "llama3":
        model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = PlancraftDialogueDataset(
            use_images=False,
            max_message_window=max_message_window,
            split="train",
            trace_mode=trace_mode,
        )
        val_dataset = PlancraftDialogueDataset(
            use_images=False,
            max_message_window=max_message_window,
            split="val",
            trace_mode=trace_mode,
        )
        collate_fn = get_collate_fn(
            tokenizer=tokenizer,
            only_assistant=True,
            template_name=template_name,
            max_length=max_length,
        )
    elif template_name == "idefics2":
        model_name = "/nfs/public/hf/models/HuggingFaceM4/idefics2-8b-chatty"
        processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
        train_dataset = PlancraftDialogueDataset(
            use_images=True,
            max_message_window=max_message_window,
            split="train",
            trace_mode=trace_mode,
        )
        val_dataset = PlancraftDialogueDataset(
            use_images=True,
            max_message_window=max_message_window,
            split="val",
            trace_mode=trace_mode,
        )
        collate_fn = get_collate_fn(
            processor=processor,
            only_assistant=True,
            template_name=template_name,
            pad_token_id=processor.tokenizer.pad_token_id,
            image_token_id=32001,
            max_length=max_length,
        )
    return train_dataset, val_dataset, collate_fn
