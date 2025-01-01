import glob
import json
import random

import torch
from loguru import logger
from torch.utils.data import Dataset

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
        dataset_dir: str = "oracle_trajectories",
        use_images=False,
        trace_mode="oa",
        split="train",
        max_message_window=30,
    ):
        super().__init__()
        self.split = split
        self.use_images = use_images
        self.trace_mode = trace_mode
        self.add_system_message = True

        assert trace_mode in ["oa", "ota"], f"Invalid trace mode {trace_mode}"

        logger.info("Loading dialogue dataset")
        data = []
        for example_path in sorted(
            glob.glob(f"{dataset_dir}/{split}/{trace_mode}/*.json")
        ):
            with open(example_path) as f:
                example = json.load(f)
            data.append(example)
        self.dataset = data
        self.max_message_window = max_message_window

    def __len__(self) -> int:
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

        else:
            messages = example["messages"]
        return messages


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
    tokenizer,
    max_length=8142,
    only_assistant=False,
    template_name: str = "llama3",
):
    assert template_name in TEMPLATES

    def collate_fn(batch):
        messages_batch = []
        for messages in batch:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            # remove BOS token since it will later be added again by the tokenizer
            text = text.replace("<|begin_of_text|>", "")
            messages_batch.append(text)

        batch = tokenizer(
            messages_batch,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        batch["labels"] = labels

        # add mask for assistant response
        if only_assistant:
            mask = track_assistant_response(
                batch, tokenizer, template_name=template_name
            )
            labels[mask == 0] = -100

        return batch

    return collate_fn


def get_dataset_and_collate(
    tokenizer,
    template_name: str = "llama3",
    max_length: int = 8142,
    max_message_window: int = 30,
    trace_mode="oa",
    only_assistant=False,
):
    if template_name == "llama3":
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
            only_assistant=only_assistant,
            template_name=template_name,
            max_length=max_length,
        )
    return train_dataset, val_dataset, collate_fn
