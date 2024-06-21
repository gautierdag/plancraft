import random
import json
from datasets import Dataset
from collections import defaultdict

from transformers import (
    AutoTokenizer,
    AutoTokenizer,
)


def sample_window(example, max_window_size=30):
    # add system message
    new_messages = [example["messages"][0]]
    num_steps = len(example["messages"]) - 1

    start = random.randint(1, num_steps)
    if start % 2 == 0:
        start = start + 1
    window_size = min(max_window_size, start)
    new_messages = new_messages + example["messages"][start - window_size + 1 : start]
    # print(f"window size: {window_size}, start: {start}, num_steps: {num_steps}")
    # new_messages = new_messages + example["messages"][start : start + window_size]
    return new_messages


def oversample_long_dialogue_dataset(
    examples: list[dict], max_window_size=30, num_oversampling=3
):
    window_train = []
    for example in examples:
        if len(example["messages"]) > max_window_size:
            for _ in range(num_oversampling):
                window_train.append(
                    {
                        "messages": sample_window(
                            example, max_window_size=max_window_size
                        )
                    }
                )
        else:
            window_train.append({"messages": example["messages"]})
    return window_train


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


def get_oa_training_data(
    tokenizer: AutoTokenizer,
    data_dir: str,
    seed=42,
    max_window_size=30,
    num_oversampling=3,
):
    print("Loading tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading data")
    data = defaultdict(list)
    for split in ["train", "val"]:
        with open(f"{data_dir}/oracle/{split}.jsonl", "r") as f:
            for line in f:
                data[split].append(json.loads(line))

    # oversample to account for window size
    train_dataset = Dataset.from_list(
        oversample_long_dialogue_dataset(
            data["train"],
            max_window_size=max_window_size,
            num_oversampling=num_oversampling,
        )
    )
    val_dataset = Dataset.from_list(
        oversample_long_dialogue_dataset(
            data["val"],
            max_window_size=max_window_size,
            num_oversampling=num_oversampling,
        )
    )

    # shuffle
    train_dataset = train_dataset.shuffle(seed=seed)
    val_dataset = val_dataset.shuffle(seed=seed)

    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=False,
        num_proc=6,
        remove_columns=["messages"],
        desc="Applying chat template to train dataset",
    )
    val_dataset = val_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=False,
        num_proc=6,
        remove_columns=["messages"],
        desc="Applying chat template to val dataset",
    )

    return train_dataset, val_dataset


import torch
from torch.utils.data import DataLoader

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
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            if image_token_id is not None:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels
        else:
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


dataset = PlancraftDialogueDataset(mm=True, max_window_size=30)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=get_collate_fn(
        processor=processor, only_assistant=True, template_name="idefics2"
    ),
)
batch = next(iter(loader))
