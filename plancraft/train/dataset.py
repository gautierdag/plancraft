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


def oversample_long_dialogue_dataset(examples: list[dict], max_window_size=30, num_oversampling=3):
    window_train = []
    for example in examples:
        if len(example["messages"]) > max_window_size:
            for _ in range(num_oversampling):
                window_train.append({"messages": sample_window(example, max_window_size=max_window_size)})
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


def get_oa_training_data(tokenizer:AutoTokenizer, data_dir:str, seed=42, max_window_size=30, num_oversampling=3):
    print("Loading tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading data")
    data = defaultdict(list)
    for split in ["train", "val"]:
        with open(f"{data_dir}/oracle/{split}.jsonl", "r") as f:
            for line in f:
                data[split].append(json.loads(line))

    # oversample to account for window size
    train_dataset = Dataset.from_list(oversample_long_dialogue_dataset(data["train"], max_window_size=max_window_size, num_oversampling=num_oversampling))
    val_dataset = Dataset.from_list(oversample_long_dialogue_dataset(data["val"], max_window_size=max_window_size, num_oversampling=num_oversampling))

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