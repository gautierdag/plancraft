import random
import json
from datasets import Dataset
from collections import 

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments


data = defaultdict(list)
for split in ["train", "val"]:
    with open(f"data/oracle/{split}.jsonl", "r") as f:
        for line in f:
            data[split].append(json.loads(line))

MAX_WINDOW_SIZE = 30
NUM_OVERSAMPLING = 3


def sample_window(example):
    # add system message
    new_messages = [example["messages"][0]]
    num_steps = len(example["messages"]) - 1

    start = random.randint(1, num_steps)
    if start % 2 == 0:
        start = start + 1
    window_size = min(MAX_WINDOW_SIZE, start)
    new_messages = new_messages + example["messages"][start - window_size + 1 : start]
    # print(f"window size: {window_size}, start: {start}, num_steps: {num_steps}")
    # new_messages = new_messages + example["messages"][start : start + window_size]
    return new_messages


def oversample_long_dialogue_dataset(examples: list[dict]):
    window_train = []
    for example in examples:
        if len(example["messages"]) > MAX_WINDOW_SIZE:
            for _ in range(NUM_OVERSAMPLING):
                window_train.append({"messages": sample_window(example)})
        else:
            window_train.append({"messages": example["messages"]})
    return window_train


train_dataset = Dataset.from_list(oversample_long_dialogue_dataset(data["train"]))
val_dataset = Dataset.from_list(oversample_long_dialogue_dataset(data["val"]))

# shuffle
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)


from transformers import AutoTokenizer

model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


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


# Device map
device_map = "auto"  # for PP and running with `python test_sft.py`
# Load the model
model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer.pad_token = tokenizer.eos_token

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    local_files_only=True,
    use_cache=False,
    device_map=device_map,
)

# PEFT config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 32  # 64
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=[
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ],
)
# Args
max_seq_length = 8142
output_dir = "./outputs/training"
per_device_train_batch_size = 1  # reduced batch size to avoid OOM
gradient_accumulation_steps = 8  # 2
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4  # 2e-4
max_grad_norm = 0.3
max_steps = -1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    do_eval=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    max_seq_length=max_seq_length,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,  # gradient checkpointing
    report_to="wandb",
    dataset_text_field="text",
    seed=42,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_total_limit=3,
    run_name="sft",
)

model.gradient_checkpointing_enable()


from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=sft_config,
)


train_result = trainer.train()