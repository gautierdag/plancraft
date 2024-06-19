import random
import json
from datasets import Dataset
from collections import defaultdict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

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


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


if __name__ == "__main__":

    print("Loading tokenizer")
    model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading data")
    data = defaultdict(list)
    for split in ["train", "val"]:
        with open(f"data/oracle/{split}.jsonl", "r") as f:
            for line in f:
                data[split].append(json.loads(line))

    train_dataset = Dataset.from_list(oversample_long_dialogue_dataset(data["train"]))
    val_dataset = Dataset.from_list(oversample_long_dialogue_dataset(data["val"]))

    # shuffle
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)

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
 
    # Load the model
    print("Loading model") 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
        use_cache=False,
        device_map="auto",
    )

    # PEFT config
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
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
    training_name = f"lora_a{lora_alpha}_r{lora_r}"
    # Args
    sft_config = SFTConfig(
        output_dir=f"./outputs/training/{training_name}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=50,
        logging_steps=1,
        learning_rate=5e-5,
        do_eval=True,
        max_grad_norm=0.3,
        max_steps=-1,
        max_seq_length=8142,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,  # gradient checkpointing
        report_to="wandb",
        dataset_text_field="text",
        seed=42,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        save_total_limit=3,
        run_name=training_name,
    )
    model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_config,
    )


    train_result = trainer.train()