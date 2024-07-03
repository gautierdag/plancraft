import json
import logging
import random
import warnings

import hydra
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    Idefics2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

import wandb
from plancraft.config import TrainConfig

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class PlancraftDialogueDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/oracle",
        mm=False,
        split="train",
        max_window_size=30,
    ):
        super().__init__()
        self.split = split
        file_path = f"{dataset_dir}/{split}.jsonl"
        if mm:
            file_path = f"{dataset_dir}/{split}.mm.jsonl"

        print("Loading dialogue")
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))

        if mm:
            print("Loading images")
            # load images
            for example in data:
                example["images"] = []
                example["message_idx_to_image_idx"] = {}
                i = 0
                for message_idx, message in enumerate(example["messages"]):
                    for content in message["content"]:
                        if content["type"] == "image":
                            img_path = (
                                f"{dataset_dir}/{split}/{example['example_id']}_{i}.png"
                            )
                            img = Image.open(img_path).convert("RGB")
                            example["images"].append(img)
                            example["message_idx_to_image_idx"][message_idx] = i
                            i += 1

        self.dataset = data
        self.max_window_size = max_window_size

    def __len__(self) -> int:
        if self.split == "val":
            return int(len(self.dataset) * 0.05)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[dict, list]:
        example = self.dataset[idx]
        if len(example["messages"]) > self.max_window_size:
            # add system message
            messages = [example["messages"][0]]
            # sample window
            user_messages_idxs = list(
                range(self.max_window_size, len(example["messages"]), 2)
            )
            end = random.choice(user_messages_idxs)
            start = end - self.max_window_size + 1
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


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = TrainConfig(**dict(cfg))
    torch.set_float32_matmul_precision("medium")

    if cfg.training.base_model == "llama3":
        model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        train_dataset = PlancraftDialogueDataset(
            mm=False, max_window_size=cfg.training.max_window_length, split="train"
        )
        val_dataset = PlancraftDialogueDataset(
            mm=False, max_window_size=cfg.training.max_window_length, split="val"
        )
        collate_fn = get_collate_fn(
            tokenizer=tokenizer,
            only_assistant=True,
            template_name=cfg.training.base_model,
            max_length=cfg.training.max_seq_length,
        )
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
        ]
    elif cfg.training.base_model == "idefics2":
        model_name = "/nfs/public/hf/models/HuggingFaceM4/idefics2-8b-chatty"
        processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        train_dataset = PlancraftDialogueDataset(
            mm=True, max_window_size=cfg.training.max_window_length, split="train"
        )
        val_dataset = PlancraftDialogueDataset(
            mm=True, max_window_size=cfg.training.max_window_length, split="val"
        )
        collate_fn = get_collate_fn(
            processor=processor,
            only_assistant=True,
            template_name=cfg.training.base_model,
            pad_token_id=processor.tokenizer.pad_token_id,
            image_token_id=model.image_token_id,
            max_length=cfg.training.max_seq_length,
        )
        target_modules = ".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
    else:
        raise ValueError(f"Model {cfg.training.base_model} not supported")

    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=target_modules,
        use_dora=True,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config, adapter_name="default")
    model.print_trainable_parameters()

    name = f"hf-{cfg.training.base_model}-r{cfg.training.lora_r}-a{cfg.training.lora_alpha}"

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=cfg.model_dump(),
        name=name,
    )

    # ds_config = {
    #     "zero_optimization": {
    #         "stage": 2,
    #         "offload_optimizer": {
    #             "device": "cpu",
    #             "pin_memory": True,
    #         },
    #     },
    # }

    training_args = TrainingArguments(
        output_dir=f"outputs/{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        learning_rate=cfg.training.learning_rate,
        optim="adamw_hf",
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_num_workers=cfg.training.num_workers,
        logging_dir=f"outputs/logs/{name}",
        logging_steps=1,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=False,
        bf16=True if torch.cuda.is_bf16_supported() else False,  # bf16 support check
        report_to="wandb",
    )

    # Initialize the Huggingface Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    # save model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"outputs/{name}/final")


if __name__ == "__main__":
    main()
