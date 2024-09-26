import glob
import json
import logging
import math
import random
import warnings

import hydra
import imageio.v2 as imageio
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AdamW,
    get_scheduler,
)

import wandb
from plancraft.config import TrainConfig
from plancraft.models.oam import PlancraftOAM, PlancraftOAMConfig


class PlancraftDialogueDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/oracle",
        use_images: bool = False,
        trace_mode="oa",
        split="train",
        max_message_window=30,
    ):
        super().__init__()
        self.split = split
        self.trace_mode = trace_mode
        self.use_images = use_images

        assert trace_mode in ["oa"], f"Invalid trace mode {trace_mode}"

        print("Loading dialogue dataset")
        data = []
        for example_path in sorted(
            glob.glob(f"{dataset_dir}/{split}/{trace_mode}/*.json")
        ):
            with open(example_path) as f:
                example = json.load(f)

            example["images"] = []
            if self.use_images:
                example_id = example["example_id"]
                image_file = f"{dataset_dir}/{split}/images/{example_id}.gif"
                example["images"] = [
                    Image.fromarray(img) for img in imageio.mimread(image_file)
                ]
                example["message_idx_to_image_idx"] = {}
                j = 0
                for i, m in enumerate(example["messages"]):
                    # observation
                    if m["role"] == "user":
                        example["messages"][i]["content"] = (
                            example["messages"][i]["content"].split("\ninventory:")[0]
                            + "\ninventory:<|inventory|>"
                        )
                        example["message_idx_to_image_idx"][i] = j
                        j += 1

            data.append(example)
        self.dataset = data
        self.max_message_window = max_message_window

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[dict, list]:
        example = self.dataset[idx]
        if len(example["messages"]) > self.max_message_window:
            # sample window
            user_messages_idxs = list(
                range(self.max_message_window, len(example["messages"]), 2)
            )
            end = random.choice(user_messages_idxs)
            start = end - self.max_message_window + 1
            assert start > 0
            # add system message
            messages = [example["messages"][0]] + example["messages"][start : end + 1]

            if self.use_images:
                user_message_indices = list(range(start, end, 2))
                images = [
                    example["images"][example["message_idx_to_image_idx"][i]]
                    for i in user_message_indices
                ]
            else:
                images = []
        else:
            messages = example["messages"]
            images = example["images"]
        return messages, images


def collate_fn(batch):
    messages = []
    images = []
    for m, i in batch:
        messages.append(m)
        images.append(i)
    return {"batch_messages": messages, "batch_images": images}


wandb.require("core")

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def flatten_cfg(cfg):
    # for some reason hydra wraps file paths from config path
    if len(cfg) == 1:
        return flatten_cfg(cfg[list(cfg.keys())[0]])
    return cfg


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = TrainConfig(**flatten_cfg(dict(cfg)))
    torch.set_float32_matmul_precision("medium")

    # Initialize the Accelerator
    accelerator = Accelerator()

    train_dataset = PlancraftDialogueDataset(use_images=True, split="train")
    val_dataset = PlancraftDialogueDataset(use_images=True, split="val")
    model = PlancraftOAM(config=PlancraftOAMConfig(from_llama=True))

    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
    ]

    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "embed_tokens",
            "lm_head",
            "vision_to_text_embedding",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    name = name = (
        f"oam-{cfg.training.base_model}-r{cfg.training.lora_r}-a{cfg.training.lora_alpha}"
    )

    # Initialize Weights & Biases (wandb) if needed
    if accelerator.is_main_process:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode="online",  # Change to "online" to enable logging
            config=cfg.model_dump(),
            name=name,
        )

    # Extract training arguments from cfg
    batch_size = cfg.training.batch_size
    num_train_epochs = cfg.training.num_train_epochs
    gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    max_grad_norm = cfg.training.max_grad_norm
    learning_rate = cfg.training.learning_rate
    num_workers = cfg.training.num_workers
    warmup_ratio = cfg.training.warmup_ratio

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Define the optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # Calculate total training steps
    total_train_samples = len(train_dataset)
    batch_size_per_update = (
        batch_size * accelerator.num_processes * gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(total_train_samples / batch_size_per_update)
    total_training_steps = num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_ratio * total_training_steps)

    # Define the scheduler
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Training Loop
    global_step = 0

    for epoch in range(num_train_epochs):
        model.train()
        total_loss = 0.0

        # Set epoch for the sampler (shuffling)
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # Normalize loss for gradient accumulation

                # Backward pass
                accelerator.backward(loss)
                total_loss += loss.item()

                # Optimize when gradient accumulation is reached
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(
                    train_loader
                ):
                    # Clip gradients if necessary
                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if accelerator.is_main_process:
                        wandb.log(
                            {"train_loss": total_loss * gradient_accumulation_steps},
                        )
                        accelerator.print(
                            f"Epoch {epoch} - Step {step} - Train loss: {total_loss * gradient_accumulation_steps}"
                        )
                    total_loss = 0.0

        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_steps = 0

        for batch in val_loader:
            with torch.no_grad():
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_steps += 1

        avg_eval_loss = eval_loss / eval_steps

        # Logging
        if accelerator.is_main_process:
            wandb.log({"eval_loss": avg_eval_loss})
            accelerator.print(f"Epoch {epoch} - Avg eval loss: {avg_eval_loss}")

    # Push to hub if required
    if cfg.training.push_to_hub:
        accelerator.print("saving model to hub")

        # This will call the unwrap model as well
        model = accelerator.unwrap_model(model)
        model = model.merge_and_unload()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            model.push_to_hub(name)

    # Finish logging
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
