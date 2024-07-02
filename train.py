import json
import random
import logging
import warnings

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import torch
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Idefics2ForConditionalGeneration,
)

import hydra
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


class PLWrapperModule(L.LightningModule):
    def __init__(self, config: TrainConfig, model, total_steps: int):
        super().__init__()
        self.config = config
        self.model = model
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", outputs.loss.item(), on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        outputs = self.model(**batch)
        self.log("val_loss", outputs.loss.item())

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.training.learning_rate
        )

        # warmup with cosine decay
        num_warmup_steps = int(self.total_steps * self.config.training.warmup_ratio)

        def warm_decay(step):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            return num_warmup_steps**0.5 * step**-0.5

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warm_decay),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler_dict]


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
            device_map="cpu",
            torch_dtype=torch.bfloat16,
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
            device_map="cpu",
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config, adapter_name="default", mixed=True)
    model.print_trainable_parameters()

    dataset_length = len(train_dataset)
    total_steps = (
        dataset_length // cfg.training.batch_size
    ) * cfg.training.num_train_epochs

    model_module = PLWrapperModule(config=cfg, model=model, total_steps=total_steps)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    name = (
        f"{cfg.training.base_model}-r{cfg.training.lora_r}-a{cfg.training.lora_alpha}"
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=cfg.model_dump(),
        name=name,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="deepspeed_stage_2",
        max_epochs=cfg.training.num_train_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        precision="bf16-true",
        log_every_n_steps=1,
        limit_val_batches=20,
        callbacks=[
            ModelCheckpoint(dirpath=f"outputs/{name}"),
            EarlyStopping(monitor="val_loss", patience=1),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(
        model_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    # save model
    merged_model = model_module.model.merge_and_unload()
    merged_model.save_pretrained(f"outputs/{name}/final")


if __name__ == "__main__":
    main()
