import json
import random
import logging
import warnings

import lightning as L
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
            mask_token_id = image_token_id
            batch = processor(
                text=messages_batch,
                images=images_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            labels[labels == pad_token_id] = mask_token_id
            batch["labels"] = labels
        else:
            mask_token_id = -100
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
            labels[mask == 0] = mask_token_id

        return batch

    return collate_fn


class PLWrapperModule(L.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.batch_size = config.get("batch_size", 1)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss.item(), on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        outputs = self.model(**batch)
        loss = outputs.loss
        # self.log("val_loss", loss)
        self.log("val_loss", loss.item())

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.get("lr", 0.0002)
        )
        return optimizer


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
    if cfg.training.base_model == "llama3":
        model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        train_dataset = PlancraftDialogueDataset(
            mm=False, max_window_size=cfg.plancraft.max_message_window, split="train"
        )
        val_dataset = PlancraftDialogueDataset(
            mm=False, max_window_size=cfg.plancraft.max_message_window, split="val"
        )
        collate_fn = get_collate_fn(
            tokenizer=tokenizer,
            only_assistant=True,
            template_name=cfg.training.base_model,
        )
    elif cfg.training.base_model == "idefics2":
        model_name = "/nfs/public/hf/models/HuggingFaceM4/idefics2-8b-chatty"
        processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
        if cfg.training.lora or cfg.training.qlora:
            if cfg.training.qlora:
                bnb_config = BitsAndBytesConfig(
                    # load_in_4bit=True,
                    device_map="cpu",
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=bnb_config if cfg.training.qlora else None,
        )
        if cfg.training.use_adapter:
            lora_config = LoraConfig(
                r=cfg.training.lora_r,
                lora_alpha=cfg.training.lora_alpha,
                lora_dropout=cfg.training.lora_dropout,
                target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
                use_dora=False if cfg.training.qlora else True,
                init_lora_weights="gaussian",
            )
            model.add_adapter(lora_config)
            model.enable_adapters()
        else:
            lora_config = LoraConfig(
                r=cfg.training.lora_r,
                lora_alpha=cfg.training.lora_alpha,
                lora_dropout=cfg.training.lora_dropout,
                target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
                use_dora=False if cfg.training.qlora else True,
                init_lora_weights="gaussian",
            )

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)

        train_dataset = PlancraftDialogueDataset(
            mm=True, max_window_size=30, split="train"
        )
        val_dataset = PlancraftDialogueDataset(mm=True, max_window_size=30, split="val")
        collate_fn = get_collate_fn(
            processor=processor,
            only_assistant=True,
            template_name=cfg.training.base_model,
            pad_token_id=processor.tokenizer.pad_token_id,
            image_token_id=model.image_token_id,
        )
    else:
        raise ValueError(f"Model {cfg.training.base_model} not supported")

    model_module = PLWrapperModule(config={}, model=model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=12,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=12,
    )

    name = f"{cfg.training.base_model}-lora-{cfg.training.lora}-qlora-{cfg.training.qlora}-adapter-{cfg.training.use_adapter}"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=cfg.model_dump(),
        name=name,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=3,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        log_every_n_steps=1,
        limit_val_batches=10,
    )
    trainer.fit(
        model_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    # save model
    model_module.model.save_pretrained("outputs/tmp_model")


if __name__ == "__main__":
    main()
