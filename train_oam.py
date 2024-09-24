import glob
import json
import logging
import random
import warnings

import hydra
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

import wandb
from plancraft.config import TrainConfig
from plancraft.train.train_fast_rcnn import IntegratedBoundingBoxModel


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


class PlancraftAOMConfig(PretrainedConfig):
    model_type = "plancraft-aom"
    is_composition = True

    def __init__(
        self,
        use_cache=True,
        **kwargs,
    ):
        self.use_cache = use_cache
        super().__init__(**kwargs)


class PlancraftAOM(PreTrainedModel):
    config_class = PlancraftAOMConfig

    def __init__(self, config: PlancraftAOMConfig):
        super().__init__(config)

        self.config = config
        # load text model
        self.text_model = AutoModelForCausalLM.from_pretrained(
            "/nfs/public/hf/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            device_map="auto",
        )
        # load vision model
        self.vision_model = IntegratedBoundingBoxModel.from_pretrained(
            "gautierdag/plancraft-maskrcnn"
        )
        self.vision_model.eval()
        self.vision_model = self.vision_model.cuda()

        # convert vision features to text embedding
        self.vision_to_text_embedding = nn.Linear(
            1024, self.text_model.config.hidden_size
        )
        self.vision_to_text_embedding = self.vision_to_text_embedding.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "/nfs/public/hf/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            trust_remote=True,
        )
        # add special tokens
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|inventory|>",
                ]
            }
        )
        self.inventory_idx = self.tokenizer.convert_tokens_to_ids("<|inventory|>")
        self.text_model.resize_token_embeddings(len(self.tokenizer))

        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    @torch.no_grad()
    def extract_bboxes(self, images: list) -> list[dict]:
        img_tensors = torch.stack([self.transforms(img) for img in images])
        img_tensors = img_tensors.cuda()
        # disable gradients
        self.vision_model.freeze()
        # get bounding box predictions
        bbox_preds = self.vision_model(img_tensors)
        return bbox_preds

    def prepare_messages(self, messages: list, bboxes: list[dict]) -> str:
        # merge inventory tokens into text inputs
        new_messages = []
        i_pred = 0
        for m in messages:
            new_message = m.copy()
            if new_message["role"] == "user" and new_message["content"].endswith(
                "<|inventory|>"
            ):
                # add inventory tokens for each bounding box
                new_message["content"] = new_message["content"].replace(
                    "<|inventory|>",
                    "<|inventory|>" * (bboxes[i_pred]["features"].shape[0]),
                )
                i_pred += 1
            new_messages.append(new_message)
        assert i_pred == len(bboxes)
        # add special tokens
        text = self.tokenizer.apply_chat_template(
            new_messages, add_generation_prompt=False, tokenize=False
        )
        text = text.replace("<|begin_of_text|>", "")
        return text

    def forward(
        self, batch_images: list[list], batch_messages: list[list[dict]], **kwargs
    ):
        texts_batch = []
        features_batch = []
        total_boxes = 0
        for images, messages in zip(batch_images, batch_messages):
            # process images
            bboxes = self.extract_bboxes(images)
            # get bbox features
            features = torch.concat([p["features"] for p in bboxes], dim=0)
            # upscale to text embedding size
            features_embeds = self.vision_to_text_embedding(features)
            features_batch.append(features_embeds)
            # count bboxes total
            total_boxes += features.shape[0]
            # process messages
            text = self.prepare_messages(messages, bboxes)
            texts_batch.append(text)

        # tokenize text
        batch = self.tokenizer(
            texts_batch,
            truncation=True,
            max_length=16000,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        # remove inventory tokens from labels
        labels[labels == self.inventory_idx] = -100
        # sanity check: should have same number of boxes as inventory tokens
        assert (labels == -100).sum() == total_boxes

        # get text embeddings
        inputs_embeds = self.text_model.get_input_embeddings()(batch["input_ids"])

        # along batch dimension
        for i in range(len(features_batch)):
            # replace inventory tokens with bbox features
            inputs_embeds[i, batch["input_ids"][i] == self.inventory_idx] = (
                features_batch[i]
            )

        # forward pass
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
        )
        return outputs


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

    train_dataset = PlancraftDialogueDataset(use_images=True, split="train")
    val_dataset = PlancraftDialogueDataset(use_images=True, split="val")
    model = PlancraftAOM(config=PlancraftAOMConfig())

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

    name = "test-plancraft-aom"

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        # mode=cfg.wandb.mode,
        mode="disabled",
        config=cfg.model_dump(),
        name=name,
    )

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
        dataloader_pin_memory=True,
        logging_dir=f"outputs/logs/{name}",
        logging_steps=1,
        save_total_limit=1,
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

    if cfg.training.push_to_hub:
        model.push_to_hub(name)


if __name__ == "__main__":
    main()
