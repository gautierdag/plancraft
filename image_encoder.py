import json
import glob
import torch
import wandb

import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import Dataset
from torch.optim import AdamW

wandb.require("core")


class PlancraftEnvironmentDataset(Dataset):
    def __init__(self, dataset_dir: str = "data/oracle", split="train"):
        super().__init__()
        self.split = split
        self.transform = transforms.ToTensor()
        data = []
        for example_path in sorted(glob.glob(f"{dataset_dir}/{split}/oa/*.json")):
            with open(example_path) as f:
                messages = json.load(f)
                environments = []
                for message in messages:
                    if "inventory=" in message["content"] and message["role"] == "user":
                        environments.append(
                            self.clean((message["content"].split("\ninventory=")[-1]))
                        )
                example = {
                    "environments": environments,
                    "example_id": example_path.split("/")[-1].split(".json")[0],
                }
                data.append(example)

        print("Loading images")
        for example in data:
            example["images"] = []
            for message_idx, _ in enumerate(example["environments"]):
                img_path = f"{dataset_dir}/{split}/imgs/{example['example_id']}_{message_idx}.png"
                img = Image.open(img_path).convert("RGB")
                example["images"].append(img)

        self.dataset = []
        for example in data:
            for i, (env, img) in enumerate(
                zip(example["environments"], example["images"])
            ):
                self.dataset.append((env, img))

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def clean(s: str):
        return (
            s.replace('"type": "', "")
            .replace('"quantity": ', "")
            .replace('"index": ', "")
            .replace('"', "")
            .replace("{", "")
            .replace("}", "")
            .replace(",", "")
            .replace("[", "")
            .replace("]", "")
            .replace("_", " ")
        )

    def __getitem__(self, idx: int) -> tuple:
        return self.dataset[idx]


if __name__ == "__main__":
    # Load the dataset
    dataset = PlancraftEnvironmentDataset(split="train")
    val_dataset = PlancraftEnvironmentDataset(split="val")

    def collate_fn(batch):
        img_tensors = []
        texts = []
        for text, img in batch:
            img_tensors.append(img)
            texts.append(text)
        return {
            "images": img_tensors,
            "texts": texts,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    # Load model and processor
    model = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        attn_implementation="sdpa",
        #   torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    wandb.init(
        project="plancraft-img-encoder",
        entity="itl",
        config={
            "model": "google/siglip-so400m-patch14-384",
            "optimizer": "AdamW",
            "lr": 1e-5,
            "batch_size": 32,
        },
    )
    model.train()
    for batch in tqdm(dataloader, total=len(dataset) // 32):
        optimizer.zero_grad()
        texts = batch["texts"]
        images = batch["images"]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        image_preds = torch.sigmoid(outputs.logits_per_image)
        text_preds = torch.sigmoid(outputs.logits_per_text)
        eye = torch.eye(len(texts), len(images), device=model.device)
        image_loss = F.binary_cross_entropy(image_preds, eye)
        text_loss = F.binary_cross_entropy(text_preds, eye)
        loss = image_loss + text_loss
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item()})
    torch.save(model.state_dict(), "image_encoder.pth")
