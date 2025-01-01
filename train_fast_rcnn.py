import random

import torch
import torchvision.transforms.v2 as v2
from PIL import Image, ImageDraw
from tqdm import tqdm

import wandb
from plancraft.environment.env import PlancraftEnvironment
from plancraft.environment.items import ALL_ITEMS
from plancraft.environment.recipes import RECIPES, ShapedRecipe, ShapelessRecipe
from plancraft.environment.sampler import sample_distractors
from plancraft.models.bbox_model import IntegratedBoundingBoxModel, slot_to_bbox

CRAFTING_RECIPES = [
    r
    for recipes in RECIPES.values()
    for r in recipes
    if isinstance(r, ShapelessRecipe) or isinstance(r, ShapedRecipe)
]


def sample_random_recipe_crafting_table() -> list[dict[str, int]]:
    return random.choice(CRAFTING_RECIPES).sample_input_crafting_grid()


def sample_starting_inv():
    inventory = []
    is_crafting = random.choice([True, False])
    start_slot_idx = 1
    max_num_items = 45
    if is_crafting:
        start_slot_idx = 11
        max_num_items = 34
        inventory = sample_random_recipe_crafting_table()

    # random number of items
    selected_slots = random.sample(
        range(start_slot_idx, 46), random.randint(1, max_num_items)
    )
    for slot in selected_slots:
        item = sample_distractors(num_distractors=1)
        item_name = list(item.keys())[0]
        item_count = item[item_name]
        inventory.append(
            {
                "type": item_name,
                "slot": slot,
                "quantity": item_count,
            }
        )
    # sort by slot
    inventory = sorted(inventory, key=lambda x: x["slot"])
    return inventory


class EnvWrapper:
    def __init__(self):
        self.env = PlancraftEnvironment(
            inventory=sample_starting_inv(),
        )

    def step(self, starting_inv: list[dict[str, int]], resolution: str):
        self.env.reset(new_inventory=starting_inv)
        self.env.table.resolution = resolution
        obs = self.env.step()
        return obs


def sample_environment(batch_size=32, N=100):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    env = EnvWrapper()
    print("Env loaded")
    i = 0
    while i < N:
        batch_images = []
        batch_targets = []
        batch_images_raw = []
        batch_inventory = []
        # resolution = random.choice(["low", "medium", "high"])
        resolution = "high"
        while len(batch_images) < batch_size:
            starting_inv = sample_starting_inv()
            obs = env.step(starting_inv, resolution=resolution)
            # create targets/boxes
            target = {"labels": [], "boxes": [], "quantity_labels": []}
            inv = []
            for slot, item in obs["inventory"].items():
                if item["quantity"] > 0:
                    target["labels"].append(ALL_ITEMS.index(item["type"]))
                    target["boxes"].append(slot_to_bbox(slot, resolution))
                    target["quantity_labels"].append(item["quantity"])

                    inv.append(item)

            # convert to tensors
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
            target["quantity_labels"] = torch.tensor(
                target["quantity_labels"], dtype=torch.int64
            )
            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.int64)

            if len(target["labels"]) == 0:
                continue

            img_tensor = transform(obs["image"].copy())
            batch_images.append(img_tensor)
            batch_targets.append(target)
            batch_images_raw.append(obs["image"])
            batch_inventory.append(inv)

        yield (
            torch.stack(batch_images),
            batch_targets,
            batch_images_raw,
            batch_inventory,
        )
        i += 1


if __name__ == "__main__":
    m1_path = "latest_fasterrcnn_high.pth"
    M1_model = IntegratedBoundingBoxModel(load_resnet_weights=True)
    M1_model = M1_model.cuda()

    print("Loaded model")
    N = 20000
    m1_lr = 0.0005
    batch_size = 8
    save_every = 500
    count = 0
    m1_optimizer = torch.optim.AdamW(M1_model.parameters(), lr=m1_lr)

    wandb.init(project="plancraft-img-encoder", entity="itl", name="all-res")
    pbar = tqdm(total=N)

    for images, targets, raw_images, inv in sample_environment(
        N=N,
        batch_size=batch_size,
    ):
        M1_model.train()
        images = images.cuda()
        for i in range(len(targets)):
            targets[i]["boxes"] = targets[i]["boxes"].cuda()
            targets[i]["labels"] = targets[i]["labels"].cuda()
            targets[i]["quantity_labels"] = targets[i]["quantity_labels"].cuda()

        m1_loss_dict = M1_model(images, targets)
        m1_losses = sum(loss for loss in m1_loss_dict.values())
        wandb.log(m1_loss_dict)
        wandb.log({"m1_train_loss": m1_losses})
        pbar.update(1)
        pbar.set_description(f"Loss: {m1_losses.item()}")

        m1_optimizer.zero_grad()
        m1_losses.backward()
        m1_optimizer.step()

        if count % save_every == 0:
            # save model
            M1_model.eval()
            M1_model.save(m1_path)
            with torch.no_grad():
                predictions = M1_model(images)

            for img_idx in range(len(images)):
                # generate image and target for validation
                img = Image.fromarray(raw_images[img_idx].copy())
                for box in targets[img_idx]["boxes"]:
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(box.cpu().tolist(), outline="red")

                for box_idx in range(len(predictions[img_idx]["boxes"])):
                    box = predictions[img_idx]["boxes"][box_idx]
                    score = predictions[img_idx]["scores"][box_idx]
                    label = predictions[img_idx]["labels"][box_idx]
                    quantity = predictions[img_idx]["quantities"][box_idx]

                    if score > 0:
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(box.cpu().tolist(), outline="green")
                        draw.text(
                            (box[0], box[1] + 10),
                            f"{quantity.item()}",
                            fill="blue",
                        )
                # log image
                wandb.log({f"image_{img_idx}": wandb.Image(img)})
                break

        count += 1
        if count >= N:
            break

    pbar.close()
    wandb.finish()

    # save model
    M1_model.push_to_hub("plancraft-fasterrcnn-high")
