import json
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import wandb

from plancraft.environments.env_real import RealPlancraft

wandb.require("core")


def slot_to_bbox(slot: int):
    # crafting slot
    if slot == 0:
        # slot size: 25x25
        # top left corner: (x= 118, y=30)
        box_size = 25
        left_x = 117
        top_y = 29
    # crafting grid
    elif slot < 10:
        # slot size: 18x18
        # top left corner: (x = 28 + 18 * col, y = 16 + 18 * row)
        box_size = 18
        row = (slot - 1) // 3
        col = (slot - 1) % 3
        left_x = 27 + (box_size * col)
        top_y = 15 + (box_size * row)
    # inventory
    elif slot < 37:
        # slot size: 18x18
        # top left corner: (x= 6 + 18 * col, y=83 + 18 * row)
        box_size = 18
        row = (slot - 10) // 9
        col = (slot - 10) % 9
        left_x = 5 + (box_size * col)
        top_y = 82 + (box_size * row)
    # hotbar
    else:
        # slot size: 18x18
        # top left corner: (x= 6 + 18 * col, y=141)
        box_size = 18
        col = (slot - 37) % 9
        left_x = 5 + (box_size * col)
        top_y = 140
    return [left_x, top_y, left_x + box_size, top_y + box_size]


def sample_starting_inv(seen_items):
    inventory = []
    # random number of items
    selected_slots = random.sample(range(1, 45), random.randint(1, 44))
    for slot in selected_slots:
        inventory.append(
            {
                "type": random.choice(list(seen_items)),
                "slot": slot,
                "quantity": random.randint(1, 64),
            }
        )
    # sort by slot
    inventory = sorted(inventory, key=lambda x: x["slot"])
    return inventory


def sample_environment(batch_size=32, seen_items=None, N=100):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    inventory = sample_starting_inv(seen_items)
    env = RealPlancraft(
        inventory=inventory,
        symbolic_action_space=True,
        symbolic_observation_space=True,
        resolution=[512, 512],
        crop=True,
    )
    env.reset()
    print("Env loaded")

    i = 0
    while i < N:
        batch_images = []
        batch_targets = []
        for _ in range(batch_size):
            starting_inv = sample_starting_inv(seen_items)
            env.fast_reset(new_inventory=starting_inv)
            obs, _, _, _ = env.step(env.action_space.no_op())

            # create targets/boxes
            target = {"labels": [], "boxes": []}
            for item in starting_inv:
                if item["quantity"] > 0:
                    target["labels"].append(item["quantity"])
                    target["boxes"].append(slot_to_bbox(item["slot"]))

            # convert to tensors
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64) - 1
            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.int64)
            img_tensor = transform(obs["pov"].copy())

            batch_images.append(img_tensor)
            batch_targets.append(target)

        yield torch.stack(batch_images), batch_targets
        i += 1


if __name__ == "__main__":
    print("loaded env")
    with open("data/train.json", "r") as f:
        train = json.load(f)
    seen_items = set()
    for example in train:
        seen_items.update(example["inventory"].keys())
        seen_items.add(example["target"])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 64)
    model = model.cuda()
    print("Loaded model")

    N = 500
    lr = 0.0001

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    wandb.init(project="plancraft-img-encoder", entity="itl")
    for images, targets in sample_environment(seen_items=seen_items, N=N, batch_size=4):
        images = images.cuda()
        for i in range(len(targets)):
            targets[i]["boxes"] = targets[i]["boxes"].cuda()
            targets[i]["labels"] = targets[i]["labels"].cuda()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        wandb.log(loss_dict)
        wandb.log({"train_loss": losses})
        print(loss_dict)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    wandb.finish()

    # save model
    torch.save(model.state_dict(), "minecraft_rcnn.pth")
