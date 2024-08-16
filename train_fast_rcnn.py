import random

import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import wandb
from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.recipes import RECIPES, ShapedRecipe, ShapelessRecipe
from plancraft.environments.sampler import sample_distractors

CRAFTING_RECIPES = [
    r
    for recipes in RECIPES.values()
    for r in recipes
    if isinstance(r, ShapelessRecipe) or isinstance(r, ShapedRecipe)
]

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


def split_image_by_bboxes(
    image: Image.Image, bboxes: list[tuple[int, int, int, int]]
) -> list[Image.Image]:
    """
    Splits an image into sub-images based on the provided bounding boxes.

    Parameters:
        - image: A PIL Image to be split.
        - bboxes: A list of bounding boxes, each defined by (x_min, y_min, x_max, y_max).

    Returns:
        - A list of PIL Image objects corresponding to the cropped regions of the original image.
    """
    cropped_images = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        # Crop the image using the bounding box coordinates
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        # set to 64x64
        cropped_image = cropped_image.resize((64, 64))
        cropped_images.append(cropped_image)
    return cropped_images


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


def sample_environment(batch_size=32, N=100):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    inventory = sample_starting_inv()
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
        batch_images_raw = []
        while len(batch_images) < batch_size:
            starting_inv = sample_starting_inv()
            try:
                env.fast_reset(new_inventory=starting_inv)
                obs, _, _, _ = env.step(env.action_space.no_op())
            except RuntimeError:
                print("Env reset due to RuntimeError")
                # reset env
                # env = RealPlancraft(
                #     inventory=starting_inv,
                #     symbolic_action_space=True,
                #     symbolic_observation_space=True,
                #     resolution=[512, 512],
                #     crop=True,
                # )
                # env.reset()
                # env.fast_reset(new_inventory=starting_inv)
                # obs, _, _, _ = env.step(env.action_space.no_op())
                return

            # create targets/boxes
            target = {"labels": [], "boxes": []}
            for item in obs["inventory"]:
                if item["quantity"] > 0:
                    target["labels"].append(item["quantity"])
                    target["boxes"].append(slot_to_bbox(item["index"]))

            # convert to tensors
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.int64)

            if len(target["labels"]) == 0:
                continue

            img_tensor = transform(obs["pov"].copy())
            batch_images.append(img_tensor)
            batch_targets.append(target)
            batch_images_raw.append(obs["pov"])

        yield torch.stack(batch_images), batch_targets, batch_images_raw
        i += 1


class M1(torch.nn.Module):
    """
    Model 1: FasterRCNN with ResNet50 backbone
    This model splits the image into bounding boxes and detects the quantity of items in each box.

    The class labels are 65 (64 items + 1 background class)
    """

    def __init__(self):
        super(M1, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 65)

    def forward(self, x, targets=None):
        if self.training:
            return self.model(x, targets)
        else:
            return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))


class M2(torch.nn.Module):
    """
    Model 2: Autoencoder for bounding boxes
    This model uses an autoencoder to encode the pixels of each bounding boxes,
    their location, and the quantity of items and predicts the type, location,
    quantity of each item in the image.

    Inputs:
        - bbox: the image at the bounding box
        - bbox_location: the 2 coordinates of the middle of the bounding box
        - item_quantity: the quantity of the item in the bounding box

    Outputs (training):
        - type: the type of item in the bounding box
        - location: the location of the bounding box in the image (2 coordinates - middle of the box)
        - quantity: the quantity of the item in the bounding box

    Outputs (inference):
        - intermediate vector representation at bottleneck layer
    """

    def __init__(self):
        super(M2, self).__init__()

    def forward(self, bboxes, targets=None):
        pass


class M3(torch.nn.Module):
    """
    Model 3: Takes in an image and outputs a list of vectors that describes the items

    Uses model 1 to detect the items in the image and then encodes the bounding boxes
    and then uses model 2 to encode the predictions from model 1 as vectors.

    """

    def __init__(self):
        super(M3, self).__init__()

    def forward(self, x, targets=None):
        pass


if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 65)

    weights_path = "minecraft_rcnn.pth"
    # model.load_state_dict(torch.load(weights_path, weights_only=True))

    model = model.cuda()
    print("Loaded model")

    N = 1000
    lr = 0.001
    batch_size = 4
    save_every = 100
    count = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    wandb.init(project="plancraft-img-encoder", entity="itl")
    for images, targets, raw_images in sample_environment(N=N, batch_size=batch_size):
        model.train()

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

        if count % save_every == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(images)
            # save model
            torch.save(model.state_dict(), weights_path)

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

                    if score > 0.5:
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(box.cpu().tolist(), outline="green")
                        draw.text(
                            (box[0], box[1]),
                            f"{label.item()}",
                            fill="green",
                        )
                # log image
                wandb.log({f"image_{img_idx}": wandb.Image(img)})

        count += 1

    wandb.finish()

    # save model
    torch.save(model.state_dict(), weights_path)
