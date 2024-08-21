import random
import os

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms.v2 as v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import mobilenet_v3_small

import wandb
from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.recipes import RECIPES, ShapedRecipe, ShapelessRecipe
from plancraft.environments.sampler import sample_distractors

from minerl.herobraine.hero.mc import ALL_ITEMS

CRAFTING_RECIPES = [
    r
    for recipes in RECIPES.values()
    for r in recipes
    if isinstance(r, ShapelessRecipe) or isinstance(r, ShapedRecipe)
]

wandb.require("core")


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


m2_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.508, 0.492, 0.476], std=[0.241, 0.244, 0.255]),
    ]
)


def prepare_m2_batch(img, inventory):
    items = [item for item in inventory if item["quantity"] > 0]
    bbox_images = split_image_by_bboxes(
        img, [slot_to_bbox(item["index"]) for item in items]
    )

    bbox_tensors = []
    for bbox_image in bbox_images:
        bbox_tensors.append(m2_transforms(bbox_image))

    item_locations = []
    for item in items:
        x_min, y_min, x_max, y_max = slot_to_bbox(item["index"])
        # TODO normalize and add small noise
        x_center = (x_min + x_max) / 2 / 175
        y_center = (y_min + y_max) / 2 / 175
        item_locations.append((x_center, y_center))

    item_quantity = torch.tensor([item["quantity"] for item in items]) / 64
    item_locations = torch.tensor(item_locations)
    item_types = torch.tensor([ALL_ITEMS.index(item["type"]) for item in items])

    batch = {}
    batch["bbox_tensors"] = bbox_tensors
    batch["item_quantity"] = item_quantity
    batch["item_locations"] = item_locations
    batch["item_types"] = item_types

    return batch


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
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
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
        batch_inventory = []
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
            inv = []
            for item in obs["inventory"]:
                if item["quantity"] > 0:
                    target["labels"].append(item["quantity"])
                    target["boxes"].append(slot_to_bbox(item["index"]))
                    inv.append(item)

            # convert to tensors
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.int64)

            if len(target["labels"]) == 0:
                continue

            img_tensor = transform(obs["pov"].copy())
            batch_images.append(img_tensor)
            batch_targets.append(target)
            batch_images_raw.append(obs["pov"])
            batch_inventory.append(inv)

        yield (
            torch.stack(batch_images),
            batch_targets,
            batch_images_raw,
            batch_inventory,
        )
        i += 1


class M1(torch.nn.Module):
    """
    Model 1: FasterRCNN with ResNet50 backbone
    This model splits the image into bounding boxes and detects the quantity of items in each box.

    The class labels are 65 (64 items + 1 background class)
    """

    def __init__(self, weights_path=None):
        super(M1, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=None
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 65)

        if weights_path:
            self.load(weights_path)

    def forward(self, x, targets=None):
        if self.training:
            return self.model(x, targets)
        else:
            return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
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

    def __init__(self, dropout=0.05, weights_path=None):
        super(M2, self).__init__()
        # image encoder
        self.img_encoder = mobilenet_v3_small(pretrained=False)
        self.img_encoder.classifier = nn.Identity()

        # load image encoder weights from "mobilenet.pth"
        self.img_encoder.load_state_dict(
            torch.load(
                "mobilenet.pth", map_location=torch.device("cpu"), weights_only=True
            )
        )

        self.quantity_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.location_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        N = 576 + 64 + 128
        self.intermediate = nn.Sequential(
            nn.Linear(N, 256 * 3),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 3, 256 * 3),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.quantity_decoder = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.location_decoder = nn.Sequential(
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

        self.type_decoder = nn.Sequential(
            nn.Linear(256, len(ALL_ITEMS)),
            nn.Softmax(dim=1),
        )
        if weights_path:
            self.load(weights_path)

    def forward(self, batch):
        if len(batch["item_quantity"].shape) == 1:
            batch["item_quantity"] = batch["item_quantity"].unsqueeze(1)

        encoded_quantities = self.quantity_encoder(batch["item_quantity"])
        encoded_locations = self.location_encoder(batch["item_locations"])
        encoded_bboxes = self.img_encoder(batch["bbox_tensors"])
        # flatten

        encoded_features = torch.cat(
            [encoded_bboxes, encoded_quantities, encoded_locations], dim=1
        )
        intermediate_representation = self.intermediate(encoded_features)

        if not self.training:
            # concatenate all features
            return intermediate_representation

        quantities, locations, object_type = torch.split(
            intermediate_representation, [256, 256, 256], dim=1
        )

        quantities = self.quantity_decoder(quantities)
        locations = self.location_decoder(locations)
        object_type = self.type_decoder(object_type)

        # loss functions
        loss_quantity = nn.MSELoss()(quantities, batch["item_quantity"])
        loss_location = nn.MSELoss()(locations, batch["item_locations"])
        loss_type = nn.CrossEntropyLoss()(object_type, batch["item_types"])

        return (
            {
                "loss_quantity": loss_quantity,
                "loss_location": loss_location,
                "loss_type": loss_type,
            },
            object_type.argmax(dim=1),
        )

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))


# class M3(torch.nn.Module):
#     """
#     Model 3: Takes in an image and outputs a list of vectors that describes the items
#     Uses model 1 to detect the items in the image and then encodes the bounding boxes
#     and then uses model 2 to encode the predictions from model 1 as vectors.
#     """
#     def __init__(self):
#         super(M3, self).__init__()
#     def forward(self, x, targets=None):
#         pass


if __name__ == "__main__":
    m1_path = "m1.pth"
    m2_path = "m2.pth"
    # m1_path = None
    # m2_path = None

    # M1_model = M1(weights_path=m1_path)
    # M1_model = M1_model.cuda()

    M2_model = M2(weights_path=None)
    M2_model = M2_model.cuda()
    print("Loaded model")

    N = 1000
    m1_lr = 0.001
    m2_lr = 0.001
    batch_size = 4
    save_every = 100
    count = 0
    # m1_optimizer = torch.optim.AdamW(M1_model.parameters(), lr=m1_lr)
    m2_optimizer = torch.optim.AdamW(M2_model.parameters(), lr=m2_lr)

    wandb.init(project="plancraft-img-encoder", entity="itl")  # , mode="disabled")
    for images, targets, raw_images, inv in sample_environment(
        N=N, batch_size=batch_size
    ):
        # M1_model.train()
        # images = images.cuda()
        # for i in range(len(targets)):
        #     targets[i]["boxes"] = targets[i]["boxes"].cuda()
        #     targets[i]["labels"] = targets[i]["labels"].cuda()

        # m1_loss_dict = M1_model(images, targets)
        # m1_losses = sum(loss for loss in m1_loss_dict.values())
        # wandb.log(m1_loss_dict)
        # wandb.log({"m1_train_loss": m1_losses})
        # print(m1_loss_dict)

        # m1_optimizer.zero_grad()
        # m1_losses.backward()
        # m1_optimizer.step()

        # train M2
        M2_model.train()
        bbox_count = 0
        correct_count = 0
        m2_losses = 0
        for img, inventory in zip(raw_images, inv):
            i = Image.fromarray(img.copy())

            m2_batch = prepare_m2_batch(i, inventory)
            m2_batch["bbox_tensors"] = torch.stack(m2_batch["bbox_tensors"]).cuda()
            m2_batch["item_quantity"] = m2_batch["item_quantity"].cuda()
            m2_batch["item_locations"] = m2_batch["item_locations"].cuda()
            m2_batch["item_types"] = m2_batch["item_types"].cuda()

            m2_loss_dict, preds = M2_model(m2_batch)
            wandb.log(m2_loss_dict)

            m2_losses += sum(loss for loss in m2_loss_dict.values())
            correct_count += (preds == m2_batch["item_types"]).sum().item()
            bbox_count += len(preds)

            # save images
            # for idx in range(len(m2_batch["bbox_tensors"])):
            #     img = v2.ToPILImage()(m2_batch["bbox_tensors"][idx])
            #     name = ALL_ITEMS[m2_batch["item_types"][idx].item()]
            #     # save to output
            #     img.save(f"data/bboxes/{count+idx}_{name}.png")

        wandb.log({"m2_train_loss": m2_losses})
        m2_optimizer.zero_grad()
        m2_losses.backward()
        m2_optimizer.step()

        batch_acc = correct_count / bbox_count
        wandb.log({"m2_type_accuracy": batch_acc})

        if count % save_every == 0:
            # M1_model.eval()
            # with torch.no_grad():
            #     predictions = M1_model(images)
            # # save model
            # M1_model.save(m1_path)
            M2_model.save(m2_path)

        # for img_idx in range(len(images)):
        #     # generate image and target for validation
        #     img = Image.fromarray(raw_images[img_idx].copy())
        #     for box in targets[img_idx]["boxes"]:
        #         draw = ImageDraw.Draw(img)
        #         draw.rectangle(box.cpu().tolist(), outline="red")

        #     for box_idx in range(len(predictions[img_idx]["boxes"])):
        #         box = predictions[img_idx]["boxes"][box_idx]
        #         score = predictions[img_idx]["scores"][box_idx]
        #         label = predictions[img_idx]["labels"][box_idx]

        #         if score > 0.5:
        #             draw = ImageDraw.Draw(img)
        #             draw.rectangle(box.cpu().tolist(), outline="green")
        #             draw.text(
        #                 (box[0], box[1]),
        #                 f"{label.item()}",
        #                 fill="green",
        #             )
        #     # log image
        #     wandb.log({f"image_{img_idx}": wandb.Image(img)})

        count += 1

    wandb.finish()

    # save model
    # torch.save(M1_model.state_dict(), m1_path)
    torch.save(M2_model.state_dict(), m2_path)
