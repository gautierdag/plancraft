import os
import random
import time

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from minerl.herobraine.hero.mc import ALL_ITEMS
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.roi_heads import (
    fastrcnn_loss,
    keypointrcnn_inference,
    keypointrcnn_loss,
    maskrcnn_inference,
    maskrcnn_loss,
)
from torchvision.ops import boxes as box_ops

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
        self.env = RealPlancraft(
            inventory=sample_starting_inv(),
            symbolic_action_space=True,
            symbolic_observation_space=True,
            resolution=[512, 512],
            crop=True,
        )
        self.env.reset()

    def reset(self):
        self.env.reset()

    def step(self, starting_inv: list[dict[str, int]]):
        try:
            self.env.fast_reset(new_inventory=starting_inv)
            obs, _, _, _ = self.env.step(self.env.action_space.no_op())
            return obs
        except RuntimeError:
            print("Env reset due to RuntimeError")
            self.env.close()
            time.sleep(5)
            # reset env
            self.env = RealPlancraft(
                inventory=starting_inv,
                symbolic_action_space=True,
                symbolic_observation_space=True,
                resolution=[512, 512],
                crop=True,
            )
            self.env.reset()
            self.env.fast_reset(new_inventory=starting_inv)
            obs, _, _, _ = self.env.step(self.env.action_space.no_op())
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
        while len(batch_images) < batch_size:
            starting_inv = sample_starting_inv()
            obs = env.step(starting_inv)
            # create targets/boxes
            target = {"labels": [], "boxes": [], "quantity_labels": []}
            inv = []
            for item in obs["inventory"]:
                if item["quantity"] > 0:
                    target["labels"].append(ALL_ITEMS.index(item["type"]))
                    target["boxes"].append(slot_to_bbox(item["index"]))
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


def postprocess_detections_custom(
    self,
    class_logits,
    quantity_logits,
    box_regression,
    proposals,
    image_shapes,
):
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_quantity = F.softmax(quantity_logits, -1).argmax(dim=-1)
    # repeat the quantities, once for each class
    pred_quantity = einops.repeat(
        pred_quantity, "n -> n c", c=num_classes, n=pred_quantity.shape[0]
    )

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)
    pred_quantity_list = pred_quantity.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_quantity_labels = []

    for boxes, scores, quantities, image_shape in zip(
        pred_boxes_list, pred_scores_list, pred_quantity_list, image_shapes
    ):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        quantities = quantities[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        quantities = quantities.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels, quantities = (
            boxes[inds],
            scores[inds],
            labels[inds],
            quantities[inds],
        )

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, quantities = (
            boxes[keep],
            scores[keep],
            labels[keep],
            quantities[keep],
        )

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels, quantities = (
            boxes[keep],
            scores[keep],
            labels[keep],
            quantities[keep],
        )

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_quantity_labels.append(quantities)

    return all_boxes, all_scores, all_labels, all_quantity_labels


def forward_custom(
    self,
    features,
    proposals,
    image_shapes,
    targets=None,
):
    if targets is not None:
        for t in targets:
            floating_point_types = (torch.float, torch.double, torch.half)
            if t["boxes"].dtype not in floating_point_types:
                raise TypeError(
                    f"target boxes must of float type, instead got {t['boxes'].dtype}"
                )
            if not t["labels"].dtype == torch.int64:
                raise TypeError(
                    f"target labels must of int64 type, instead got {t['labels'].dtype}"
                )
            if self.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(
                        f"target keypoints must of float type, instead got {t['keypoints'].dtype}"
                    )

    if self.training:
        proposals, matched_idxs, labels, regression_targets = (
            self.select_training_samples(proposals, targets)
        )
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result = []
    losses = {}
    if self.training:
        if labels is None:
            raise ValueError("labels cannot be None")
        if regression_targets is None:
            raise ValueError("regression_targets cannot be None")
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets
        )

        # custom addition to calculate quantity loss
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["quantity_labels"] for t in targets]
        _, quantity_labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        quantity_labels = torch.cat(quantity_labels, dim=0)
        # needs quantity_prediction layer to be added to class
        quantity_preds = self.quantity_prediction(box_features)
        loss_classsifier_quantity = F.cross_entropy(
            quantity_preds,
            quantity_labels,
        )
        losses = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_classifier_quantity": loss_classsifier_quantity,
        }
    else:
        quantity_logits = self.quantity_prediction(box_features)

        boxes, scores, labels, quantities = postprocess_detections_custom(
            self, class_logits, quantity_logits, box_regression, proposals, image_shapes
        )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "quantities": quantities[i],
                }
            )

    if self.has_mask():
        mask_proposals = [p["boxes"] for p in result]
        if self.training:
            if matched_idxs is None:
                raise ValueError("if in training, matched_idxs should not be None")

            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}
        if self.training:
            if targets is None or pos_matched_idxs is None or mask_logits is None:
                raise ValueError(
                    "targets, pos_matched_idxs, mask_logits cannot be None when training"
                )

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            rcnn_loss_mask = maskrcnn_loss(
                mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
            )
            loss_mask = {"loss_mask": rcnn_loss_mask}
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

    # keep none checks in if conditional so torchscript will conditionally
    # compile each branch
    if (
        self.keypoint_roi_pool is not None
        and self.keypoint_head is not None
        and self.keypoint_predictor is not None
    ):
        keypoint_proposals = [p["boxes"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            keypoint_proposals = []
            pos_matched_idxs = []
            if matched_idxs is None:
                raise ValueError("if in trainning, matched_idxs should not be None")

            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                keypoint_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        keypoint_features = self.keypoint_roi_pool(
            features, keypoint_proposals, image_shapes
        )
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        loss_keypoint = {}
        if self.training:
            if targets is None or pos_matched_idxs is None:
                raise ValueError(
                    "both targets and pos_matched_idxs should not be None when in training mode"
                )

            gt_keypoints = [t["keypoints"] for t in targets]
            rcnn_loss_keypoint = keypointrcnn_loss(
                keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
            )
            loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
        else:
            if keypoint_logits is None or keypoint_proposals is None:
                raise ValueError(
                    "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                )

            keypoints_probs, kp_scores = keypointrcnn_inference(
                keypoint_logits, keypoint_proposals
            )
            for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps
        losses.update(loss_keypoint)

    return result, losses


class IntegratedBoundingBoxModel(nn.Module):
    def __init__(self):
        super(IntegratedBoundingBoxModel, self).__init__()
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            image_mean=[0.63, 0.63, 0.63],
            image_std=[0.21, 0.21, 0.21],
            min_size=64,
            max_size=64,
            num_classes=len(ALL_ITEMS),
            box_score_thresh=0.001,
        )
        # self.model.roi_heads.quantity_prediction = nn.Linear(1024, 65)
        # self.model.roi_heads.forward = forward_custom.__get__(
        #     self.model.roi_heads, type(self.model.roi_heads)
        # )

    def forward(self, x, targets=None):
        if self.training:
            # normal forward pass
            loss_dict = self.model(x, targets)
            return loss_dict
        else:
            preds = self.model(x)
            return preds

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))


if __name__ == "__main__":
    m1_path = "m1.pth"
    M1_model = IntegratedBoundingBoxModel()
    M1_model = M1_model.cuda()
    print("Loaded model")

    N = 10000
    m1_lr = 0.001
    batch_size = 4
    save_every = 100
    count = 0
    m1_optimizer = torch.optim.AdamW(M1_model.parameters(), lr=m1_lr)

    wandb.init(project="plancraft-img-encoder", entity="itl")  # , mode="disabled")
    for images, targets, raw_images, inv in sample_environment(
        N=N, batch_size=batch_size
    ):
        M1_model.train()
        images = images.cuda()
        for i in range(len(targets)):
            targets[i]["boxes"] = targets[i]["boxes"].cuda()
            targets[i]["labels"] = targets[i]["labels"].cuda()
            targets[i]["quantity_labels"] = targets[i]["quantity_labels"].cuda()

        m1_loss_dict = M1_model(images, targets)

        # artificially decrease the classifier loss
        # m1_loss_dict["loss_classifier"] = m1_loss_dict["loss_classifier"] * 0.5
        # m1_loss_dict["loss_classifier_quantity"] = (
        #     m1_loss_dict["loss_classifier_quantity"] * 0.5
        # )

        m1_losses = sum(loss for loss in m1_loss_dict.values())
        wandb.log(m1_loss_dict)
        wandb.log({"m1_train_loss": m1_losses})
        print(m1_loss_dict)

        m1_optimizer.zero_grad()
        m1_losses.backward()
        m1_optimizer.step()

        if count % save_every == 0:
            M1_model.eval()
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
                    # quantity = predictions[img_idx]["quantities"][box_idx]

                    # if score > 0:
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(box.cpu().tolist(), outline="green")
                    draw.text(
                        (box[0], box[1]),
                        f"{label.item()}",
                        fill="green",
                    )
                    # draw.text(
                    #     (box[0], box[1] + 10),
                    #     f"{quantity.item()}",
                    #     fill="blue",
                    # )

                # log image
                wandb.log({f"image_{img_idx}": wandb.Image(img)})

        count += 1

    wandb.finish()

    # save model
    M1_model.save(m1_path)
