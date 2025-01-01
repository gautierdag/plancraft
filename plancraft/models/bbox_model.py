from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models.detection.faster_rcnn import (
    ResNet50_Weights,
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

from plancraft.environment.items import ALL_ITEMS

from functools import lru_cache


@lru_cache
def slot_to_bbox(
    slot: int, resolution: Literal["low", "medium", "high"] = "high"
) -> list[int]:
    # crafting slot
    if slot == 0:
        # slot size: 25x25
        # top left corner: (x= 118, y=30)
        box_size = 26
        left_x = 119
        top_y = 30
    # crafting grid
    elif slot < 10:
        # slot size: 18x18
        # top left corner: (x = 28 + 18 * col, y = 16 + 18 * row)
        box_size = 18
        row = (slot - 1) // 3
        col = (slot - 1) % 3
        left_x = 29 + (box_size * col)
        top_y = 16 + (box_size * row)
    # inventory
    elif slot < 37:
        # slot size: 18x18
        # top left corner: (x= 6 + 18 * col, y=83 + 18 * row)
        box_size = 18
        row = (slot - 10) // 9
        col = (slot - 10) % 9
        left_x = 7 + (box_size * col)
        top_y = 83 + (box_size * row)
    # hotbar
    else:
        # slot size: 18x18
        # top left corner: (x= 6 + 18 * col, y=141)
        box_size = 18
        col = (slot - 37) % 9
        left_x = 7 + (box_size * col)
        top_y = 141
    bounding_box = [left_x, top_y, left_x + box_size, top_y + box_size]
    if resolution == "medium":
        bounding_box = [i * 2 for i in bounding_box]
    elif resolution == "high":
        bounding_box = [i * 4 for i in bounding_box]
    return bounding_box


def postprocess_detections_custom(
    self,
    class_logits,
    quantity_logits,
    box_features,
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
    pred_features_list = box_features.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_quantity_labels = []
    all_features = []

    for boxes, scores, quantities, features, image_shape in zip(
        pred_boxes_list,
        pred_scores_list,
        pred_quantity_list,
        pred_features_list,
        image_shapes,
    ):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        box_idxs = (
            torch.arange(boxes.size(0), device=device).view(-1, 1).expand_as(labels)
        )

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        quantities = quantities[:, 1:]
        box_idxs = box_idxs[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        quantities = quantities.reshape(-1)
        box_idxs = box_idxs.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels, quantities, box_idxs = (
            boxes[inds],
            scores[inds],
            labels[inds],
            quantities[inds],
            box_idxs[inds],
        )

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, quantities, box_idxs = (
            boxes[keep],
            scores[keep],
            labels[keep],
            quantities[keep],
            box_idxs[keep],
        )

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels, quantities, box_idxs = (
            boxes[keep],
            scores[keep],
            labels[keep],
            quantities[keep],
            box_idxs[keep],
        )

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_quantity_labels.append(quantities)
        all_features.append(features[box_idxs])

    return all_boxes, all_scores, all_labels, all_quantity_labels, all_features


def forward_custom(
    self,
    features,
    proposals,
    image_shapes,
    targets=None,
):
    training = False
    if targets is not None:
        training = True
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

    if training:
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
    if training:
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

        boxes, scores, labels, quantities, features = postprocess_detections_custom(
            self,
            class_logits,
            quantity_logits,
            box_features,
            box_regression,
            proposals,
            image_shapes,
        )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "quantities": quantities[i],
                    "features": features[i],
                }
            )

    if self.has_mask():
        mask_proposals = [p["boxes"] for p in result]
        if training:
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
        if training:
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
        if training:
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
        if training:
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


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def bbox_to_slot_index_iou(bbox: tuple[int, int, int, int], resolution="high") -> int:
    """Assign the given bounding box to the slot with the highest IoU."""
    best_slot = None
    best_iou = -1

    IDX_TO_BBOX = {
        slot: slot_to_bbox(slot, resolution=resolution) for slot in range(46)
    }

    # Iterate through all precomputed slot bounding boxes
    for slot, slot_bbox in IDX_TO_BBOX.items():
        iou = calculate_iou(bbox, slot_bbox)
        if iou > best_iou:
            best_iou = iou
            best_slot = slot
    return best_slot


class IntegratedBoundingBoxModel(nn.Module, PyTorchModelHubMixin):
    """
    Custom mask rcnn model with quantity prediction

    Also returns the feature vectors of the detected boxes
    """

    def __init__(self, load_resnet_weights=False):
        super(IntegratedBoundingBoxModel, self).__init__()
        weights = None
        if load_resnet_weights:
            weights = ResNet50_Weights.DEFAULT

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights_backbone=weights,
            image_mean=[0.63, 0.63, 0.63],
            image_std=[0.21, 0.21, 0.21],
            min_size=128,
            max_size=512,
            num_classes=len(ALL_ITEMS),
            box_score_thresh=0.05,
            rpn_batch_size_per_image=64,
            box_detections_per_img=64,
            box_batch_size_per_image=128,
        )
        self.model.roi_heads.quantity_prediction = nn.Linear(1024, 65)

        # replace the head with leaky activations
        self.model.roi_heads.forward = forward_custom.__get__(
            self.model.roi_heads, type(self.model.roi_heads)
        )

        self.transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def forward(self, x, targets=None):
        if self.training:
            # normal forward pass
            loss_dict = self.model(x, targets)
            return loss_dict
        else:
            preds = self.model(x)
            return preds

    def get_inventory(self, pil_image, resolution="high") -> dict:
        """
        Predict boxes and quantities
        """
        img_tensor = self.transform(pil_image)
        if next(self.model.parameters()).is_cuda:
            img_tensor = img_tensor.cuda()
        with torch.no_grad():
            predictions = self.model(img_tensor.unsqueeze(0))
        return self.prediction_to_inventory(predictions[0], resolution=resolution)

    @staticmethod
    def prediction_to_inventory(prediction, threshold=0.9, resolution="high") -> dict:
        inventory = {}
        seen_slots = set()
        for bbox, score, label, quantity in zip(
            prediction["boxes"],
            prediction["scores"],
            prediction["labels"],
            prediction["quantities"],
        ):
            slot = bbox_to_slot_index_iou(bbox, resolution=resolution)
            if score < threshold:
                break
            if slot in seen_slots:
                continue
            label = ALL_ITEMS[label.item()]
            quantity = quantity.item()
            inventory["slot"] = {"type": label, "quantity": quantity}
            seen_slots.add(slot)
        return inventory

    def freeze(self):
        # NOTE: this might seem excessive
        # but transformers trainer is really good at enabling gradients against my will
        self.eval()
        self.model.eval()
        self.training = False
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.training = False
        self.model.roi_heads.training = False
        self.model.rpn.training = False

    def save(self, path: str):
        torch.save(self.state_dict(), path)
