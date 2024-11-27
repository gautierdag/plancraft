import glob
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image


def get_downloaded_models() -> dict:
    """
    Get the list of downloaded models on the NFS partition (EIDF).
    """
    downloaded_models = {}
    # known models on NFS partition
    if pathlib.Path("/nfs").exists():
        local_models = glob.glob("/nfs/public/hf/models/*/*")
        downloaded_models = {
            model.replace("/nfs/public/hf/models/", ""): model for model in local_models
        }
    return downloaded_models


def get_torch_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.info(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            device = torch.device("mps")
    return device


def resize_image(img, target_resolution=(128, 128)):
    if type(img) is np.ndarray:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    elif type(img) is torch.Tensor:
        img = F.interpolate(img, size=target_resolution, mode="bilinear")
    else:
        raise ValueError("Unsupported type for img")
    return img


def save_frames_to_video(frames: list, out_path: str):
    imgs = []
    for id, (frame, goal) in enumerate(frames):
        # if torch.is_tensor(frame):
        # frame = frame.permute(0, 2, 3, 1).cpu().numpy()

        frame = resize_image(frame, (320, 240)).astype("uint8")
        cv2.putText(
            frame,
            f"FID: {id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Goal: {goal}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        imgs.append(Image.fromarray(frame))
    imgs = imgs[::3]
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        optimize=False,
        quality=0,
        duration=150,
        loop=0,
    )
