import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def get_torch_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            device = torch.device("mps")
    return device


def resize_image(img, target_resolution=(128, 128)):
    if type(img) == np.ndarray:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    elif type(img) == torch.Tensor:
        img = F.interpolate(img, size=target_resolution, mode="bilinear")
    else:
        raise ValueError("Unsupported type for img")
    return img


def preprocess_obs(obs: dict, device=torch.device("cpu")) -> dict:
    res_obs = {}
    rgb = (
        torch.from_numpy(obs["rgb"])
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
    )
    res_obs["rgb"] = resize_image(rgb, target_resolution=(120, 160))
    res_obs["voxels"] = (
        torch.from_numpy(obs["voxels"])
        .reshape(-1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.long)
    )
    res_obs["compass"] = (
        torch.from_numpy(obs["compass"])
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    res_obs["gps"] = (
        torch.from_numpy(obs["gps"] / np.array([1000.0, 100.0, 1000.0]))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    res_obs["biome"] = (
        torch.from_numpy(obs["biome_id"])
        .unsqueeze(0)
        .to(device=device, dtype=torch.long)
    )
    return res_obs


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


def stack_obs(prev_obs: dict, obs: dict) -> dict:
    stacked_obs = {}
    stacked_obs["rgb"] = torch.cat([prev_obs["rgb"], obs["rgb"]], dim=0)
    stacked_obs["voxels"] = torch.cat([prev_obs["voxels"], obs["voxels"]], dim=0)
    stacked_obs["compass"] = torch.cat([prev_obs["compass"], obs["compass"]], dim=0)
    stacked_obs["gps"] = torch.cat([prev_obs["gps"], obs["gps"]], dim=0)
    stacked_obs["biome"] = torch.cat([prev_obs["biome"], obs["biome"]], dim=0)
    return stacked_obs


def slice_obs(obs: dict, slice: torch.tensor) -> dict:
    res = {}
    for k, v in obs.items():
        res[k] = v[slice]
    return res
