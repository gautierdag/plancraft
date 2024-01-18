import torch
import torch.nn.functional as F
import numpy as np
import cv2


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
        raise ValueError
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
