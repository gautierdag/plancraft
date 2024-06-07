import os
import glob

import numpy as np
import imageio


def get_few_shot_images_path():
    return os.path.dirname(__file__)


def load_prompt_images() -> list[np.ndarray]:
    current_dir = get_few_shot_images_path()
    files = glob.glob(os.path.join(current_dir, "*.png"))
    images = [imageio.imread(file) for file in files]
    return images
