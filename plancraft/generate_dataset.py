import random
import concurrent.futures
from functools import partial

import json
import numpy as np

from tqdm import tqdm

from plancraft.environment.recipes import RECIPES
from plancraft.environment.sampler import construct_example


def reset_construct_example(target, num_distractors, impossible):
    """
    Simple retry mechanism if planner takes too long (can happen because of random sampling of initial conditions).
    """
    try:
        return construct_example(
            target=target, num_distractors=num_distractors, impossible=impossible
        )
    except TimeoutError:
        print(f"TimeoutError for {target} with {num_distractors} distractors.")
        return reset_construct_example(
            target=target, num_distractors=num_distractors, impossible=impossible
        )


def process_target(recipe_target, distractors, dataset, pbar):
    """
    Process a single target recipe with different numbers of distractors and impossible vs possible
    using a thread pool executor.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for num_distractors in distractors:
            construct_partial_false = partial(
                reset_construct_example, recipe_target, num_distractors, False
            )
            construct_partial_true = partial(
                reset_construct_example, recipe_target, num_distractors, True
            )
            for _ in range(10):
                future = executor.submit(construct_partial_false)
                futures.append((future, 1))
            for _ in range(5):
                future = executor.submit(construct_partial_true)
                futures.append((future, 1))

        for future, update in futures:
            example = future.result()
            dataset.append(example)
            pbar.update(update)


if __name__ == "__main__":
    seed = 2024
    distractors = [4, 8, 16]

    random.seed(seed)
    np.random.seed(seed)

    dataset = []
    with tqdm(total=len(RECIPES) * len(distractors) * 15) as pbar:
        for recipe_target in list(RECIPES.keys()):
            if len(RECIPES[recipe_target]) == 0:
                continue
            process_target(recipe_target, distractors, dataset, pbar)

    # save the dataset
    with open("dataset.json", "w") as file:
        json.dump(dataset, file, indent=4)
