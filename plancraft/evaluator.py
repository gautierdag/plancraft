import os
import time
import json
import logging
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm

import wandb
from plancraft.config import Config, PlancraftExample
from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.env_symbolic import SymbolicPlancraft
from plancraft.models import get_model


logger = logging.getLogger(__name__)


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment is created based on the configuration and the examples are loaded from the dataset.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir = (
            f"{cfg.plancraft.output_dir}/{self.evaluator_name()}/{cfg.plancraft.split}"
        )
        self.generation_number = 0

        self.examples = self.load_dataset(cfg.plancraft.split)

        self.batch_size = cfg.plancraft.batch_size
        self.envs = [self.create_env(cfg) for _ in range(self.batch_size)]
        self.model = get_model(cfg)

        self.record_frames = not (cfg.plancraft.environment.symbolic)

        # no_op action
        if cfg.plancraft.environment.symbolic:
            self.no_op = {
                "inventory_command": (0, 0, 0),
            }
        else:
            self.no_op = self.envs[0].action_space.no_op()

    def evaluator_name(self) -> str:
        model_name = self.cfg.plancraft.model.split("/")[-1]
        mode = self.cfg.plancraft.mode
        if mode in ["dummy", "oracle"]:
            return f"{mode}"
        return (
            f"{self.cfg.plancraft.mode}_{model_name}_stp{self.cfg.plancraft.max_steps}"
        )

    def save_results_dict(self, example: PlancraftExample, results_dict: dict):
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        json_path = f"{output_dir}/{example.id}.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        wandb.save(json_path, policy="now")

    def load_results_dict(self, example: PlancraftExample) -> dict:
        path = f"{self.output_dir}/{self.generation_number}/{example.id}.json"
        if not os.path.exists(path) or not self.cfg.plancraft.resume:
            return None
        with open(path, "r") as f:
            return json.load(f)

    def create_env(self, cfg: Config) -> RealPlancraft | SymbolicPlancraft:
        if cfg.plancraft.environment.symbolic:
            return SymbolicPlancraft(inventory=self.examples[0].slotted_inventory)
        return RealPlancraft(
            inventory=self.examples[0].slotted_inventory,
            symbolic_action_space=cfg.plancraft.environment.symbolic_action_space,
            symbolic_observation_space=cfg.plancraft.environment.symbolic_observation_space,
            preferred_spawn_biome=cfg.plancraft.environment.preferred_spawn_biome,
            resolution=cfg.plancraft.environment.resolution,
        )

    def close_envs(self):
        for env in self.envs:
            env.close()

    def load_dataset(self, dataset_split: str) -> list[PlancraftExample]:
        with open(f"data/{dataset_split}.json", "r") as f:
            dataset = json.load(f)
            return [PlancraftExample(**example) for example in dataset]

    def reset(
        self,
        example: PlancraftExample,
        env_idx: int = 0,
    ):
        current_inventory = example.slotted_inventory
        self.envs[env_idx].fast_reset(new_inventory=current_inventory)
        objective = f"Craft an item of type: {example.target}"
        self.model.reset_history(history_idx=env_idx, objective=objective)

    def check_done(self, inventory: list[dict[str, int]], target: str):
        """
        Check that target object is obtained
        """
        for item in inventory:
            if target == item["type"]:
                return True
        return False

    def check_stuck(self, env_idx: int, max_steps_no_change: int = 10) -> bool:
        """
        If inventory content does not change for max_steps_no_change steps
        the agent is considered stuck.

        With N=10, the oracle solver can still solve 100% of the examples
        """
        inventory_history = self.model.histories[env_idx].inventory_history
        if len(inventory_history) < max_steps_no_change:
            return False
        inventory_history = inventory_history[-max_steps_no_change:]
        counters = []
        for inventory in inventory_history:
            counter = Counter()
            for item in inventory:
                counter[item["type"]] += item["quantity"]
            counters.append(counter)
        return all(c == counters[0] for c in counters)

    @torch.no_grad()
    def eval_all_examples(self, progress_bar=False) -> list:
        examples_queue = self.examples.copy()
        assigned_examples = {env_idx: None for env_idx in range(self.batch_size)}

        results = []

        actions = [self.no_op.copy() for _ in range(self.batch_size)]
        observations = [None for _ in range(self.batch_size)]
        done = [False for _ in range(self.batch_size)]
        pbar = tqdm(
            total=len(self.examples) * self.cfg.plancraft.max_steps,
            disable=not progress_bar,
        )

        while len(examples_queue) > 0:
            # assign example to environment if not already assigned
            for env_idx, example in assigned_examples.items():
                if example is None and len(examples_queue) > 0:
                    new_example = examples_queue.pop()
                    # TODO: implement flow for impossible examples
                    if new_example.impossible:
                        pbar.update(self.cfg.plancraft.max_steps)
                        continue
                    if resume_result := self.load_results_dict(new_example):
                        pbar.update(self.cfg.plancraft.max_steps)
                        results.append(resume_result)
                        continue

                    assigned_examples[env_idx] = new_example
                    self.reset(new_example, env_idx)
                    actions[env_idx] = self.no_op.copy()
                    continue

                num_steps = self.model.histories[env_idx].num_steps
                if (
                    done[env_idx]
                    or num_steps >= self.cfg.plancraft.max_steps
                    or self.check_stuck(env_idx)
                ):
                    # save results and reset
                    result = {
                        "success": done[env_idx],
                        "number_of_steps": num_steps,
                        "model_trace": self.model.histories[env_idx].trace(),
                        "example_id": example.id,
                    }
                    results.append(result)
                    self.save_results_dict(example, result)
                    assigned_examples[env_idx] = None
                    done[env_idx] = False
                    pbar.update((self.cfg.plancraft.max_steps - num_steps) + 1)

            # step actions
            for env_idx, example in assigned_examples.items():
                if example is None:
                    observations[env_idx] = None
                    continue

                obs, _, _, _ = self.envs[env_idx].step(actions[env_idx])
                observations[env_idx] = obs
                done[env_idx] = self.check_done(obs["inventory"], example.target)
                # # don't predict actions if observation is None
                if done[env_idx]:
                    observations[env_idx] = None

            time_now = time.time()

            # get actions from model (batched)
            actions = self.model.step(observations)
            if any(actions):
                logger.info(
                    f"predicted {len(actions)} actions in {time.time()-time_now:.2f}s"
                )
            pbar.update(len(observations))

        return results

    def eval_all(self):
        logger.info(
            f"Running evaluation over {len(self.examples)} examples {self.cfg.plancraft.num_generations} times."
        )
        for n in range(self.cfg.plancraft.num_generations):
            logger.info(f"Generation {n+1}/{self.cfg.plancraft.num_generations}")

            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                mode=self.cfg.wandb.mode,
                group=self.cfg.plancraft.model,
                job_type=self.cfg.plancraft.mode,
                config=self.cfg.model_dump(),
            )
            time_now = time.time()

            results_list = self.eval_all_examples(progress_bar=True)

            results_df = pd.DataFrame(results_list)
            results_df["model_name"] = self.cfg.plancraft.model
            results_df["mode"] = self.cfg.plancraft.mode

            time_elapsed = time.time() - time_now
            logger.info(f"Time elapsed: {time_elapsed:.2f}s")

            table = wandb.Table(dataframe=results_df)
            wandb.log({"results": table})
            wandb.finish()

            self.generation_number += 1

        logger.info("Done")
