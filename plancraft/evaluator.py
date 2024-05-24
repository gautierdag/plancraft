import time
import json
import logging

import pandas as pd
import torch


import wandb
from plancraft.config import Config, PlancraftExample
from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.env_symbolic import SymbolicPlancraft
from plancraft.models import get_model
from plancraft.models.react import TransformersGenerator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment is created based on the configuration and the examples are loaded from the dataset.
    """

    def __init__(self, cfg: Config, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir

        self.examples = self.load_dataset(cfg.plancraft.split)

        self.llm = None
        if cfg.plancraft.mode not in ["dummy", "oracle"]:
            self.llm = TransformersGenerator(
                model_name=cfg.plancraft.model,
                quantize=cfg.plancraft.quantize,
            )

        self.batch_size = cfg.plancraft.batch_size
        self.envs = [self.create_env(cfg) for _ in range(self.batch_size)]
        self.model = get_model(cfg, self.llm)

        self.record_frames = not (cfg.plancraft.environment.symbolic)

        # no_op action
        if cfg.plancraft.environment.symbolic:
            self.no_op = {
                "inventory_command": (0, 0, 0),
            }
        else:
            self.no_op = self.envs[0].action_space.no_op()

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
        for item in inventory:
            if target == item["type"]:
                return True
        return False

    @torch.no_grad()
    def eval_all_examples(self) -> list:
        examples_queue = self.examples.copy()
        assigned_examples = {env_idx: None for env_idx in range(self.batch_size)}

        results = []

        actions = [self.no_op.copy() for _ in range(self.batch_size)]
        observations = [None for _ in range(self.batch_size)]
        done = [False for _ in range(self.batch_size)]

        while len(examples_queue) > 0:
            # assign example to environment if not already assigned
            for env_idx, example in assigned_examples.items():
                if example is None and len(examples_queue) > 0:
                    new_example = examples_queue.pop()
                    # TODO: implement flow for impossible examples
                    if new_example.impossible:
                        continue

                    assigned_examples[env_idx] = new_example
                    self.reset(new_example, env_idx)
                    actions[env_idx] = self.no_op.copy()
                    continue

                num_steps = self.model.histories[env_idx].num_steps
                if done[env_idx] or num_steps > self.cfg.plancraft.max_steps:
                    # save results and reset
                    results.append(
                        {
                            "success": done[env_idx],
                            "number_of_steps": num_steps,
                            "model_trace": self.model.histories[env_idx].trace(),
                            "example": example.model_dump(),
                        }
                    )
                    assigned_examples[env_idx] = None
                    done[env_idx] = False

            # step actions
            obs_batch = []
            obs_env_idx = []
            for env_idx, example in assigned_examples.items():
                if example is None:
                    continue

                obs, _, _, _ = self.envs[env_idx].step(actions[env_idx])
                observations[env_idx] = obs
                done[env_idx] = self.check_done(obs["inventory"], example.target)
                if not done[env_idx]:
                    obs_batch.append(obs)
                    obs_env_idx.append(env_idx)

            # get actions from model (batched)
            # only send observations for environments with examples
            pred_actions = self.model.step(obs_batch)
            for action, env_idx in zip(pred_actions, obs_env_idx):
                actions[env_idx] = action
                self.model.histories[env_idx].add_action_to_history(action)

        return results

    def eval_all(self):
        logger.info(
            f"Running evaluation over {len(self.examples)} examples {self.cfg.plancraft.num_generations} times."
        )
        for n in range(self.cfg.plancraft.num_generations):
            logger.info(f"Generation {n+1}/{self.cfg.plancraft.num_generations}")

            # wandb.init(
            #     project=self.cfg.wandb.project,
            #     entity=self.cfg.wandb.entity,
            #     mode=self.cfg.wandb.mode,
            #     group=self.cfg.plancraft.model,
            #     job_type=self.cfg.plancraft.mode,
            #     config=self.cfg.model_dump(),
            # )
            time_now = time.time()

            results_list = self.eval_all_examples()

            results_df = pd.DataFrame(results_list)
            results_df["model_name"] = self.cfg.plancraft.model
            results_df["mode"] = self.cfg.plancraft.mode

            time_elapsed = time.time() - time_now
            logger.info(f"Time elapsed: {time_elapsed:.2f}s")

            # table = wandb.Table(dataframe=results_df)
            # wandb.log({"results": table})
            # wandb.finish()

        logger.info("Done")
