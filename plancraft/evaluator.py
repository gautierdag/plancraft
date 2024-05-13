import json
import logging

import pandas as pd
import torch

import wandb
from plancraft.config import Config, PlancraftExample
from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.env_symbolic import SymbolicPlancraft
from plancraft.models import get_model

logger = logging.getLogger(__name__)


class Evaluator:
    """
    The evaluator class handles the environment loop and model interaction

    The environment can
    """

    def __init__(self, cfg: Config, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir

        self.examples = self.load_dataset(cfg.plancraft.split)
        self.example_idx = 0

        if cfg.plancraft.environment.symbolic:
            self.env = SymbolicPlancraft(
                inventory=self.examples[self.example_idx].slotted_inventory
            )
        else:
            self.env = RealPlancraft(
                inventory=self.examples[self.example_idx].slotted_inventory,
                symbolic_action_space=cfg.plancraft.environment.symbolic_action_space,
                symbolic_observation_space=cfg.plancraft.environment.symbolic_observation_space,
                preferred_spawn_biome=cfg.plancraft.environment.preferred_spawn_biome,
                resolution=cfg.plancraft.environment.resolution,
            )

        self.record_frames = not (cfg.plancraft.environment.symbolic)
        self.model = get_model(cfg)

        # no_op action
        self.no_op = self.env.action_space.no_op()

    def load_dataset(self, dataset_split: str) -> list[PlancraftExample]:
        with open(f"data/{dataset_split}.json", "r") as f:
            dataset = json.load(f)
            return [PlancraftExample(**example) for example in dataset]

    def reset(self, example_idx: int = 0):
        self.example_idx = example_idx
        current_inventory = self.examples[example_idx].slotted_inventory
        self.env.fast_reset(new_inventory=current_inventory)
        self.model.reset()

    def check_done(self, inventory: list[dict[str, int]], target: str):
        for item in inventory:
            if target == item["type"]:
                return True
        return False

    @torch.no_grad()
    def eval_example(self, example_idx) -> dict:
        self.reset(example_idx)

        target = self.examples[example_idx].target
        target_question = (
            f"Combine the items in the inventory to obtain an item of type {target}"
        )

        # set global objective/target in model
        self.model.set_objective(target_question)

        observations = []

        obs, _, _, info = self.env.step(self.no_op.copy())
        observations.append(obs)
        done = self.check_done(obs["inventory"], target)
        step = 0

        while step < self.cfg.plancraft.max_steps and not done:
            action = self.model.step(obs)
            obs, _, done, _ = self.env.step(action)
            done = self.check_done(obs["inventory"], target)
            observations.append(obs)
            step += 1

        return {
            "success": done,
            "number_of_steps": step,
            "model_trace": self.model.trace,
            "observations": observations,
        }

    def eval_all(self):
        logger.info(
            f"Running evaluation over {len(self.examples)} examples {self.cfg.plancraft.num_generations} times."
        )
        for n in range(self.cfg.plancraft.num_generations):
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                mode=self.cfg.wandb.mode,
                group=self.cfg.plancraft.model,
                job_type=self.cfg.plancraft.mode,
                config=self.cfg.model_dump(),
            )
            results = []
            for example_idx in range(len(self.examples)):
                result = self.eval_example(example_idx)
                results.append(result)

            results_df = pd.DataFrame(results)
            results_df["model_name"] = self.cfg.plancraft.model
            results_df["mode"] = self.cfg.plancraft.mode

            table = wandb.Table(dataframe=results_df)
            wandb.log({"results": table})
            wandb.finish()

        logger.info("Done")
