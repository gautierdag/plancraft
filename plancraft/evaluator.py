import os
import json
import logging
import warnings

from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.env_symbolic import SymbolicPlancraft
from plancraft.config import Config

import torch

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        cfg: Config,
        output_dir: str,
    ):
        self.cfg = cfg

        if cfg.plancraft.environment.symbolic:
            self.env = SymbolicPlancraft()
        else:
            self.env = RealPlancraft(
                symbolic_observation_space=cfg.plancraft.environment.symbolic_observation_space,
                symbolic_action_space=cfg.plancraft.environment.symbolic_action_space,
            )

        self.output_dir = output_dir
        self.observations = []

    def reset(self, task_name: str, iteration: int = 0):
        # check that task directory exists
        logger.info(f"resetting the task {task_name}")
        self.task_name = task_name
        # self.task = TASK_INFO[task_name]
        self.iteration = iteration
        self.model.reset(self.task["question"])

    def check_done(self, inventory, task_obj: str):
        for item in inventory:
            if task_obj == item["name"]:
                return True
        return False

    @torch.no_grad()
    def eval_step(self):
        obs = self.env.reset()
        logger.info(f"Evaluating the task is {self.task_name}")
        logger.info(f"Start position: {obs['compass']} {obs['gps']}")
        obs, _, _, info = self.env.step(self.no_op.copy())
        self.frames = [(obs["rgb"], "start")]
        # obs = preprocess_obs(obs)

    def eval_task(self, task_name: str, iteration: int = 0):
        self.reset(task_name, iteration)
        self.eval_step()
        self.observations.append(self.env.get_observation())
        self.env.close()

    def evaluate(self):
        for task in self.cfg.plancraft.tasks:
            self.eval_task(task)
        with open(os.path.join(self.output_dir, "observations.json"), "w") as f:
            json.dump(self.observations, f, indent=4)
