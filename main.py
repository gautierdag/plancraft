import json
import logging
import os
import warnings

import hydra
import torch

from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.env_symbolic import SymbolicPlancraft

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg, output_dir: str, mode):
        self.cfg = cfg
        self.output_dir = output_dir

        # self.env = MineDojoEnv(cfg)
        # self.env =

        self.record_frames = cfg["record"]["frames"]
        self.frames = []

        # if cfg["eval"]["model"] == "oracle":
        #     self.model = OraclePlanner(cfg, self.env)
        # else:
        #     raise ValueError(f"Model {cfg['eval']['model']} not supported")

        # self.no_op = self.env.action_space.no_op()

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
        obs = preprocess_obs(obs)


# def evaluate(cfg: dict, output_dir: str) -> None:
# evaluate_task(task_name, cfg, output_dir)


@hydra.main(config_path="configs", config_name="dep", version_base=None)
def main(cfg):
    logger.info(cfg)
    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # with Display(size=(480, 640), visible=False) as disp:
    # display is active
    # logger.info(f"display is alive: {disp.is_alive()}")
    # evaluate(cfg, output_dir)


if __name__ == "__main__":
    main()
