import json
import logging
import os
import warnings

import hydra
import torch

from plancraft.environments.env_real import RealPlancraft
from plancraft.environments.env_symbolic import SymbolicPlancraft
from plancraft.config import Config, PlancraftExample
from plancraft.baselines import ReactLLM, 

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: Config, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir

        self.examples = self.load_dataset(cfg.plancraft.split)
        self.iteration = 0

        if cfg.plancraft.environment.symbolic:
            self.env = SymbolicPlancraft(
                inventory=self.examples[self.iteration].slotted_inventory
            )
        else:
            self.env = RealPlancraft(
                inventory=self.examples[self.iteration].slotted_inventory,
                symbolic_action_space=cfg.plancraft.environment.symbolic_action_space,
                symbolic_observation_space=cfg.plancraft.environment.symbolic_observation_space,
                preferred_spawn_biome=cfg.plancraft.environment.preferred_spawn_biome,
                resolution=cfg.plancraft.environment.resolution,
            )

        self.record_frames = not (cfg.plancraft.environment.symbolic)
        self.frames = []
        # self.model =
        self.model =            
    # llm_model.reset()
    #     question = v["question"]
    #     target = question.split()[-1].replace("?", "")
    #     hash_key = f"react_{model_name}_{target}_{i}"

    #     step = 1
    #     model = ReactLLM(model=llm_model, guidance=cfg["guidance"])

        # no_op action
        self.no_op = self.env.action_space.no_op()

    def load_dataset(self, dataset_split: str) -> list[PlancraftExample]:
        with open(f"data/{dataset_split}.json", "r") as f:
            dataset = json.load(f)
            return [PlancraftExample(**example) for example in dataset]

    def reset(self, iteration: int = 0):
        self.iteration = iteration
        current_inventory = self.examples[iteration].slotted_inventory
        self.env.fast_reset(new_inventory=current_inventory)
        # self.model.reset()

    def check_done(self, inventory: list[dict[str, int]], target: str):
        for item in inventory:
            if target == item["name"]:
                return True
        return False

    @torch.no_grad()
    def eval_example(self):
        target = self.examples[self.iteration].target
        obs, _, _, info = self.env.step(self.no_op.copy())
        task_success = self.check_done(
            obs["inventory"],
        )
        if self.record_frames:
            self.frames = [(obs["rgb"], "start")]

        step = 0
        while step < self.cfg.plancraft.max_steps and not task_success:
            pass

    def eval_all(self):
        for iteration in range(len(self.examples)):
            for n in range(self.cfg.plancraft.num_generations):
                self.reset(iteration)
                self.eval_example()


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = Config(**dict(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    evaluator = Evaluator(cfg, output_dir)


if __name__ == "__main__":
    main()
