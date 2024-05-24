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
        self.histories[env_idx].reset(objective=objective)

    def check_done(self, inventory: list[dict[str, int]], target: str):
        for item in inventory:
            if target == item["type"]:
                return True
        return False

    # @torch.no_grad()
    # def eval_examples(self, examples: list[int]):
    #     # assign examples to environments

    #     # reset the environment
    #     self.reset(env_id, example_idx)
    #     # reset the model
    #     model = self.models[env_id]
    #     model.reset()

    #     # get the environment
    #     env = self.envs[env_id]

    #     # TODO: implement flow for impossible examples
    #     if self.examples[example_idx].impossible:
    #         return {}

    #     target = self.examples[example_idx].target
    #     target_question = f"Craft an item of type: {target}"

    #     # set global objective/target in model
    #     model.set_objective(target_question)
    #     observations = []

    #     obs, _, _, info = env.step(self.no_op.copy())
    #     observations.append(obs)
    #     done = self.check_done(obs["inventory"], target)
    #     step = 0

    #     while step < self.cfg.plancraft.max_steps and not done:
    #         action = model.step(obs)
    #         obs, _, done, _ = env.step(action)
    #         done = self.check_done(obs["inventory"], target)
    #         if done:
    #             logger.info(f"Done in {step} steps")
    #         observations.append(obs)
    #         step += 1

    #     return {
    #         "success": done,
    #         "number_of_steps": step,
    #         "model_trace": model.trace,
    #         "observations": observations,
    #     }

    @torch.no_grad()
    def eval_batch(self, examples: list[int]):
        results = []

        while len(examples) > 0:
            # check if done with any of the examples

            # assign examples to environments


        # reset the environment
        # self.reset(env_id, example_idx)
        # reset the model
        model = self.models[env_id]
        model.reset()

        # get the environment
        env = self.envs[env_id]

        # TODO: implement flow for impossible examples
        if self.examples[example_idx].impossible:
            return {}

        target = self.examples[example_idx].target
        target_question = f"Craft an item of type: {target}"

        # set global objective/target in model
        model.set_objective(target_question)
        observations = []

        obs, _, _, info = env.step(self.no_op.copy())
        observations.append(obs)
        done = self.check_done(obs["inventory"], target)
        step = 0

        while step < self.cfg.plancraft.max_steps and not done:
            action = model.step(obs)
            obs, _, done, _ = env.step(action)
            done = self.check_done(obs["inventory"], target)
            if done:
                logger.info(f"Done in {step} steps")
            observations.append(obs)
            step += 1

        return {
            "success": done,
            "number_of_steps": step,
            "model_trace": model.trace,
            "observations": observations,
        }

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

           

            # results_list = []
            # while not results.empty():
            # results_list.append(results.get())
            # logger.info(f"Number of results: {len(results_list)}")
            # results_df = pd.DataFrame(results_list)
            # results_df["model_name"] = self.cfg.plancraft.model
            # results_df["mode"] = self.cfg.plancraft.mode

            # table = wandb.Table(dataframe=results_df)
            # wandb.log({"results": table})
            # wandb.finish()

        logger.info("Done")
