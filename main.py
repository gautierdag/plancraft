import json
import logging
import os
import warnings

import hydra
import torch
from minedojo import MineDojoEnv
from pyvirtualdisplay import Display

from models.utils import preprocess_obs, save_frames_to_video
from dep import DEP
from oracle import OraclePlanner

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


with open("data/task_info.json", "r") as f:
    TASK_INFO = json.load(f)


class Evaluator:
    def __init__(self, cfg, output_dir: str, biome: str = "plains"):
        self.cfg = cfg
        self.output_dir = output_dir
        self.env = MineDojoEnv(
            name=biome,
            img_size=(
                cfg["simulator"]["resolution"][0],
                cfg["simulator"]["resolution"][1],
            ),
            rgb_only=False,
            seed=cfg["eval"]["seed"],
            world_seed=cfg["eval"]["seed"],
            fast_reset=True,
            force_slow_reset_interval=cfg["eval"]["num_evals"],
        )

        self.record_frames = cfg["record"]["frames"]
        self.frames = []

        if cfg["eval"]["model"] == "DEP":
            self.model = DEP(cfg, self.env)
        elif cfg["eval"]["model"] == "oracle":
            self.model = OraclePlanner(cfg, self.env)
        else:
            raise ValueError(f"Model {cfg['eval']['model']} not supported")

        self.no_op = self.env.action_space.no_op()

    def reset(self, task_name: str, iteration: int = 0):
        # check that task directory exists
        logger.info(f"resetting the task {task_name}")
        self.task_name = task_name
        self.task = TASK_INFO[task_name]
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

        success = False
        for t in range(0, self.task["episode"]):
            try:
                action, goal_name = self.model.step(obs, info)
            except Exception as e:
                logger.error(e)
                raise e
                # break

            if t % 20 == 0:
                logger.info(f"{t}: current goal {goal_name}")

            obs, _, _, info = self.env.step(action)
            # append the video frames
            self.frames.append((obs["rgb"], goal_name))

            obs = preprocess_obs(obs)

            # check if the task is done?
            if self.check_done(info["inventory"], self.task["object"]):
                success = True
                logger.info(f"{t}: completed task: {self.task_name}")
                break

        # record the video
        if self.record_frames:
            logger.info("saving video")
            video_path = os.path.join(
                self.output_dir,
                f"{self.task_name}_{self.iteration}.gif",
            )
            save_frames_to_video(self.frames, video_path)
            self.frames = []

        self.model.save_logs(self.output_dir, self.task_name, self.iteration)

        return success, t  # True or False, episode length

    def close(self):
        self.env.close()


def evaluate_task(task_name: str, cfg: dict, output_dir: str):
    num_evals = cfg["eval"]["num_evals"]  # number of evals in each biome for each task
    success_rate = 0
    episode_lengths = []
    results = []

    biomes = ["plains", "forest"]
    for biome in biomes:
        biome_output_dir = f"{output_dir}/{task_name}/{biome}"
        os.makedirs(biome_output_dir, exist_ok=True)
        evaluator = Evaluator(cfg, output_dir=biome_output_dir, biome=biome)
        for i in range(num_evals):
            evaluator.reset(task_name, iteration=i)
            succ_flag, min_episode = evaluator.eval_step()
            success_rate += succ_flag
            result = {
                "biome": biome,
                "task_name": task_name,
                "iteration": i,
                "success": succ_flag,
            }
            if succ_flag:
                result["episode_length"] = min_episode

            logger.info(
                f"Biome {biome} | Task {task_name} | Iteration {i} | Successful {succ_flag} | Episode length {min_episode} | Success rate {success_rate/(i+1)}"
            )
            results.append(result)
        logger.info(f"success rate: {success_rate / num_evals}")
        logger.info(
            f"average episode length: {sum(episode_lengths) / (len(episode_lengths) + 0.01)}",
        )
        evaluator.close()

    # save the results
    with open(f"{output_dir}/{task_name}/{task_name}_results.json", "w") as f:
        json.dump(results, f)

    # save the overall success rate and episode length
    with open(f"{output_dir}/{task_name}/{task_name}_overall_results.json", "w") as f:
        json.dump(
            {
                "success_count": success_rate,
                "success_rate": success_rate / (num_evals * len(biomes)),
                "num_evals": num_evals * len(biomes),
            },
            f,
        )


def evaluate(cfg: dict, output_dir: str) -> None:
    task_names = cfg["eval"]["tasks"]
    if len(task_names) == 0:
        logger.info("Evaluating all tasks")
        task_names = list(TASK_INFO.keys())
    for task_name in task_names:
        evaluate_task(task_name, cfg, output_dir)


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    logger.info(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    with Display(size=(480, 640), visible=False) as disp:
        # display is active
        logger.info(f"display is alive: {disp.is_alive()}")
        evaluate(cfg, output_dir)


if __name__ == "__main__":
    main()
