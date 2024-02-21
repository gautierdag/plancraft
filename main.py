import json
import logging
import os
import warnings

import hydra
import torch
from minedojo import MineDojoEnv

from models.utils import preprocess_obs, save_frames_to_video
from dep import DEP

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


with open("data/task_info.json", "r") as f:
    TASK_INFO = json.load(f)


class Evaluator:
    def __init__(self, cfg, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        self.env = MineDojoEnv(
            name=cfg["eval"]["env_name"],
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

        self.model = DEP(cfg, self.env)
        self.no_op = self.env.action_space.no_op()

    def reset(self, task_name: str, iteration: int = 0):
        # check that task directory exists
        os.makedirs(f"{self.output_dir}/{task_name}", exist_ok=True)

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
        obs = preprocess_obs(obs)
        self.frames = [(obs["rgb"], "start")]
        success = False

        for t in range(0, self.task["episode"]):
            action, goal_name = self.model.step(obs, info)
            obs, _, _, info = self.env.step(action)
            obs = preprocess_obs(obs)

            # append the video frames
            self.frames.append((obs["rgb"], goal_name))

            # check if the task is done?
            if self.check_done(info["inventory"], self.task["object"]):
                success = True
                logger.info(f"{t}: finish goal {goal_name}.")
                break

        # record the video
        if self.record_frames:
            logger.info("saving video")
            video_path = os.path.join(
                self.output_dir,
                f"{self.task_name}/{self.task_name}_{self.iteration}.gif",
            )
            save_frames_to_video(self.frames, video_path)
            self.frames = []

        self.model.save_logs(self.output_dir, self.task_name, self.iteration)

        return success, t  # True or False, episode length

    def evaluate_task(self, task_name: str):
        num_evals = self.cfg["eval"]["num_evals"]
        success_rate = 0
        episode_lengths = []
        for i in range(num_evals):
            try:
                self.reset(task_name, iteration=i)
                succ_flag, min_episode = self.eval_step()
            except Exception as e:
                logger.error(e)
                succ_flag = False
                min_episode = 0
            # raise e
            success_rate += succ_flag
            if succ_flag:
                episode_lengths.append(min_episode)
            logger.info(
                f"Task {self.task_name} | Iteration {i} | Successful {succ_flag} | Episode length {min_episode} | Success rate {success_rate/(i+1)}"
            )
        logger.info(f"success rate: {success_rate / num_evals}")
        logger.info(
            f"average episode length: {sum(episode_lengths) / (len(episode_lengths) + 0.01)}",
        )
        # save the success rate and episode length
        with open(f"{self.output_dir}/{task_name}/{task_name}_results.json", "w") as f:
            json.dump(
                {
                    "success_count": success_rate,
                    "success_rate": success_rate / num_evals,
                    "num_evals": num_evals,
                    "episode_lengths": episode_lengths,
                },
                f,
            )

    def evaluate(self, task_names: list):
        if len(task_names) == 0:
            logger.info("Evaluating all tasks")
            task_names = list(TASK_INFO.keys())
        for task_name in task_names:
            self.evaluate_task(task_name)


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    logger.info(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    evaluator = Evaluator(cfg, output_dir=output_dir)
    evaluator.evaluate(cfg["eval"]["tasks"])


if __name__ == "__main__":
    main()
