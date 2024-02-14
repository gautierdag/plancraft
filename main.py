import json
import logging
import os
import warnings

import hydra
import torch
from minedojo import MineDojoEnv

from models import CraftAgent, MineAgent
from models.utils import preprocess_obs, save_frames_to_video, slice_obs, stack_obs
from planner import Planner

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


prefix = os.getcwd()
goal_mapping_json = os.path.join(prefix, "data/goal_mapping.json")

with open("data/task_info.json", "r") as f:
    TASK_INFO = json.load(f)

prefix = os.getcwd()


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

        self.goal_mapping_cfg = self.load_goal_mapping_config()
        self.mineclip_prompt_dict = self.goal_mapping_cfg["mineclip"]

        # unify the mineclip and clip
        self.clip_prompt_dict = self.goal_mapping_cfg["clip"]
        self.goal_mapping_dict = self.goal_mapping_cfg["horizon"]

        self.record_frames = cfg["record"]["frames"]
        self.frames = []

        self.no_op = self.env.action_space.no_op()
        self.miner = MineAgent(cfg, no_op=self.no_op)
        self.crafter = CraftAgent(self.env, no_op=self.no_op)
        self.planner = Planner()

    def reset(self, task_name: str, iteration: int = 0):
        # check that task directory exists
        os.makedirs(f"{self.output_dir}/{task_name}", exist_ok=True)

        logger.info(f"resetting the task {task_name}")
        self.task_name = task_name
        self.task = TASK_INFO[task_name]
        self.iteration = iteration
        self.goal_history = []

        self.planner.reset()
        plan = self.planner.initial_planning(self.task["group"], self.task["question"])
        self.goal_list = self.planner.generate_goal_list(plan)
        # NOTE: if the goal list is empty, then the DEPS default goal is to mine a log
        if len(self.goal_list) == 0:
            self.curr_goal = {
                "name": "mine_log",
                "type": "mine",
                "object": {"log": 1},
                "precondition": {},
                "ranking": 1,
            }
        else:
            self.curr_goal = self.goal_list[0]
        self.goal_eps = 0
        self.replan_rounds = 0

    def load_goal_mapping_config(self):
        with open(goal_mapping_json, "r") as f:
            goal_mapping_cfg = json.load(f)
        return goal_mapping_cfg

    # check if the inventory has the object items
    def check_inventory(
        self, inventory, items: dict
    ):  # items: {"planks": 4, "stick": 2}
        for key in items.keys():  # check every object item
            if (
                sum([item["quantity"] for item in inventory if item["name"] == key])
                < items[key]
            ):
                return False
        return True

    def check_precondition(self, inventory, precondition: dict):
        for key in precondition.keys():  # check every object item
            if (
                sum([item["quantity"] for item in inventory if item["name"] == key])
                < precondition[key]
            ):
                return False
        return True

    def check_done(self, inventory, task_obj: str):
        for item in inventory:
            if task_obj == item["name"]:
                return True
        return False

    def update_goal(self, inventory):
        if (
            self.check_inventory(inventory, self.curr_goal["object"])
            and self.goal_eps > 1
        ):
            logger.info(f"finish goal {self.curr_goal['name']}.")
            self.planner.generate_success_description(self.curr_goal["ranking"])
            self.goal_list.remove(self.goal_list[0])
            self.curr_goal = self.goal_list[0]
            self.goal_eps = 0

    def replan_task(self, inventory, task_question):
        self.planner.generate_failure_description(self.curr_goal["ranking"])
        self.planner.generate_inventory_description(inventory)
        self.planner.generate_explanation()
        plan = self.planner.replan(task_question)

        self.goal_list = self.planner.generate_goal_list(plan)
        if len(self.goal_list) == 0:
            self.curr_goal = {
                "name": "mine_log",
                "type": "mine",
                "object": {"log": 1},
                "precondition": {},
                "ranking": 1,
            }
        else:
            self.curr_goal = self.goal_list[0]
        self.goal_eps = 0
        self.replan_rounds += 1

    @torch.no_grad()
    def eval_step(self):
        obs = self.env.reset()

        logger.info(f"Evaluating the task is {self.task_name}")
        # log start position
        logger.info(f"Start position: {obs['compass']} {obs['gps']}")

        self.frames = [(obs["rgb"], "start")]

        obs = preprocess_obs(obs)
        states = obs
        actions = torch.zeros(1, self.miner.model.action_dim)
        curr_goal = None
        prev_goal = None

        seek_point = 0
        obs, reward, env_done, info = self.env.step(self.no_op.copy())

        for t in range(0, self.task["episode"]):
            self.update_goal(info["inventory"])
            curr_goal = self.curr_goal
            self.goal_history.append(curr_goal)

            if t % 20 == 0:
                logger.info(f"Episode Step {t}, Current Goal {curr_goal['name']}")

            if not prev_goal == curr_goal:
                logger.info(f"Episode Step {t}, Current Goal {curr_goal}")
                seek_point = t
                actions = torch.zeros(actions.shape[0], self.miner.model.action_dim)

            prev_goal = curr_goal

            # take the current goal type
            curr_goal_type = self.curr_goal["type"]

            # get the rolling window of the actions
            sf = 5  # skip frame
            wl = 10  # window len
            end = actions.shape[0] - 1
            rg = torch.arange(
                end, min(max(end - sf * (wl - 1) - 1, seek_point - 1), end - 1), -sf
            ).flip(0)

            # DONE: change the craft agent into craft actions
            goal = list(self.curr_goal["object"].keys())[0]
            if curr_goal_type in ["craft", "smelt"]:
                preconditions = self.curr_goal["precondition"].keys()
                action = self.crafter.get_action(preconditions, curr_goal_type, goal)
            elif curr_goal_type == "mine":
                clip_goal = self.goal_mapping_dict[goal]
                complete_states = slice_obs(states, rg)
                complete_states["prev_action"] = actions[rg]
                action = self.miner.get_action(clip_goal, complete_states)
            else:
                logger.info("Undefined action type !!")

            # check if the inventory has the preconditions
            if len(self.curr_goal["precondition"].keys()):
                for cond in self.curr_goal["precondition"].keys():
                    if cond not in [
                        "wooden_pickaxe",
                        "stone_pickaxe",
                        "iron_pickaxe",
                        "diamond_pickaxe",
                        "wooden_axe",
                        "stone_axe",
                        "iron_axe",
                        "diamond_axe",
                    ]:
                        continue
                    if info["inventory"][0]["name"] != cond:
                        for item in info["inventory"]:
                            if (
                                item["name"] == cond
                                and item["quantity"] > 0
                                and item["index"] > 0
                            ):
                                act = self.no_op.copy()
                                act[5] = 5
                                act[7] = item["index"]
                                self.env.step(act)
                                break
            #! indent change
            if torch.is_tensor(action):
                action = action.cpu().numpy()

            obs, _, env_done, info = self.env.step(action)
            # append the video frames
            self.frames.append((obs["rgb"], curr_goal["name"]))

            obs = preprocess_obs(obs)

            if not torch.is_tensor(action):
                action = torch.from_numpy(action)

            states = stack_obs(states, obs)
            actions = torch.cat([actions, action.unsqueeze(0)], dim=0)

            self.goal_eps += 1
            if curr_goal_type == "mine" and not self.check_precondition(
                info["inventory"], self.curr_goal["precondition"]
            ):
                self.replan_task(info["inventory"], self.task["question"])
            elif curr_goal_type == "craft" and self.goal_eps > 150:
                self.replan_task(info["inventory"], self.task["question"])
            elif curr_goal_type == "smelt" and self.goal_eps > 200:
                self.replan_task(info["inventory"], self.task["question"])

            if self.replan_rounds > 12:
                logger.info(f"{t}: replanning over rounds")
                break

            if self.check_done(
                info["inventory"], self.task["object"]
            ):  # check if the task is done?
                env_done = True
                logger.info(f"{t}: finish goal {self.curr_goal['name']}.")
                self.planner.generate_success_description(self.curr_goal["ranking"])
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

        self.planner.save_dialogue(
            f"{self.output_dir}/{self.task_name}/{self.task_name}_{self.iteration}.txt"
        )

        # save goal history as jsonl
        with open(
            f"{self.output_dir}/{self.task_name}/{self.task_name}_{self.iteration}_goal_history.jsonl",
            "w",
        ) as f:
            for item in self.goal_history:
                f.write(json.dumps(item) + "\n")

        return env_done, t  # True or False, episode length

    def evaluate_task(self, task_name: str):
        num_evals = self.cfg["eval"]["num_evals"]
        success_rate = 0
        episode_lengths = []
        for i in range(num_evals):
            try:
                self.reset(task_name, iteration=i)
                succ_flag, min_episode = self.eval_step()
            except Exception as e:
                logger.info(e)
                succ_flag = False
                min_episode = 0
                raise e
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
