import json
import logging
import os
import warnings

import hydra
import torch
from minedojo import MineDojoEnv
from PIL import Image

from models import CraftAgent, MineAgent
from models.utils import preprocess_obs, resize_image
from planner import Planner

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


prefix = os.getcwd()
task_info_json = os.path.join(prefix, "data/task_info.json")
goal_mapping_json = os.path.join(prefix, "data/goal_mapping.json")

TASK_LIST = []
with open(task_info_json, "r") as f:
    task_info = json.load(f)
TASK_LIST = list(task_info.keys())

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
        )
        self.task_list = TASK_LIST

        self.goal_mapping_cfg = self.load_goal_mapping_config()
        self.mineclip_prompt_dict = self.goal_mapping_cfg["mineclip"]
        self.clip_prompt_dict = self.goal_mapping_cfg[
            "clip"
        ]  # unify the mineclip and clip
        self.goal_mapping_dict = self.goal_mapping_cfg["horizon"]
        self.goal_model_freq = cfg["goal_model"]["freq"]
        self.goal_list_size = cfg["goal_model"]["queue_size"]
        self.record_frames = cfg["record"]["frames"]

        self.no_op = self.env.action_space.no_op()
        self.miner = MineAgent(cfg, no_op=self.no_op)
        self.crafter = CraftAgent(self.env, no_op=self.no_op)
        self.planner = Planner()

        task = cfg["eval"]["task_name"]
        self.reset(task)

    def reset(self, task):
        logger.info(f"[INFO]: resetting the task {task}")
        self.planner.reset()
        self.task = task
        (
            self.task_obj,
            self.max_ep_len,
            self.task_question,
            self.task_group,
        ) = self.load_task_info(self.task)
        plan = self.planner.initial_planning(self.task_group, self.task_question)
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
        self.replan_rounds = 0
        self.logs = {}

    # @TODO remove since it's really redundant to reread the json file
    def load_task_info(self, task):
        with open(task_info_json, "r") as f:
            task_info = json.load(f)
        target_item = task_info[task]["object"]
        episode_length = int(task_info[task]["episode"])
        task_question = task_info[task]["question"]
        task_group = task_info[task]["group"]
        return target_item, episode_length, task_question, task_group

    def load_goal_mapping_config(self):
        with open(goal_mapping_json, "r") as f:
            goal_mapping_cfg = json.load(f)
        return goal_mapping_cfg

    # check if the inventory has the object items
    def check_inventory(
        self, inventory, items: dict
    ):  # items: {"planks": 4, "stick": 2}
        for key in items.keys():  # check every object item
            # item_flag = False
            if (
                sum([item["quantity"] for item in inventory if item["name"] == key])
                < items[key]
            ):
                return False
        return True

    def check_precondition(self, inventory, precondition: dict):
        for key in precondition.keys():  # check every object item
            # item_flag = False
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
        # while self.check_inventory(inventory, self.curr_goal["object"]):
        if (
            self.check_inventory(inventory, self.curr_goal["object"])
            and self.goal_eps > 1
        ):
            logger.info(f"[INFO]: finish goal {self.curr_goal['name']}.")
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

        logger.info(f"Evaluating the task is {self.task}")

        if self.record_frames:
            video_frames = [obs["rgb"]]
            goal_frames = ["start"]

        def stack_obs(prev_obs: dict, obs: dict):
            stacked_obs = {}
            stacked_obs["rgb"] = torch.cat([prev_obs["rgb"], obs["rgb"]], dim=0)
            stacked_obs["voxels"] = torch.cat(
                [prev_obs["voxels"], obs["voxels"]], dim=0
            )
            stacked_obs["compass"] = torch.cat(
                [prev_obs["compass"], obs["compass"]], dim=0
            )
            stacked_obs["gps"] = torch.cat([prev_obs["gps"], obs["gps"]], dim=0)
            stacked_obs["biome"] = torch.cat([prev_obs["biome"], obs["biome"]], dim=0)
            return stacked_obs

        def slice_obs(obs: dict, slice: torch.tensor):
            res = {}
            for k, v in obs.items():
                res[k] = v[slice]
            return res

        obs = preprocess_obs(obs)

        states = obs
        actions = torch.zeros(1, self.miner.model.action_dim)

        curr_goal = None
        prev_goal = None
        seek_point = 0

        obs, reward, env_done, info = self.env.step(self.no_op.copy())

        # max_ep_len = task_eps[self.task]
        for t in range(0, self.max_ep_len):
            self.update_goal(info["inventory"])
            curr_goal = self.curr_goal

            if not prev_goal == curr_goal:
                logger.info(f"[INFO]: Episode Step {t}, Current Goal {curr_goal}")
                seek_point = t
                actions = torch.zeros(actions.shape[0], self.miner.model.action_dim)

            prev_goal = curr_goal

            # take the current goal type
            curr_goal_type = self.curr_goal["type"]

            sf = 5  # skip frame
            wl = 10  # window len
            end = actions.shape[0] - 1
            rg = torch.arange(
                end, min(max(end - sf * (wl - 1) - 1, seek_point - 1), end - 1), -sf
            ).flip(0)

            # DONE: change the craft agent into craft actions
            goal = list(self.curr_goal["object"].keys())[0]
            logger.info(f"[INFO]: goal is {goal}")
            if curr_goal_type in ["craft", "smelt"]:
                logger.info("[INFO]: craft or smelt")
                preconditions = self.curr_goal["precondition"].keys()
                action = self.crafter.get_action(preconditions, curr_goal_type, goal)
            elif curr_goal_type == "mine":
                logger.info("[INFO]: mine")
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

            logger.info(f"[INFO]: Taking action: {action}")
            obs, reward, env_done, info = self.env.step(action)

            if self.record_frames:
                video_frames.append(obs["rgb"])
                goal_frames.append(curr_goal["name"])
            obs = preprocess_obs(obs)

            if not torch.is_tensor(action):
                action = torch.from_numpy(action)

            states = stack_obs(states, obs)
            actions = torch.cat([actions, action.unsqueeze(0)], dim=0)

            self.goal_eps += 1
            if curr_goal_type == "mine" and not self.check_precondition(
                info["inventory"], self.curr_goal["precondition"]
            ):
                self.replan_task(info["inventory"], self.task_question)
            elif curr_goal_type == "craft" and self.goal_eps > 150:
                self.replan_task(info["inventory"], self.task_question)
            elif curr_goal_type == "smelt" and self.goal_eps > 200:
                self.replan_task(info["inventory"], self.task_question)

            if self.replan_rounds > 12:
                logger.info("[INFO]: replanning over rounds")
                break

            if self.check_done(
                info["inventory"], self.task_obj
            ):  # check if the task is done?
                env_done = True
                logger.info(f"[INFO]: finish goal {self.curr_goal['name']}.")
                self.planner.generate_success_description(self.curr_goal["ranking"])
                self.logs[t] = {}
                self.logs[t]["curr_plan"] = self.goal_list
                self.logs[t]["curr_goal"] = self.curr_goal
                self.logs[t]["curr_dialogue"] = self.planner.logging_dialogue
                self.logs[t]["result"] = True
                break

        # record the video
        if env_done and self.record_frames:
            # if self.record_frames:
            logger.info("[INFO]: saving the frames")
            imgs = []
            for id, frame in enumerate(video_frames):
                frame = resize_image(frame, (320, 240)).astype("uint8")
                # cv2.putText(
                #     frame,
                #     f"FID: {id}",
                #     (10, 25),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     (255, 255, 255),
                #     2,
                # )
                # cv2.putText(
                #     frame,
                #     f"Goal: {goal_frames[id]}",
                #     (10, 55),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     (255, 0, 0),
                #     2,
                # )
                imgs.append(Image.fromarray(frame))
            imgs = imgs[::3]
            logger.info(f"record imgs length: {len(imgs)}")
            # @ todo move the recording/saving to utils
            # now = datetime.now()
            # timestamp = (
            #     f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
            # )
            # folder_name = os.path.join(prefix, "recordings/" + timestamp + "/")
            # if not os.path.exists(folder_name):
            #     os.mkdir(folder_name)
            # imgs[0].save(
            #     folder_name + self.task + ".gif",
            #     save_all=True,
            #     append_images=imgs[1:],
            #     optimize=False,
            #     quality=0,
            #     duration=150,
            #     loop=0,
            # )
            # with open(folder_name + self.task + ".json", "w") as f:
            #     json.dump(self.logs, f, indent=4)

        return env_done, t  # True or False, episode length

    def single_task_evaluate(self):
        loops = self.cfg["eval"]["goal_ratio"]
        succ_rate = 0
        episode_lengths = []
        for i in range(loops):
            try:
                self.reset(self.task)
                succ_flag, min_episode = self.eval_step()

            except Exception as e:
                logger.info(e)
                succ_flag = False
                min_episode = 0
                raise e
            succ_rate += succ_flag
            if succ_flag:
                episode_lengths.append(min_episode)
            logger.info(
                f"Task {self.task} | Iteration {i} | Successful {succ_flag} | Episode length {min_episode} | Success rate {succ_rate/(i+1)}"
            )
        logger.info("success rate: ", succ_rate / loops)
        logger.info(
            "average episode length:",
            sum(episode_lengths) / (len(episode_lengths) + 0.01),
        )


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    logger.info(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    evaluator = Evaluator(cfg, output_dir=output_dir)
    evaluator.single_task_evaluate()


if __name__ == "__main__":
    main()
