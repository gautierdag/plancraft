import functools
import json
import logging
import time

from openai import OpenAI
import os
import torch
from models import CraftAgent, MineAgent
from models.utils import slice_obs, stack_obs

prefix = os.getcwd()
goal_mapping_json = os.path.join(prefix, "data/goal_mapping.json")

logger = logging.getLogger(__name__)


class OraclePlanner:
    def __init__(self, cfg, env):
        self.goal_mapping_cfg = self.load_goal_mapping_config()
        self.mineclip_prompt_dict = self.goal_mapping_cfg["mineclip"]

        # unify the mineclip and clip
        self.clip_prompt_dict = self.goal_mapping_cfg["clip"]
        self.goal_mapping_dict = self.goal_mapping_cfg["horizon"]

        no_op = env.action_space.no_op()
        self.miner = MineAgent(cfg, no_op=no_op)
        self.crafter = CraftAgent(env, no_op=no_op)

    def reset(self, task_question: str):
        self.task_question = task_question

        # parse question

        # # NOTE: if the goal list is empty, then the DEP default goal is to mine a log
        # if len(self.goal_list) == 0:
        #     self.curr_goal = {
        #         "name": "mine_log",
        #         "type": "mine",
        #         "object": {"log": 1},
        #         "precondition": {},
        #         "ranking": 1,
        #     }

        self.seek_point = 0
        self.goal_history = []
        self.prev_goal = None
        self.actions = torch.zeros(1, self.miner.model.action_dim)
        self.states = None

    def step(self, obs, info, t=0):
        if self.states is None:
            self.states = obs
        else:
            self.states = stack_obs(self.states, obs)

        self.update_goal(info["inventory"])
        curr_goal = self.curr_goal

        self.goal_history.append(curr_goal)

        if not self.prev_goal == curr_goal:
            logger.info(f"Episode Step {t}, Current Goal {curr_goal}")
            self.seek_point = t
            self.actions = torch.zeros(
                self.actions.shape[0], self.miner.model.action_dim
            )

        self.prev_goal = curr_goal

        

        # get the rolling window of the actions
        sf = 5  # skip frame
        wl = 10  # window len
        end = self.actions.shape[0] - 1
        rg = torch.arange(
            end, min(max(end - sf * (wl - 1) - 1, self.seek_point - 1), end - 1), -sf
        ).flip(0)

        curr_goal_type = self.curr_goal["type"]
        goal = list(self.curr_goal["object"].keys())[0]

        if curr_goal_type in ["craft", "smelt"]:
            preconditions = self.curr_goal["precondition"].keys()
            action = self.crafter.get_action(preconditions, curr_goal_type, goal)
        elif curr_goal_type == "mine":
            clip_goal = self.goal_mapping_dict[goal]
            complete_states = slice_obs(self.states, rg)
            complete_states["prev_action"] = self.actions[rg]
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

        # convert the action to numpy
        if torch.is_tensor(action):
            action = action.cpu().numpy()

        # update actions history
        self.actions = torch.cat(
            [self.actions, torch.from_numpy(action).unsqueeze(0)], dim=0
        )

        return action, curr_goal["name"]

    def save_goal_history(self, filename: str):
        with open(filename, "w") as f:
            for item in self.goal_history:
                f.write(json.dumps(item) + "\n")

    def save_logs(self, output_dir: str, task_name: str, iteration: int):
        self.save_goal_history(
            f"{output_dir}/{task_name}/{task_name}_{iteration}_goal_history.jsonl"
        )
