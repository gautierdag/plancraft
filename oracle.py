import json
import logging

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

        # unify the mineclip and clip
        self.clip_prompt_dict = self.goal_mapping_cfg["clip"]
        self.goal_mapping_dict = self.goal_mapping_cfg["horizon"]

        no_op = env.action_space.no_op()
        self.miner = MineAgent(cfg, no_op=no_op)
        self.crafter = CraftAgent(env, no_op=no_op)

        with open("data/goal_lib.json", "r") as f:
            goal_lib = json.load(f)

        self.tech_tree = {}
        for g in goal_lib:
            k = g.replace("smelt_", "").replace("craft_", "").replace("mine_", "")
            self.tech_tree[k] = goal_lib[g]
            self.tech_tree[k]["name"] = k

    def load_goal_mapping_config(self):
        with open(goal_mapping_json, "r") as f:
            goal_mapping_cfg = json.load(f)
        return goal_mapping_cfg

    def get_plan(self, target, num_needed=1) -> list[dict]:
        plan = []
        if len(self.tech_tree[target]["precondition"]) > 0:
            for p in self.tech_tree[target]["precondition"]:
                plan += self.get_plan(p, self.tech_tree[target]["precondition"][p])
        goal = self.tech_tree[target]
        # number of resources needed
        goal["need"] = num_needed
        return plan + [goal]

    def reset(self, task_question: str):
        self.task_question = task_question

        # parse question
        target = self.task_question.split()[-1].replace("?", "")
        self.plan = self.get_plan(target)
        self.curr_goal = self.plan[0]

        self.seek_point = 0
        self.goal_history = []
        self.prev_goal = None
        self.actions = torch.zeros(1, self.miner.model.action_dim)
        self.states = None

    def update_goal(self, inventory):
        item_name = self.curr_goal["name"]
        num_items_acquired = sum(
            [item["quantity"] for item in inventory if item["name"] == item_name]
        )
        # keep goal until the number of items acquired is >= to needed
        if num_items_acquired >= self.curr_goal["need"]:
            logger.info(f"finish goal {self.curr_goal['name']}.")
            self.plan.remove(self.plan[0])
            self.curr_goal = self.plan[0]

    def step(self, obs, info, t=0):
        if self.states is None:
            self.states = obs
        else:
            self.states = stack_obs(self.states, obs)

        self.update_goal(info["inventory"])
        curr_goal = self.curr_goal

        self.goal_history.append(curr_goal)

        if self.prev_goal != curr_goal:
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
        goal = list(self.curr_goal["output"].keys())[0]

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
