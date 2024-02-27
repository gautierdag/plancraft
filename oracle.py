import json
import logging
import math

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
        goal = self.tech_tree[target]
        goal["quantity_needed"] = num_needed
        goal["depth"] = 0
        tree = {target: goal}

        def travel_tech_tree(current: str, quantity_needed: int, depth=1):
            """
            Recursive function to travel the tech tree
            """
            # check if the current node has any preconditions
            requirements = (
                self.tech_tree[current]["precondition"]
                | self.tech_tree[current]["tool"]
            )
            # cost to produce the current node
            quantity_to_produce = self.tech_tree[current]["output"][current]
            for r in requirements:
                cost_to_produce = requirements[r]
                if quantity_to_produce < quantity_needed:
                    cost_to_produce = math.ceil(
                        cost_to_produce * (quantity_needed / quantity_to_produce)
                    )
                # node already exists
                if r in tree:
                    tree[r]["quantity_needed"] += cost_to_produce
                    tree[r]["depth"] = max(tree[r]["depth"], depth)
                    travel_tech_tree(r, cost_to_produce, depth=depth + 1)

                # new tech
                else:
                    tree[r] = self.tech_tree[r]
                    tree[r]["quantity_needed"] = cost_to_produce
                    tree[r]["depth"] = depth
                    travel_tech_tree(r, cost_to_produce, depth=depth + 1)

        travel_tech_tree(target, num_needed)

        # sort by depth
        plan = sorted(tree.values(), key=lambda x: x["depth"], reverse=True)
        return plan

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
        # keep goal until the number of items acquired is >= to quantity_needed
        if num_items_acquired >= self.curr_goal["quantity_needed"]:
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
            action = self.crafter.get_action(
                self.curr_goal["tool"], curr_goal_type, goal
            )
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
