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


def cache_results(func):
    cache_file = func.__name__ + ".cache.json"

    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        # Load the cache file if it exists, or initialize an empty cache
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except FileNotFoundError:
            cache = {}

        # The first argument is assumed to be 'self', and the second one is 'prompt_text'
        prompt_text = args[1]

        # Check if the result is cached
        if prompt_text in cache:
            logger.info("Returning cached result")
            return cache[prompt_text]

        # Call the original function and cache its result
        result = func(*args, **kwargs)
        cache[prompt_text] = result

        # Save the updated cache
        with open(cache_file, "w") as f:
            json.dump(cache, f)

        return result

    return wrapper_cache


class Planner:
    def __init__(
        self,
        goal_lib_json="data/goal_lib.json",
        task_prompt_file="data/task_prompt.txt",
        replan_prompt_file="data/deps_prompt.txt",
        parse_prompt_file="data/parse_prompt.txt",
        openai_key_file="data/openai_keys.txt",
    ):
        self.dialogue = ""
        self.logging_dialogue = ""

        self.task_prompt_file = task_prompt_file
        self.replan_prompt_file = replan_prompt_file
        self.parse_prompt_file = parse_prompt_file

        self.goal_lib = self.load_goal_lib(goal_lib_json)
        self.openai_client = OpenAI(api_key=self.load_openai_key(openai_key_file))
        self.supported_objects = self.get_supported_objects(self.goal_lib)

    def reset(self):
        self.dialogue = ""
        self.logging_dialogue = ""

    def load_openai_key(self, openai_key_file) -> str:
        with open(openai_key_file, "r") as f:
            context = f.read()
        return context.split("\n")[0]

    def load_goal_lib(self, goal_lib_json):
        with open(goal_lib_json, "r") as f:
            goal_lib = json.load(f)
        return goal_lib

    def get_supported_objects(self, goal_lib):
        supported_objs = {}
        for key in goal_lib.keys():
            obj = list(goal_lib[key]["output"].keys())[0]
            supported_objs[obj] = goal_lib[key]
            supported_objs[obj]["name"] = key
        return supported_objs

    def load_parser_prompt(
        self,
    ):
        with open(self.parse_prompt_file, "r") as f:
            context = f.read()
        return context

    def load_initial_planning_prompt(self):
        with open(self.task_prompt_file, "r") as f:
            context = f.read()
        with open(self.replan_prompt_file, "r") as f:
            context += f.read()
        return context

    # @cache_results
    def query_codex(self, prompt_text):
        server_flag = 0
        server_error_cnt = 0
        response = ""
        while server_error_cnt < 2:
            try:
                logger.info(f"codex prompt_text length: {len(prompt_text)}")
                max_length = 4000 - 512
                prompt_text = prompt_text[-max_length:]
                response = self.openai_client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt_text,
                    temperature=0.7,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[
                        "Human:",
                    ],
                )
                server_flag = 1
                if server_flag:
                    break
            except Exception as e:
                server_error_cnt += 1
                time.sleep(1)
                logger.info(e)
        return response.choices[0].text

    # @cache_results
    def query_gpt3(self, prompt_text):
        server_flag = 0
        server_cnt = 0
        response = ""
        logger.info(f"gpt3 prompt_text length: {len(prompt_text)}")
        max_length = 4000 - 256
        prompt_text = prompt_text[-max_length:]
        while server_cnt < 3:
            try:
                response = self.openai_client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt_text,
                    temperature=0,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                server_flag = 1
                if server_flag:
                    break
            except Exception as e:
                server_cnt += 1
                time.sleep(1)
                logger.info(e)
        return response.choices[0].text

    def online_parser(self, text):
        self.parser_prompt = self.load_parser_prompt()
        parser_prompt_text = self.parser_prompt + text
        parsed_info = self.query_gpt3(parser_prompt_text)

        lines = parsed_info.split("\n")
        name = None
        obj = None
        rank = None

        for line in lines:
            line = line.replace(" ", "")
            # if "action:" in line:
            # action = line[7:]
            if "name:" in line:
                name = line[5:]
            elif "object:" in line:
                obj = eval(line[7:])
            elif "rank:" in line:
                rank = int(line[5:])
            else:
                pass
        return name, obj, rank

    def check_object(self, object):
        object_flag = False
        try:
            object_name = list(object.keys())[0]
            for goal in self.goal_lib.keys():
                if object_name == list(self.goal_lib[goal]["output"].keys())[0]:
                    object_flag = True
                    return goal
        except Exception as e:
            logger.info(e)
        return object_flag

    def generate_goal_list(self, plan):
        lines = plan.split("\n")
        goal_list = []
        for line in lines:
            if "#" in line:
                name, obj, rank = self.online_parser(f"input: {line}")
                logger.info(f"{name} {obj} {rank}")

                if name in self.goal_lib.keys():
                    goal_type = self.goal_lib[name]["type"]
                    goal_object = obj
                    goal_rank = rank
                    goal_precondition = {
                        **self.goal_lib[name]["precondition"],
                        **self.goal_lib[name]["tool"],
                    }
                    goal = {}
                    goal["name"] = name
                    goal["type"] = goal_type
                    goal["object"] = goal_object
                    goal["precondition"] = goal_precondition
                    goal["ranking"] = goal_rank
                    goal_list.append(goal)
                elif self.check_object(obj):
                    logger.info(
                        "parsed goal is not in controller goal keys. Now search the object items ..."
                    )
                    obj_name = list(obj.keys())[0]
                    goal_type = self.supported_objects[obj_name]["type"]
                    goal_object = obj
                    goal_rank = rank
                    goal_precondition = {
                        **self.supported_objects[obj_name]["precondition"],
                        **self.supported_objects[obj_name]["tool"],
                    }
                    goal = {}
                    goal["name"] = self.supported_objects[obj_name]["name"]
                    goal["type"] = goal_type
                    goal["object"] = goal_object
                    goal["precondition"] = goal_precondition
                    goal["ranking"] = goal_rank
                    goal_list.append(goal)
                else:
                    logger.error("parsed goal is not supported by current controller.")
        logger.info(f"Current Plan is {goal_list}")
        return goal_list

    def initial_planning(self, task_question):
        task_prompt = self.load_initial_planning_prompt()
        question = f"Human: {task_question}\n"
        task_prompt_text = task_prompt + question
        plan = self.query_codex(task_prompt_text)
        self.dialogue = self.load_initial_planning_prompt() + question
        self.dialogue += plan
        self.logging_dialogue = question
        self.logging_dialogue += plan
        logger.info(plan)
        return plan

    def generate_inventory_description(self, inventory):
        inventory_text = "Human: My inventory now has "
        for inv_item in inventory:
            if inv_item["name"] == "diamond_axe":
                continue
            if not inv_item["name"] == "air":
                inventory_text += f'{inv_item["quantity"]} {inv_item["name"]}, '
        logger.info(inventory_text)
        inventory_text += "\n"
        self.dialogue += inventory_text
        self.logging_dialogue += inventory_text
        return inventory_text

    def generate_success_description(self, step):
        result_description = f"Human: I succeed on step {step}.\n"
        self.dialogue += result_description
        self.logging_dialogue += result_description
        return result_description

    def generate_failure_description(self, step):
        result_description = f"Human: I fail on step {step}"
        self.dialogue += result_description
        self.logging_dialogue += result_description
        logger.info(result_description)
        detail_result_description = self.query_codex(self.dialogue)
        self.dialogue += detail_result_description
        self.logging_dialogue += detail_result_description
        logger.info(detail_result_description)
        return detail_result_description

    def generate_explanation(self):
        explanation = self.query_codex(self.dialogue)
        self.dialogue += explanation
        self.logging_dialogue += explanation
        logger.info(explanation)
        return explanation

    def replan(self, task_question):
        replan_description = (
            f"Human: Please fix above errors and replan the task '{task_question}'.\n"
        )
        self.dialogue += replan_description
        self.logging_dialogue += replan_description
        plan = self.query_codex(self.dialogue)
        logger.info(plan)
        self.dialogue += plan
        self.logging_dialogue += plan
        return plan

    def save_dialogue(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.logging_dialogue)


class DEP:
    def __init__(self, cfg, env):
        self.goal_mapping_cfg = self.load_goal_mapping_config()
        self.mineclip_prompt_dict = self.goal_mapping_cfg["mineclip"]

        # unify the mineclip and clip
        self.clip_prompt_dict = self.goal_mapping_cfg["clip"]
        self.goal_mapping_dict = self.goal_mapping_cfg["horizon"]

        no_op = env.action_space.no_op()
        self.miner = MineAgent(cfg, no_op=no_op)
        self.crafter = CraftAgent(env, no_op=no_op)
        self.planner = Planner()

    def load_goal_mapping_config(self):
        with open(goal_mapping_json, "r") as f:
            goal_mapping_cfg = json.load(f)
        return goal_mapping_cfg

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

    def reset(self, task_question: str):
        self.task_question = task_question

        self.planner.reset()
        plan = self.planner.initial_planning(task_question)
        self.goal_list = self.planner.generate_goal_list(plan)
        # NOTE: if the goal list is empty, then the DEP default goal is to mine a log
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

        curr_goal_type = self.curr_goal["type"]

        # check if the inventory has the preconditions
        if curr_goal_type == "mine" and not self.check_precondition(
            info["inventory"], self.curr_goal["precondition"]
        ):
            self.replan_task(info["inventory"], self.task_question)
        # limit craft and smelt goal to 150 and 200 steps before replanning
        elif curr_goal_type == "craft" and self.goal_eps > 150:
            self.replan_task(info["inventory"], self.task_question)
        elif curr_goal_type == "smelt" and self.goal_eps > 200:
            self.replan_task(info["inventory"], self.task_question)

        # limit the # replanning rounds to 12
        if self.replan_rounds > 12:
            logger.info(f"{t}: replanning over rounds")
            raise ValueError("Too many replanning rounds")

        # get the rolling window of the actions
        sf = 5  # skip frame
        wl = 10  # window len
        end = self.actions.shape[0] - 1
        rg = torch.arange(
            end, min(max(end - sf * (wl - 1) - 1, self.seek_point - 1), end - 1), -sf
        ).flip(0)

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

        # update the goal_eps
        self.goal_eps += 1

        return action, curr_goal["name"]

    def save_dialogue(self, filename: str):
        self.planner.save_dialogue(filename)

    def save_goal_history(self, filename: str):
        with open(filename, "w") as f:
            for item in self.goal_history:
                f.write(json.dumps(item) + "\n")

    def save_logs(self, output_dir: str, task_name: str, iteration: int):
        self.save_dialogue(f"{output_dir}/{task_name}/{task_name}_{iteration}.txt")
        self.save_goal_history(
            f"{output_dir}/{task_name}/{task_name}_{iteration}_goal_history.jsonl"
        )
