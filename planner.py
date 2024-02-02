import functools
import json
import logging

from openai import OpenAI

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

    def load_initial_planning_prompt(self, group):
        with open(self.task_prompt_file, "r") as f:
            context = f.read()
        with open(self.replan_prompt_file, "r") as f:
            context += f.read()
        return context

    @cache_results
    def query_codex(self, prompt_text):
        server_flag = 0
        server_error_cnt = 0
        response = ""
        while server_error_cnt < 2:
            try:
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
                logger.info(e)
        return response.choices[0].text

    @cache_results
    def query_gpt3(self, prompt_text):
        server_flag = 0
        server_cnt = 0
        response = ""
        logger.info("prompt_text length:", len(prompt_text))
        prompt_text = prompt_text[-4000:]
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
                logger.info(f"[INFO]: {name} {obj} {rank}")

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
                        "[INFO]: parsed goal is not in controller goal keys. Now search the object items ..."
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
                    logger.info(
                        "[ERROR]: parsed goal is not supported by current controller."
                    )
        logger.info(f"[INFO]: Current Plan is {goal_list}")
        return goal_list

    def initial_planning(self, group, task_question):
        task_prompt = self.load_initial_planning_prompt(group)
        question = f"Human: {task_question}\n"
        task_prompt_text = task_prompt + question
        plan = self.query_codex(task_prompt_text)
        self.dialogue = self.load_initial_planning_prompt(group) + question
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
