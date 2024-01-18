import json
import os
import time
import warnings
from datetime import datetime

import hydra
import torch

from openai import OpenAI
from PIL import Image

from minedojo import MineDojoEnv
from models import CraftAgent, MineAgent

from models.utils import preprocess_obs, resize_image

warnings.filterwarnings("ignore")

prefix = os.getcwd()
task_info_json = os.path.join(prefix, "data/task_info.json")
goal_lib_json = os.path.join(prefix, "data/goal_lib.json")
goal_mapping_json = os.path.join(prefix, "data/goal_mapping.json")
task_prompt_file = os.path.join(prefix, "data/task_prompt.txt")
replan_prompt_file = os.path.join(prefix, "data/deps_prompt.txt")
parse_prompt_file = os.path.join(prefix, "data/parse_prompt.txt")
openai_key_file = os.path.join(prefix, "data/openai_keys.txt")

TASK_LIST = []
with open(task_info_json, "r") as f:
    task_info = json.load(f)
TASK_LIST = list(task_info.keys())


prefix = os.getcwd()


class Planner:
    def __init__(self):
        self.dialogue = ""
        self.logging_dialogue = ""
        self.goal_lib = self.load_goal_lib()
        self.openai_client = OpenAI(api_key=self.load_openai_key())
        self.supported_objects = self.get_supported_objects(self.goal_lib)

    def reset(self):
        self.dialogue = ""
        self.logging_dialogue = ""

    def load_openai_key(
        self,
    ) -> str:
        with open(openai_key_file, "r") as f:
            context = f.read()
        return context.split("\n")[0]

    def load_goal_lib(
        self,
    ):
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
        with open(parse_prompt_file, "r") as f:
            context = f.read()
        return context

    def load_initial_planning_prompt(self, group):
        with open(task_prompt_file, "r") as f:
            context = f.read()
        with open(replan_prompt_file, "r") as f:
            context += f.read()
        return context

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
                print(e)
        return response

    def query_gpt3(self, prompt_text):
        server_flag = 0
        server_cnt = 0
        response = ""
        print("prompt_text length:", len(prompt_text))
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
                print(e)
        return response

    def online_parser(self, text):
        self.parser_prompt = self.load_parser_prompt()
        parser_prompt_text = self.parser_prompt + text
        response = self.query_gpt3(parser_prompt_text)
        parsed_info = response.choices[0].text
        # print(parsed_info)
        lines = parsed_info.split("\n")

        name = None
        obj = None
        rank = None

        for line in lines:
            line = line.replace(" ", "")
            if "action:" in line:
                action = line[7:]
            elif "name:" in line:
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
            print(e)
        return object_flag

    def generate_goal_list(self, plan):
        lines = plan.split("\n")
        goal_list = []
        for line in lines:
            if "#" in line:
                name, obj, rank = self.online_parser(f"input: {line}")
                print("[INFO]:", name, obj, rank)

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
                    print(
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
                    print(
                        "[ERROR]: parsed goal is not supported by current controller."
                    )
        print(f"[INFO]: Current Plan is {goal_list}")
        return goal_list

    def initial_planning(self, group, task_question):
        task_prompt = self.load_initial_planning_prompt(group)
        question = f"Human: {task_question}\n"
        task_prompt_text = task_prompt + question
        response = self.query_codex(task_prompt_text)
        plan = response.choices[0].text
        self.dialogue = self.load_initial_planning_prompt(group) + question
        self.dialogue += plan
        self.logging_dialogue = question
        self.logging_dialogue += plan
        print(plan)
        return plan

    def generate_inventory_description(self, inventory):
        inventory_text = "Human: My inventory now has "
        for inv_item in inventory:
            if inv_item["name"] == "diamond_axe":
                continue
            if not inv_item["name"] == "air":
                inventory_text += f'{inv_item["quantity"]} {inv_item["name"]}, '
        print(inventory_text)
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
        print(result_description)
        response = self.query_codex(self.dialogue)
        detail_result_description = response.choices[0].text
        self.dialogue += detail_result_description
        self.logging_dialogue += detail_result_description
        print(detail_result_description)
        return detail_result_description

    def generate_explanation(self):
        response = self.query_codex(self.dialogue)
        explanation = response.choices[0].text
        self.dialogue += explanation
        self.logging_dialogue += explanation
        print(explanation)
        return explanation

    def replan(self, task_question):
        replan_description = (
            f"Human: Please fix above errors and replan the task '{task_question}'.\n"
        )
        self.dialogue += replan_description
        self.logging_dialogue += replan_description
        response = self.query_codex(self.dialogue)
        plan = response.choices[0].text
        print(plan)
        self.dialogue += plan
        self.logging_dialogue += plan
        return plan


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_workers = 0
        self.env = MineDojoEnv(
            name=cfg["eval"]["env_name"],
            img_size=(
                cfg["simulator"]["resolution"][0],
                cfg["simulator"]["resolution"][1],
            ),
            rgb_only=False,
        )
        self.task_list = TASK_LIST

        self.use_ranking_goal = cfg["goal_model"]["use_ranking_goal"]

        self.goal_mapping_cfg = self.load_goal_mapping_config()
        self.mineclip_prompt_dict = self.goal_mapping_cfg["mineclip"]
        self.clip_prompt_dict = self.goal_mapping_cfg[
            "clip"
        ]  # unify the mineclip and clip
        self.goal_mapping_dict = self.goal_mapping_cfg["horizon"]

        print(
            "[Progress] [red]Computing goal embeddings using MineClip's text encoder..."
        )
        # rely_goals = [val for val in self.goal_mapping_dict.values()]
        # self.embedding_dict = accquire_goal_embeddings(
        #     cfg["pretrains"]["clip_path"], rely_goals, device=device
        # )

        self.goal_model_freq = cfg["goal_model"]["freq"]
        self.goal_list_size = cfg["goal_model"]["queue_size"]

        self.record_frames = cfg["record"]["frames"]

        no_op = self.env.action_space.no_op()

        self.miner = MineAgent(cfg, no_op=no_op, max_ranking=15)
        self.crafter = CraftAgent(self.env, no_op=no_op)
        self.planner = Planner()

        task = cfg["eval"]["task_name"]
        self.reset(task)

    def reset(self, task):
        print(f"[INFO]: resetting the task {task}")
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
            goal_mapping_dict = json.load(f)
        return goal_mapping_dict

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
            print(f"[INFO]: finish goal {self.curr_goal['name']}.")
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

    def logging(self, t):
        self.logs[t] = {}
        self.logs[t]["curr_plan"] = self.goal_list
        self.logs[t]["curr_goal"] = self.curr_goal
        self.logs[t]["curr_dialogue"] = self.planner.logging_dialogue

    @torch.no_grad()
    def eval_step(self, fps=200):
        self.mine_agent.eval()

        obs = self.env.reset()

        print("[INFO]: Evaluating the task is ", self.task)

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
        actions = torch.zeros(1, self.mine_agent.action_dim, device=self.device)

        curr_goal = None
        prev_goal = None
        seek_point = 0

        obs, reward, env_done, info = self.env.step(self.env.action_space.no_op())

        now = datetime.now()
        timestamp = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_"
        )
        log_folder_name = os.path.join(prefix, "logs/")
        if not os.path.exists(log_folder_name):
            os.mkdir(log_folder_name)
        log_file_name = log_folder_name + timestamp + self.task + ".json"
        with open(log_file_name, "w") as f:
            json.dump(self.logs, f, indent=4)

        # max_ep_len = task_eps[self.task]
        for t in range(0, self.max_ep_len):
            time.sleep(1 / fps)

            self.update_goal(info["inventory"])
            curr_goal = self.curr_goal

            if not prev_goal == curr_goal:
                print(f"[INFO]: Episode Step {t}, Current Goal {curr_goal}")
                seek_point = t
                actions = torch.zeros(
                    actions.shape[0], self.mine_agent.action_dim, device=self.device
                )
                self.logging(t)
                with open(log_file_name, "w") as f:
                    json.dump(self.logs, f, indent=4)
            prev_goal = curr_goal

            # take the current goal type
            curr_goal_type = self.curr_goal["type"]

            sf = self.cfg["data"]["skip_frame"]
            wl = self.cfg["data"]["window_len"]

            end = actions.shape[0] - 1
            rg = torch.arange(
                end, min(max(end - sf * (wl - 1) - 1, seek_point - 1), end - 1), -sf
            ).flip(0)

            # DONE: change the craft agent into craft actions
            if curr_goal_type in ["craft", "smelt"]:
                # action_done = False
                preconditions = self.curr_goal["precondition"].keys()
                goal = list(self.curr_goal["object"].keys())[0]
                curr_actions, action_done = self.craft_agent.get_action(
                    preconditions, curr_goal_type, goal
                )

            elif curr_goal_type == "mine":
                # action_done = True
                goal = self.goal_mapping_dict[list(self.curr_goal["object"].keys())[0]]
                goal_embedding = self.embedding_dict[goal]
                goals = (
                    torch.from_numpy(goal_embedding).to(self.device).repeat(len(rg), 1)
                )
                complete_states = slice_obs(states, rg)
                complete_states["prev_action"] = actions[rg]

                _ranking, _action = self.mine_wrapper.get_action(
                    goal, goals, complete_states
                )
                curr_actions = _action
            else:
                print("Undefined action type !!")

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
                                act = self.env.action_space.no_op()
                                act[5] = 5
                                act[7] = item["index"]
                                self.env.step(act)
                                break
            #! indent change
            action = curr_actions
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            print(f"[INFO]: Taking action: {action}")
            obs, reward, env_done, info = self.env.step(action)

            if self.record_frames:
                video_frames.append(obs["rgb"])
                goal_frames.append(curr_goal["name"])
            obs = preprocess_obs(obs)

            if type(action) != torch.Tensor:
                action = torch.from_numpy(action)
            if action.device != self.device:
                action = action.to(self.device)

            states = stack_obs(states, obs)
            actions = torch.cat([actions, action.unsqueeze(0)], dim=0)

            self.goal_eps += 1
            if curr_goal_type == "mine" and not self.check_precondition(
                info["inventory"], self.curr_goal["precondition"]
            ):
                self.replan_task(info["inventory"], self.task_question)
                self.logging(t)
                with open(log_file_name, "w") as f:
                    json.dump(self.logs, f, indent=4)
            elif curr_goal_type == "craft" and self.goal_eps > 150:
                self.replan_task(info["inventory"], self.task_question)
                self.logging(t)
                with open(log_file_name, "w") as f:
                    json.dump(self.logs, f, indent=4)
            elif curr_goal_type == "smelt" and self.goal_eps > 200:
                self.replan_task(info["inventory"], self.task_question)
                self.logging(t)
                with open(log_file_name, "w") as f:
                    json.dump(self.logs, f, indent=4)

            if self.replan_rounds > 12:
                print("[INFO]: replanning over rounds")
                break

            if self.check_done(
                info["inventory"], self.task_obj
            ):  # check if the task is done?
                env_done = True
                print(f"[INFO]: finish goal {self.curr_goal['name']}.")
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
            print("[INFO]: saving the frames")
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
            print(f"record imgs length: {len(imgs)}")
            now = datetime.now()
            timestamp = (
                f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
            )
            folder_name = os.path.join(prefix, "recordings/" + timestamp + "/")
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            imgs[0].save(
                folder_name + self.task + ".gif",
                save_all=True,
                append_images=imgs[1:],
                optimize=False,
                quality=0,
                duration=150,
                loop=0,
            )
            with open(folder_name + self.task + ".json", "w") as f:
                json.dump(self.logs, f, indent=4)

        return env_done, t  # True or False, episode length

    def single_task_evaluate(self):
        loops = self.cfg["eval"]["goal_ratio"]
        if self.num_workers == 0:
            succ_rate = 0
            episode_lengths = []
            for i in range(loops):
                try:
                    self.reset(self.task)
                    succ_flag, min_episode = self.eval_step()
                except Exception as e:
                    print(e)
                    succ_flag = False
                    min_episode = 0
                succ_rate += succ_flag
                if succ_flag:
                    episode_lengths.append(min_episode)
                print(
                    f"Task {self.task} | Iteration {i} | Successful {succ_flag} | Episode length {min_episode} | Success rate {succ_rate/(i+1)}"
                )
            print("success rate: ", succ_rate / loops)
            print(
                "average episode length:",
                sum(episode_lengths) / (len(episode_lengths) + 0.01),
            )


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    print(cfg)
    evaluator = Evaluator(cfg)
    evaluator.single_task_evaluate()


if __name__ == "__main__":
    main()
