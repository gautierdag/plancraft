import json
import gc
import time
import math
from collections import defaultdict
from dataclasses import asdict

import hydra
import wandb
from omegaconf import DictConfig
import pandas as pd

from plancraft.baselines import ActionStep, OneShotOpenAILLM, ReactOpenAILLM
from plancraft.llms import get_llm_generator

# Load the task info
tasks_path = "data/task_info.json"
with open(tasks_path, "r") as f:
    TASKS = json.load(f)

# Load the Tech Tree
TECH_TREE = {}
with open("data/goal_lib.json", "r") as f:
    goal_lib = json.load(f)
for g in goal_lib:
    k = g.replace("smelt_", "").replace("craft_", "").replace("mine_", "")
    TECH_TREE[k] = goal_lib[g]
    TECH_TREE[k]["name"] = k


def is_tool(item: str) -> bool:
    return "pickaxe" in item or "furnace" in item or "crafting_table" in item


def parse_target_for_wood_type(name: str) -> str:
    """
    Utility function to handle different wood types and sometimes diamond_ore
    """
    if "_log" in name:
        return "log"
    if "_planks" in name:
        return "planks"
    if "diamond_ore" in name:
        return "diamond"
    return name


def get_plan(target: str, need=1) -> list[dict]:
    """
    Obtains a close to optimal plan to achieve a target item.

    NOTE: resources are calculated from top up so might lead to overestimation
    """

    goal = TECH_TREE[target]
    goal["quantity_needed"] = need
    goal["depth"] = 0
    tree = {target: goal}

    def travel_tech_tree(current: str, quantity_needed: int, depth=1):
        """
        Recursive function to travel the tech tree
        """
        # add children
        requirements = TECH_TREE[current]["precondition"] | TECH_TREE[current]["tool"]
        quantity_to_produce = TECH_TREE[current]["output"][current]

        for r in requirements:
            cost_to_produce = requirements[r]

            # if we need to produce more than single step (ignore tools)
            if quantity_to_produce < quantity_needed and not is_tool(r):
                cost_to_produce = math.ceil(
                    cost_to_produce * (quantity_needed / quantity_to_produce)
                )
            # node already exists
            if r in tree:
                # tools are multi-use
                if is_tool(r):
                    tree[r]["depth"] = max(tree[r]["depth"], depth)
                    return

                tree[r]["quantity_needed"] += cost_to_produce
                tree[r]["depth"] = max(tree[r]["depth"], depth)
                travel_tech_tree(r, cost_to_produce, depth=depth + 1)
                # return

            # new tech
            else:
                tree[r] = TECH_TREE[r]
                tree[r]["quantity_needed"] = cost_to_produce
                tree[r]["depth"] = depth
                travel_tech_tree(r, cost_to_produce, depth=depth + 1)

    travel_tech_tree(target, need)

    # sort by depth
    plan = sorted(tree.values(), key=lambda x: x["depth"], reverse=True)
    return plan


def process_step(
    goal: ActionStep, current_inventory: dict[str, int]
) -> tuple[bool, str, any]:
    """
    Process a single step of the plan and update the inventory accordingly.

    Parameters:
    - goal: A dictionary representing the goal to achieve in this step.
    - current_inventory: A dictionary representing the current inventory of items.

    Returns:
    - A tuple (success: bool, error_type: str or None, error_value: any)
    """
    success = True

    if not isinstance(goal, ActionStep):
        return False, "parsing_error", goal

    try:
        target = goal.output.strip()
        target = parse_target_for_wood_type(target)
        if target not in TECH_TREE:
            return False, "unknown_item", target
        if goal.type.strip() != TECH_TREE[target]["type"]:
            return False, "action_type_mismatch", goal.type
        if not set(TECH_TREE[target]["tool"].keys()).issubset(set(goal.tool.keys())):
            return False, "missing_tools", list(TECH_TREE[target]["tool"].keys())
        if not set(TECH_TREE[target]["precondition"].keys()).issubset(
            set(current_inventory.keys())
        ):
            return (
                False,
                "missing_materials",
                list(TECH_TREE[target]["precondition"].keys()),
            )

        # Add the outcome to the inventory
        quantity_needed = goal.quantity_needed
        while quantity_needed > 0:
            for item in TECH_TREE[target]["precondition"]:
                if (
                    current_inventory[item] - TECH_TREE[target]["precondition"][item]
                    < 0
                ):
                    return False, "insufficient_materials", item
                current_inventory[item] -= TECH_TREE[target]["precondition"][item]

            current_inventory[target] += TECH_TREE[target]["output"][target]
            quantity_needed -= TECH_TREE[target]["output"][target]

        return success, None, None
    except Exception as e:
        return False, "unknown_error", str(e)


def evaluate_generated_plan(
    parsed_plan: list[ActionStep], target: str
) -> tuple[bool, str, any]:
    success = False
    current_inventory = defaultdict(int)
    current_inventory["diamond_axe"] = 1

    if len(parsed_plan) == 0:
        return False, "no_plan", None

    for goal in parsed_plan:
        success, error_type, error_value = process_step(goal, current_inventory)
        if not success:
            return False, error_type, error_value

    if current_inventory[target] > 0:
        success = True

    return success, "", None


def eval_one_shot_llm(cfg: dict, model_name: str, num_generations=5):
    llm_model = get_llm_generator(model_name)
    for i in range(num_generations):
        wandb.init(
            **cfg["wandb"],
            group=model_name,
            job_type="one_shot",
            config=cfg,
        )
        for k, v in TASKS.items():
            time_now = time.time()
            llm_model.reset()
            question = v["question"]
            target = question.split()[-1].replace("?", "")
            hash_key = f"one_shot_{model_name}_{target}_{i}"

            model = OneShotOpenAILLM(model=llm_model)

            generation = model.generate(question, temperature=1.0, max_tokens=1024)
            parsed_plan = model.parse_generated_plan(generation)

            suc, err, missing = evaluate_generated_plan(parsed_plan, target)
            # convert to dict for saving
            parsed_plan = [asdict(p) for p in parsed_plan]

            time_taken = time.time() - time_now
            print(
                f"Task: {k} | Time taken: {time_taken:.2f}s | Success: {suc} | Tokens used: {model.token_used}"
            )
            results = {
                "hash_key": hash_key,
                "group": v["group"],
                "target": target,
                "question": question,
                "success": suc,
                "tokens_used": model.token_used,
                "model_name": model_name,
                "plan": parsed_plan,
                "number_of_steps": len(parsed_plan),
                "generation": generation,
                "error": err,
                "missing": missing,
            }
            del model
            gc.collect()

        df = pd.DataFrame(results)
        table = wandb.Table(dataframe=df)
        wandb.log({"results": table})

        # calculate aggregate statistics
        grouped_df = df.groupby("group").agg(
            {
                "success": "mean",
                "tokens_used": "mean",
            }
        )
        # log the aggregate statistics
        wandb.log({"group_results": wandb.Table(dataframe=grouped_df)})
        wandb.finish()


def eval_reactive_llm(cfg: dict, model_name: str, num_generations=5, max_steps=20):
    llm_model = get_llm_generator(model_name)
    for i in range(num_generations):
        wandb.init(
            **cfg["wandb"],
            group=model_name,
            job_type="react",
            config=cfg,
        )
        for v in TASKS.values():
            llm_model.reset()
            question = v["question"]
            target = question.split()[-1].replace("?", "")
            hash_key = f"react_{model_name}_{target}_{i}"

            step = 1
            model = ReactOpenAILLM(model=llm_model)

            inventory = defaultdict(int)
            inventory["diamond_axe"] = 1
            print(f"Initial inventory: {inventory}")

            plan = []
            history = ""
            errors = defaultdict(int)
            task_success = False

            action_step = model.generate_initial_step(
                question, temperature=1.0, max_tokens=512
            )
            time_now = time.time()
            while not task_success and step < max_steps:
                history += f"Step {step} inventory: {inventory}\n"
                # check if the model is overthinking
                if model.is_thinking(action_step):
                    success = False
                    error_type = "over-thinking"
                    error_value = "too many thinking steps in a row"
                # process the action step
                else:
                    parsed_action_step = model.parse_step(action_step)
                    success, error_type, error_value = process_step(
                        parsed_action_step, inventory
                    )

                time_taken = time.time() - time_now
                time_now = time.time()
                if success:
                    print(
                        f"Step {step} successful ({time_taken:.2f}s | {model.token_used//1000}k toks)"
                    )
                    history += f"Step {step} successful: {parsed_action_step}\n"
                    plan.append(parsed_action_step)
                    observation = f"Success\ninventory = {dict(inventory)}"
                    action_step = model.generate_step(
                        observation, temperature=1.0, max_tokens=512
                    )
                    if parsed_action_step.output == target:
                        task_success = True
                        break
                else:
                    print(
                        f"Step {step} failed ({time_taken:.2f}s | {model.token_used//1000}k toks): {error_type} {error_value}"
                    )
                    history += f"Step {step} failed: {error_type} {error_value}\n"
                    errors[error_type] += 1
                    observation = f"ERROR: {error_type} {error_value}\ninventory = {dict(inventory)}"
                    print(f"Step {step} observation: {observation}")
                    action_step = model.generate_step(
                        observation, temperature=1.0, max_tokens=512
                    )

                step += 1

            # convert plan to dict for saving
            plan = [asdict(p) for p in plan]
            results = {
                "hash_key": hash_key,
                "target": target,
                "group": v["group"],
                "question": question,
                "plan": plan,
                "logs": history,
                "message_history": model.history,
                "errors": errors,
                "success": task_success,
                "number_of_steps": step,
                "number_of_thinking_steps": model.num_thinking_steps,
                "model_name": model_name,
                "tokens_used": model.token_used,
            }
            del model
            gc.collect()

        df = pd.DataFrame(results)
        table = wandb.Table(dataframe=df)
        wandb.log({"results": table})

        # calculate aggregate statistics
        grouped_df = df.groupby("group").agg(
            {
                "success": "mean",
                "number_of_steps": "mean",
                "number_of_thinking_steps": "mean",
                "tokens_used": "mean",
            }
        )
        # log the aggregate statistics
        wandb.log({"group_results": wandb.Table(dataframe=grouped_df)})
        wandb.finish()


@hydra.main(config_path="configs/text-env", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = dict(cfg)
    if cfg["mode"] == "one_shot":
        print("Evaluating one-shot LLMs")
        eval_one_shot_llm(
            cfg=cfg, model_name=cfg["model"], num_generations=cfg["num_generations"]
        )
    elif cfg["mode"] == "react":
        print("Evaluating reactive LLMs")
        eval_reactive_llm(
            cfg=cfg,
            model_name=cfg["model"],
            num_generations=cfg["num_generations"],
            max_steps=cfg["max_steps"],
        )
    else:
        print("Unknown mode")


if __name__ == "__main__":
    main()
