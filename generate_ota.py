import os
import json
import glob

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StopStringCriteria,
)

from plancraft.environments.recipes import RECIPES

BASE_PROMPT = """
You are crafting in Minecraft.

The first 10 slots in the inventory are reserved for crafting and correspond to the minecraft crafting table.
[1, 2, 3]
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move or smelt items into output slot 0.
The remaining slots (10-45) are for storing items.

For a given observation action pair, you should explain the reasoning behind taking that action. Mention the state of the environment, what the next recipe to craft is (in the context of the greater planning problem) and why this move should be taken.

Examples:

TASK: Craft an item of type: andesite
Crafting path: [andesite]
Relevant recipes:
1 diorite, 1 cobblestone -> 2 andesite

1. inventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'
action: move from slot 27 to slot 4 with quantity 1
thought: To solve this task I need to craft andesite. Andesite requires placing 1 diorite and 1 cobblestone side by side in the crafting table. The crafting table is currently empty, therefore I will first move diorite from slot 27 into the crafting grid.

2. inventory=[{"type": "diorite", "slot": 4, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]
action: move from slot 39 to slot 5 with quantity 1
thought: Since the diorite has been moved into the crafting grid into slot 4, I now need to move the cobblestone to the right of it. Slot 5 is to the right of slot 4, and therefore I will move the cobblestone to slot 5.

3. inventory=[{"type": "andesite", "slot": 0, "quantity": 1},{"type": "diorite", "slot": 4, "quantity": 1},{"type": "cobblestone", "slot": 5, "quantity": 1}]
action: move from slot 0 to slot 15 with quantity 1
thought: Now I can craft the andesite by moving it from the craft slot (0) to a free inventory slot (eg., 15). 

TASK: Craft an item of type: iron_ingot
Crafting path: [iron_ingot]
Relevant recipes:
1 iron_ore -> 1 iron_ingot
iron_nugget	iron_nugget	iron_nugget
iron_nugget	iron_nugget	iron_nugget -> 1 iron_ingot
iron_nugget	iron_nugget	iron_nugget
1 iron_block -> 9 iron_ingot

1. inventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]
action: smelt from slot 45 to slot 44 with quantity 1
thought: Given my inventory, to craft an iron_ingot I need to smelt the iron_ore at slot 45 into an empty slot (eg., 44).

"""


def generate_task_message(example_text, optimal_path: list[str]):
    task_message = BASE_PROMPT + example_text
    task_message += f"\nCrafting path: {optimal_path}"
    recipe_message = ["\nRelevant recipes:"]
    for target in set(optimal_path):
        for recipe_possible in RECIPES[target]:
            recipe_message.append(recipe_possible.__prompt_repr__())
    task_message += "\n".join(recipe_message)
    return task_message


@torch.no_grad()
def generate_thoughts(
    row, model, tokenizer, max_window_size=30
) -> list[dict[str, str]]:
    step = 1
    task_message = BASE_PROMPT
    optimal_path = row.optimal_path
    initial_text = row.messages[0]["content"].split("\n")[0]
    task_message = generate_task_message(initial_text, optimal_path)

    OTA_texts = []
    OTA_messages = []
    for i in range(0, len(row.messages), 2):
        user_entry = row.messages[i]
        assert user_entry["role"] == "user"
        assistant_entry = row.messages[i + 1]
        assert assistant_entry["role"] == "assistant"
        inventory = user_entry["content"].split("inventory=")[1]
        action = assistant_entry["content"]

        # step without thought
        current_step_text = (
            f"inventory={inventory}\n{action.replace('act:', 'action:')}\nthought:"
        )

        # history of the last max_window_size steps
        current_step_text_all = "".join(
            [
                f"\n\n{s+1}. {t}"
                for s, t in enumerate(
                    OTA_texts[-(max_window_size + 1) :] + [current_step_text]
                )
            ]
        )

        # regenerate task message based on path left
        task_message = generate_task_message(initial_text, optimal_path)

        prompt = task_message + current_step_text_all

        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", max_length=8192, truncation=True
        )
        tokenized_prompt = {k: v.to("cuda") for k, v in tokenized_prompt.items()}
        outputs = model.generate(
            **tokenized_prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip(
            "\n"
        )
        thought = generated_text.split("thought:")[-1].strip().split("\n")[0]
        print(f"Step {step}: {thought}")
        OTA_texts.append(f"{current_step_text} {thought}")

        # if crafting of recipe is done -> reset the current trace and pop the recipe
        # this allows the model to generate thoughts for each subtask separately
        # to limit hallucinations / off-task responses
        if "move from slot 0 " in action:
            optimal_path.pop(0)
            OTA_texts = []

        # Observation
        OTA_messages.append(user_entry)
        # Thought
        OTA_messages.append({"role": "assistant", "content": f"thought: {thought}"})
        OTA_messages.append({"role": "user", "content": "Ok"})
        # Action
        OTA_messages.append(assistant_entry)
        step += 1

    return OTA_messages


if __name__ == "__main__":
    print("Loading model")
    model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3.1-70B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    model = torch.compile(model)

    print("Model loaded")
    stopping_criteria = StoppingCriteriaList(
        [
            StopStringCriteria(
                tokenizer=tokenizer, stop_strings=["\n", "\n\n", ".\n\n", "\n\n\n"]
            )
        ]
    )

    for split in ["train", "val"]:
        with open(f"data/{split}.json", "r") as f:
            examples = json.load(f)
        df = pd.DataFrame(examples)
        dialogues_paths = sorted(glob.glob(f"data/oracle/{split}/oa/*.json"))
        dialogues = []
        for path in dialogues_paths:
            with open(path, "r") as f:
                tmp = {
                    "messages": json.load(f),
                    "example_id": path.split("/")[-1].split(".json")[0],
                }
                dialogues.append(tmp)
        dialogue_df = pd.DataFrame(dialogues)
        df = pd.merge(df, dialogue_df, left_on="id", right_on="example_id", how="inner")
        os.makedirs(f"data/oracle/{split}/ota-v2", exist_ok=True)
        for i, row in df.iterrows():
            print(f"Processing {split}: {i}/{len(df)}")
            example_id = row["id"]
            path = f"data/oracle/{split}/ota-v2/{example_id}.json"
            if os.path.exists(path):
                continue
            output_messages = generate_thoughts(row, model, tokenizer)
            with open(path, "w") as f:
                json.dump(output_messages, f, indent=2)
