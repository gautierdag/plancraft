import os
import json

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StopStringCriteria,
)

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
thought: To solve this task I need to craft andesite. Andesite requires placing 1 diorite and 1 cobblestone side by side in the crafting table, therefore I will first need to move the diorite from slot 27 into the crafting grid. 

2. inventory=[{"type": "diorite", "slot": 4, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]
action: move from slot 39 to slot 5 with quantity 1
thought: Since the diorite has been moved into the crafting grid, I now need to move the cobblestone to the right of it. Slot 5 is to the right of slot 4, and therefore I will move the cobblestone to slot 5.

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
thought: To craft an iron_ingot, I need to smelt the iron_ore at slot 45 into an empty slot (eg., 44).

"""

from plancraft.environments.recipes import RECIPES

def generate_thoughts(
    row, model, tokenizer, max_window_size=30
) -> list[dict[str, str]]:
    step = 1
    task_message = BASE_PROMPT + row.messages[1]["content"].split("\n")[0]
    task_message += f"\nCrafting path: {row.optimal_path}"
    recipe_message = ["\nRelevant recipes:"]
    for target in set(row.optimal_path):
        for recipe_possible in RECIPES[target]:
            recipe_message.append(recipe_possible.__prompt_repr__())
    task_message += "\n".join(recipe_message)
    OTA_texts = []
    OTA_messages = []
    for i in range(1, len(row.messages), 2):
        user_entry = row.messages[i]
        assert user_entry["role"] == "user"
        assistant_entry = row.messages[i + 1]
        assert assistant_entry["role"] == "assistant"
        inventory = user_entry["content"].split("inventory=")[1]
        action = assistant_entry["content"]
        current_step_text = (
            f"\n\n{step}. inventory={inventory}\naction: {action}\nthought:"
        )
        prompt = (
            task_message + "".join(OTA_texts[-max_window_size:]) + current_step_text
        )
        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", max_length=8192, truncation=True
        )
        tokenized_prompt = {k: v.to("cuda") for k, v in tokenized_prompt.items()}
        outputs = model.generate(
            **tokenized_prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).rstrip(
            "\n"
        )
        thought = generated_text.split("thought:")[-1].strip()

        OTA_texts.append(f"{current_step_text}{thought}")

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
    model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded")
    stopping_criteria = StoppingCriteriaList(
        [
            StopStringCriteria(
                tokenizer=tokenizer, stop_strings=["\n", "\n\n", ".\n\n", "\n\n\n"]
            )
        ]
    )

    for split in ["train", "val"]:
        split = "train"
        with open(f"data/{split}.json", "r") as f:
            examples = json.load(f)
        df = pd.DataFrame(examples)
        dialogues = []
        with open(f"data/oracle/{split}.jsonl", "r") as f:
            for line in f:
                dialogues.append(json.loads(line))
        dialogue_df = pd.DataFrame(dialogues)
        df = pd.merge(df, dialogue_df, left_on="id", right_on="example_id", how="inner")
        os.makedirs(f"data/oracle/ota-v2/{split}", exist_ok=True)
        for i, row in df.iterrows():
            print(f"Processing {split}: {i}/{len(df)}")
            example_id = row["id"]
            path = f"data/oracle/ota-v2/{split}/{example_id}.json"
            if os.path.exists(path):
                continue
            output_messages = generate_thoughts(row, model, tokenizer)
            with open(path, "w") as f:
                json.dump(output_messages, f)
