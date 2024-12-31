import numpy as np

from plancraft.environment.env import PlancraftEnvironment
from plancraft.environment.search import gold_search_recipe

VALID_ACTIONS = ["move", "smelt", "think", "search", "impossible"]

ACTIONS_DESCRIPTIONS = {
    "move": {
        "description": "Transfer a specific quantity of an item from one slot to another",
        "format": "`move: from [Source] to [Target] with quantity N`",
    },
    "smelt": {
        "description": "Smelt an item in a furnace and moves the output to a specific slot",
        "format": "`smelt: from [Source] to [Target] with quantity N`",
    },
    "think": {
        "description": "Generate thoughts to help you decide on the next action",
        "format": "`think: <thought message>`",
    },
    "search": {
        "description": "Search for a recipe to craft a specific item",
        "format": "`search: <recipe name>`",
    },
    "impossible": {
        "description": "Stop task if it is certain that it is impossible with given inventory",
        "format": "`impossible: <reason>`",
    },
}

BASE_SYSTEM_PROMPT = """You are crafting in Minecraft. You need to decide on the next action.

Crafting Grid: The crafting table is organized into a 3x3 grid. Each slot in the grid has a unique identifier:
    - Top row: [A1] [A2] [A3]
    - Middle row: [B1] [B2] [B3]
    - Bottom row: [C1] [C2] [C3]

The output of the crafting process is placed in a designated output slot labeled [0] You cannot move or smelt items directly into slot [0]

Inventory Slots: The remaining inventory slots (outside of the crafting grid) are used for storing items. These slots are labeled as [I1] to [I36]"""

BASE_SYSTEM_PROMPT_EXAMPLE = """Example:
    - `move: from [I2] to [A1] with quantity 3`
    - `smelt: from [I5] to [I6] with quantity 1`

Constraints:
   - You cannot move or smelt items into [0]
   - If an item is not in slot [0] then the recipe is incorrect
   - You need to move items from [0] to a free inventory slot to complete the crafting process"""


def get_system_prompt(actions: list[str]):
    assert set(actions).issubset(VALID_ACTIONS), f"Invalid actions: {actions}"
    assert "move" in actions, "move should be one of the actions"
    assert "smelt" in actions, "smelt should be one of the actions"

    descriptions = ""
    for action in actions:
        descriptions += f"\n\t- {action}: {ACTIONS_DESCRIPTIONS[action]['description']}"

    output_format = ""
    for action in actions:
        output_format += f"\n\t- {ACTIONS_DESCRIPTIONS[action]['format']}"

    return f"{BASE_SYSTEM_PROMPT}\n\nActions:{descriptions}\n\nFormat{output_format}\n\n{BASE_SYSTEM_PROMPT_EXAMPLE}"


CRAFTING_STEPS = [
    "Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1\n - cobblestone [I30] quantity 1",
    "Craft an item of type: andesite\ninventory:\n - diorite [B1] quantity 1\n - cobblestone [I30] quantity 1",
    "Craft an item of type: andesite\ninventory:\n - andesite [0] quantity 1\n - diorite [B1] quantity 1\n - cobblestone [B2] quantity 1",
    "Craft an item of type: iron_ingot\ninventory:\n - iron_ore [I36] quantity 1\n - cobblestone [I30] quantity 1",
]

BASE_ACTION_STEPS = [
    "move: from [I18] to [B1] with quantity 1",
    "move: from [I30] to [B2] with quantity 1",
    "move: from [0] to [I6] with quantity 1",
    "smelt: from [I36] to [I35] with quantity 1",
]

THINK_STEPS = [
    "think: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.",
    "think: Now I need to move the cobblestone into position [B2] to be right of the diorite.",
    "think: Now I can craft the andesite by moving it from the craft slot [0] to a free inventory slot.",
    "think: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.",
]

SEARCH_STEPS = [
    "search: andesite",
    None,
    None,
    "search: iron_ingot",
]


def get_prompt_example(
    actions: list[str],
    use_text_inventory=True,
    use_multimodal_content_format=False,
    use_images=False,
) -> list[dict]:
    """
    Generates a few-shot prompt for the crafting task
    """

    assert set(actions).issubset(VALID_ACTIONS), f"Invalid actions: {actions}"
    assert "move" in actions, "move should be one of the actions"
    assert "smelt" in actions, "smelt should be one of the actions"

    if use_images:
        assert (
            use_multimodal_content_format
        ), "use_images requires use_multimodal_content_format"

    example_dialogue = []
    for i, step in enumerate(CRAFTING_STEPS):
        text = step
        if not use_text_inventory:
            text = text.split("\ninventory:\n")[0]

        example_dialogue.append({"role": "user", "content": text})
        if "search" in actions and SEARCH_STEPS[i]:
            example_dialogue.append({"role": "assistant", "content": SEARCH_STEPS[i]})
            search_target = text.split("seach: ")[-1].strip()
            search_response = gold_search_recipe(search_target)
            example_dialogue.append({"role": "user", "content": search_response})
        if "think" in actions:
            example_dialogue.append({"role": "assistant", "content": THINK_STEPS[i]})
            example_dialogue.append({"role": "user", "content": "Ok"})
        example_dialogue.append({"role": "assistant", "content": BASE_ACTION_STEPS[i]})

    if not use_multimodal_content_format:
        return example_dialogue

    # convert to multimodal dialogue
    multimodal_dialogue = []
    for message in example_dialogue:
        if "Craft an item" in message["content"]:
            content_list = [
                {
                    "type": "text",
                    "text": message["content"],
                }
            ]
            if use_images:
                content_list.append(
                    {
                        "type": "image",
                    }
                )

            multimodal_dialogue.append(
                {"role": message["role"], "content": content_list}
            )
        else:
            multimodal_dialogue.append(
                {
                    "role": message["role"],
                    "content": [
                        {"type": "text", "text": message["content"]},
                    ],
                }
            )
    return multimodal_dialogue


def load_prompt_images(resolution: str) -> list[np.ndarray]:
    """
    Generates the images for the few-shot prompt in prompt.py
    """
    starting_inv = [
        {"type": "diorite", "slot": 27, "quantity": 1},
        {"type": "cobblestone", "slot": 39, "quantity": 1},
    ]

    env = PlancraftEnvironment(inventory=starting_inv, resolution=resolution)
    actions = [
        {"move": [0, 0, 0]},
        {"move": [27, 4, 1]},
        {"move": [39, 5, 1]},
    ]
    images = []
    for action in actions:
        obs = env.step(action)
        images.append(obs["image"])

    second_inv = [
        {"type": "iron_ore", "slot": 45, "quantity": 1},
        {"type": "cobblestone", "slot": 39, "quantity": 1},
    ]
    new_actions = [
        {"move": [0, 0, 0]},
    ]
    env.reset(new_inventory=second_inv)
    for action in new_actions:
        obs = env.step(action)
        images.append(obs["image"])

    assert len(images) == 4
    return images
