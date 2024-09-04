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

Inventory Slots: The remaining inventory slots (outside of the crafting grid) are used for storing items. These slots are labeled as [I1] to [I36]

"""

BASE_SYSTEM_PROMPT_EXAMPLE = """Example:
    - `move: from [I2] to [A1] with quantity 3`

Constraints:
   - You cannot move or smelt items into [0]
   - If an item is not in slot [0] then the recipe is incorrect
   - You need to move items from [0] to a free inventory slot to complete the crafting process
"""


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


ACT_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """move: from [I18] to [B1] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [B1] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """move: from [I30] to [B2] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\nandesite [0] quantity 1\n - diorite [B1] quantity 1\n - cobblestone [B2] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """move: from [0] to [I6] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory:\n - iron_ore [I36] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """smelt: from [I36] to [I35] with quantity 1""",
    },
]


REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """move: from [I18] to [B1] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [B1] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: Now I need to move the cobblestone into position [B2] to be right of the diorite.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """move: from [I30] to [B2] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\nandesite [0] quantity 1\n - diorite [B1] quantity 1\n - cobblestone [B2] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: Now I can craft the andesite by moving it from the craft slot [0] to a free inventory slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """move: from [0] to [I6] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory:\n - iron_ore [I36] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """smelt: from [I36] to [I35] with quantity 1""",
    },
]


ACT_EXAMPLE_IMGS = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: andesite"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """move: from [I18] to [B1] with quantity 1""",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: andesite"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """move: from [I30] to [B2] with quantity 1""",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: andesite"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """move: from [0] to [I6] with quantity 1""",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: iron_ingot"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """smelt: from [I36] to [I35] with quantity 1""",
            }
        ],
    },
]


REACT_EXAMPLE_IMGS = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: andesite"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """think: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """move: from [I18] to [B1] with quantity 1""",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: andesite"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """think: Now I need to move the cobblestone into position [B2] to be right of the diorite.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """move: from [I30] to [B2] with quantity 1""",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: andesite"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """think: Now I can craft the andesite by moving it from craft slot to a free inventory slot.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """move: from [0] to [I6] with quantity 1""",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Craft an item of type: iron_ingot"""},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """think: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """smelt: from [I36] to [I35] with quantity 1""",
            }
        ],
    },
]


TOOLS_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: I need to craft andesite, but I'm not sure how to. I should start by searching for the recipe.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """search: andesite""",
    },
    {
        "role": "user",
        "content": """Recipes to craft andesite:\nrecipe 1:\ndiorite at [B1]\ncobblestone at [B2]""",
    },
    {
        "role": "assistant",
        "content": """move:: from [I18] to [B1] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [B1] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: Since I've moved the diorite to [B1], I now need to move the cobblestone into [B2] to complete the recipe.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """move:: from [I30] to [B2] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\nandesite [0] quantity 1\n - diorite [B1] quantity 1\n - cobblestone [B2] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: I can craft the andesite by moving it from the craft slot [0] to a free inventory slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """move:: from [0] to [I6] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory:\n - iron_ore [I36] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """think: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """smelt:: from [I36] to [I35] with quantity 1""",
    },
]


# [2024-09-04 12:16:16,259][plancraft.evaluator][INFO] - {'avg_success_rate': 0.226890756302521, 'avg_number_of_steps': 0.0, 'avg_num_tokens_used': 412118.487394958, 'impossible_success_rate': 0.95, 'shaped_success_rate': 0.15217391304347827, 'mixed_success_rate': 0.0, 'shapeless_success_rate': 0.058823529411764705, 'smelting_success_rate': 0.0}
# [2024-09-04 13:11:31,201][plancraft.evaluator][INFO] - {'avg_success_rate': 0.2773109243697479, 'avg_number_of_steps': 0.0, 'avg_num_tokens_used': 506813.9411764706, 'impossible_success_rate': 0.95, 'shaped_success_rate': 0.21739130434782608, 'mixed_success_rate': 0.03125, 'shapeless_success_rate': 0.17647058823529413, 'smelting_success_rate': 0.0}
