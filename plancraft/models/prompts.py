SYSTEM_PROMPT = """You are crafting in Minecraft. You need to decide on the next action.

**Crafting Grid**: The crafting table is organized into a 3x3 grid. Each slot in the grid has a unique identifier:
   - **Top row**: [A1] [A2] [A3]
   - **Middle row**: [B1] [B2] [B3]
   - **Bottom row**: [C1] [C2] [C3]

The output of the crafting process is placed in a designated output slot labeled [0] You cannot move or smelt items directly into slot [0]

**Inventory Slots**: The remaining inventory slots (outside of the crafting grid) are used for storing items. These slots are labeled as [I1] to [I36]

**Actions**: You can perform the following actions:
   - **move**: Transfer a specific quantity of an item from one slot to another.
   - **smelt**: Use items to smelt in a furnace (if applicable in the crafting process).

**Output Format**: Each action should be output in the following format:
   - `act: move from [Source] to [Target] with quantity N`
   - `act: smelt from [Source] to [Target] with quantity N`

   Example:
   - `act: move from [I2] to [A1] with quantity 3`

**Constraints**:
   - The output slot [0] is reserved for completed items. You cannot move or smelt items into [0]
   - Ensure items are placed correctly according to crafting recipes

Remember to always follow the grid layout and refer to slots using exact labels
"""


ACT_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """act: move from [I18] to [B1] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [B1] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """act: move from [I30] to [B2] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\nandesite [0] quantity 1\n - diorite [B1] quantity 1\n - cobblestone [B2] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """act: move from [0] to [I6] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory:\n - iron_ore [I36] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """act: smelt from [I36] to [I35] with quantity 1""",
    },
]


REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """thought: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: move from [I18] to [B1] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\n - diorite [B1] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """thought: Now I need to move the cobblestone into position 5 to be right of the diorite.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: move from [I30] to [B2] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory:\nandesite [0] quantity 1\n - diorite [B1] quantity 1\n - cobblestone [B2] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """thought: Now I can craft the andesite by moving it from the craft slot [0] to a free inventory slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: move from [0] to [I6] with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory:\n - iron_ore [I36] quantity 1\n - cobblestone [I30] quantity 1""",
    },
    {
        "role": "assistant",
        "content": """thought: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: smelt from [I36] to [I35] with quantity 1""",
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
                "text": """act: move from [I18] to [B1] with quantity 1""",
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
                "text": """act: move from [I30] to [B2] with quantity 1""",
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
                "text": """act: move from [0] to [I6] with quantity 1""",
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
                "text": """act: smelt from [I36] to [I35] with quantity 1""",
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
                "text": """thought: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """act: move from [I18] to [B1] with quantity 1""",
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
                "text": """thought: Now I need to move the cobblestone into position 5 to be right of the diorite.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """act: move from [I30] to [B2] with quantity 1""",
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
                "text": """thought: Now I can craft the andesite by moving it from craft slot to a free inventory slot.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """act: move from [0] to [I6] with quantity 1""",
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
                "text": """thought: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Ok"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """act: smelt from [I36] to [I35] with quantity 1""",
            }
        ],
    },
]
