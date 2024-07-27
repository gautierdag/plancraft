REACT_SYSTEM_PROMPT = """You are crafting in Minecraft.
You need to decide on the next action.

You must output an action like the following:
act: move from slot X to slot Y with quantity Z

There are two types of actions
- move
- smelt

To assist with planning, you first generate some thoughts before answering. For example:
think: To craft an acacia_fence, I first need to craft acacia_planks so I need to move the log from ...

The first 10 slots in the inventory are reserved for crafting and correspond to the minecraft crafting table. 

[1, 2, 3] 
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move or smelt items into output slot 0.
The remaining slots (10-45) are for storing items.
"""

REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """thought: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: move from slot 27 to slot 4 with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """thought: Now I need to move the cobblestone into position 5 to be right of the diorite.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: move from slot 39 to slot 5 with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "andesite", "slot": 0,  "quantity": 1},{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 5, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """thought: Now I can craft the andesite by moving it from craft slot to a free inventory slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: move from slot 0 to slot 15 with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """thought: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
    },
    {"role": "user", "content": "Ok"},
    {
        "role": "assistant",
        "content": """act: smelt from slot 45 to slot 44 with quantity 1""",
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
                "text": """act: move from slot 27 to slot 4 with quantity 1""",
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
                "text": """act: move from slot 39 to slot 5 with quantity 1""",
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
                "text": """act: move from slot 0 to slot 15 with quantity 1""",
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
                "text": """act: smelt from slot 45 to slot 44 with quantity 1""",
            }
        ],
    },
]


ACT_SYSTEM_PROMPT = """You are crafting in Minecraft.
You need to decide on the next action.

You must output an action like the following:
act: move from slot X to slot Y with quantity Z

There are two types of actions
- move
- smelt

The first 10 slots in the inventory are reserved for crafting and correspond to the minecraft crafting table. 

[1, 2, 3] 
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move or smelt items into output slot 0.
The remaining slots (10-45) are for storing items.
"""

ACT_EXAMPLE = [
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """act: move from slot 27 to slot 4 with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """act: move from slot 39 to slot 5 with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: andesite\ninventory=[{"type": "andesite", "slot": 0,  "quantity": 1},{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 5, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """act: move from slot 0 to slot 15 with quantity 1""",
    },
    {
        "role": "user",
        "content": """Craft an item of type: iron_ingot\ninventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """act: smelt from slot 45 to slot 44 with quantity 1""",
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
                "text": """act: move from slot 27 to slot 4 with quantity 1""",
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
                "text": """act: move from slot 39 to slot 5 with quantity 1""",
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
                "text": """act: move from slot 0 to slot 15 with quantity 1""",
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
                "text": """act: smelt from slot 45 to slot 44 with quantity 1""",
            }
        ],
    },
]
