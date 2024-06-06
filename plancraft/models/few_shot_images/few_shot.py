# Few shot learning examples taken from training set
REACT_EXAMPLE = [
    # image 0
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory='[{"type": "diorite", "slot": 27, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """think: To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side.""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """act: move from slot 27 to slot 4 with quantity 1""",
    },
    # image 1
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory=[{"type": "diorite", "slot": 4,  "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """think: Now I need to move the cobblestone into position 5 to be right of the diorite.""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """act: move from slot 39 to slot 5 with quantity 1""",
    },
    # image 3
    {
        "role": "user",
        "content": """TASK: Craft an item of type: iron_ingot\ninventory='[{"type": "iron_ore", "slot": 45, "quantity": 1},{"type": "cobblestone", "slot": 39, "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """think: To craft an iron_ingot, I need to smelt iron_ore into an empty slot.""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """act: smelt from slot 45 to slot 44 with quantity 1""",
    },
]
