ONE_SHOT_SYSTEM_PROMPT = """You are a helper AI agent in Minecraft.
You need to generate a plan of actions for a TASK in Minecraft.

You must output a list of JSON objects like the following:
{
    "output": "iron_ingot", # item to be produced
    "quantity": 1 # quantity to be produced
    "type": "smelt", # type of action
    "materials": ["iron_ore"] # optional list of materials
    "tool": "furnace" # optional tool
}

There are three types of actions
- mine
- craft
- smelt

Otherwise you wish to think, you can output a string with your thought. For example:
{
"thought": "I need to gather some logs to craft planks using the diamond axe."
}

Example:

USER: TASK: How to obtain iron_ingot?\ninventory = {'diamond_axe'}

ASSISTANT:
[
    {"thought": "To solve this task I need to smelt an iron ingot, first I need to gather some logs to craft planks using the diamond axe."}
    {"type": "mine", "output": "log", "quantity": 3, "tool": "diamond_axe"}
    {"thought": "Now I can craft planks using the logs. I need 11 planks to craft sticks and a crafting table."}
    {"type": "craft", "output": "planks", "quantity": 11, "materials": ["log"]}
    {"thought": "Now I can craft sticks using the planks. I need 4 sticks to craft a wooden pickaxe and then the stone pickaxe."}
    {"type": "craft", "output": "stick", "quantity": 4, "materials": ["planks"]}
    {"thought": "Now I can craft a crafting table using the planks."}
    {"type": "craft", "output": "crafting_table", "quantity": 1, "materials": ["planks"]}
    {"thought": "Now I can craft a wooden pickaxe using the planks, sticks and crafting table."}
    {"type": "craft", "output": "wooden_pickaxe", "quantity": 1, "materials": ["planks", "stick"] "tool": "crafting_table"}
    {"thought": "Now I need to mine 11 cobblestone using the wooden pickaxe to be able to craft a stone pickaxe and a furnace."}
    {"type": "mine", "output": "cobblestone", "quantity": 11, "tool": "wooden_pickaxe"}
    {"thought": "Now I can craft a stone pickaxe using the cobblestone, sticks and crafting table."}
    {"type": "craft", "output": "stone_pickaxe", "quantity": 1, "materials": ["cobblestone", "stick"], "tool": "crafting_table"}
    {"thought": "Now I can mine iron ore using the stone pickaxe."}
    {"type": "mine", "output": "iron_ore", "quantity": 1, "tool": "stone_pickaxe"}
    {"thought": "Now I can craft a furnace using the cobblestone to smelt the iron ore."}
    {"type": "craft", "output": "furnace", "quantity": 1, "materials": ["cobblestone"]}
    {"thought": "Now I can smelt the iron ore using the furnace to obtain iron ingot."}
    {"type": "smelt", "output": "iron_ingot", "quantity": 1, "materials": ["iron_ore"], "tool": "furnace"}
]

The first argument of each line is a dict of items to be obtained, and the second argument is a dict of items to be used.
"""


REACT_SYSTEM_PROMPT = """You are a helper AI agent in Minecraft.
You need to decide on the next action to accomplish the planning TASK in Minecraft.

You must output a JSON object like the following:
{
    "output": "iron_ingot", # item to be produced
    "quantity": 1 # quantity to be produced
    "type": "smelt", # type of action
    "materials": ["iron_ore"] # optional list of materials
    "tool": "furnace" # optional tool
}

There are three types of actions
- mine
- craft
- smelt

To assist with planning, you first generate some thoughts before answering. For example:
{
"thought": "I need to gather some logs to craft planks using the diamond axe."
}

Do not generate errors. Always generate valid JSON objects following the format above.
"""

REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: How to obtain iron_ingot?\ninventory = {'diamond_axe': 1}""",
    },
    {
        "role": "assistant",
        "content": """{"thought": "To solve this task I need to smelt an iron ingot."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"thought": "First I need to gather some logs to craft planks using the diamond axe."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "mine", "output": "log", "quantity": 3, "tool": "diamond_axe"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'log': 3}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can craft planks using the logs. I need 11 planks to craft sticks and a crafting table."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft", "output": "planks", "quantity": 11, "materials": ["log"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'planks': 12}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can craft sticks using the planks. I need 4 sticks to craft a wooden pickaxe and then the stone pickaxe."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft", "output": "stick", "quantity": 4, "materials": ["planks"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'planks': 10, 'stick': 4}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can craft a crafting table using the planks."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft", "output": "crafting_table", "quantity": 1, "materials": ["planks"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'planks': 6, 'stick': 4, 'crafting_table': 1}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can craft a wooden pickaxe using the planks, sticks and crafting table."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft", "output": "wooden_pickaxe", "quantity": 1, "materials": ["planks", "stick"] "tool": "crafting_table"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'stick': 2, 'crafting_table': 1, 'wooden_pickaxe': 1}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I need to mine 11 cobblestone using the wooden pickaxe to be able to craft a stone pickaxe and a furnace."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "mine", "output": "cobblestone", "quantity": 11, "tool": "wooden_pickaxe"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'cobblestone': 11, 'stick': 2, 'crafting_table': 1, 'wooden_pickaxe': 1}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can craft a stone pickaxe using the cobblestone, sticks and crafting table."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft", "output": "stone_pickaxe", "quantity": 1, "materials": ["cobblestone", "stick"], "tool": "crafting_table"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'cobblestone': 8, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can mine iron ore using the stone pickaxe."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "mine", "output": "iron_ore", "quantity": 1, "tool": "stone_pickaxe"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'cobblestone': 8, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1, 'iron_ore': 3}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can craft a furnace using the cobblestone to smelt the iron ore."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "craft", "output": "furnace", "quantity": 1, "materials": ["cobblestone"]}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1, 'furnace': 1}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I can smelt the iron ore using the furnace to obtain iron ingot."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"type": "smelt", "output": "iron_ingot", "quantity": 1, "materials": ["iron_ore"], "tool": "furnace"}""",
    },
    {
        "role": "user",
        "content": "Success\ninventory = {'diamond_axe': 1, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stone_pickaxe': 1, 'furnace': 1, 'iron_ingot': 1}",
    },
    {
        "role": "assistant",
        "content": """{"thought": "I have successfully obtained iron ingot."}""",
    },
]
