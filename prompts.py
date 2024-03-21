ONE_SHOT_SYSTEM_PROMPT = """You are a helper AI agent in Minecraft.
You need to generate the sequences of sub-goals (actions) for a planning task in Minecraft.

You have the choice to use the following commands:
- mine
- craft
- smelt

You must answer with code.

Example Response:

```python
def obtain_iron_ingot(inventory):
    # first I need to mine 3 logs using the diamond_axe
    mine({'log': 3}, {'diamond_axe': 1})
    # then I can craft 11 planks using the logs
    craft({'planks': 11}, {'log': 3})
    # I will also need sticks, so I can craft 6 sticks using the planks
    craft({'stick': 6}, {'planks': 2})
    # I will then need a crafting table to craft the wooden_pickaxe
    craft({'crafting_table': 1}, {'planks': 4})
    # I can now craft a wooden_pickaxe using the planks, sticks and crafting_table
    craft({'wooden_pickaxe': 1}, {'planks': 3, 'stick': 2, 'crafting_table': 1})
    # I need to mine 11 cobblestone using the wooden_pickaxe
    mine({'cobblestone': 11}, {'wooden_pickaxe': 1})
    # Using the cobblestone, sticks and crafting_table, I can craft a stone_pickaxe
    craft({'stone_pickaxe': 1}, {'cobblestone': 3, 'stick': 2, 'crafting_table': 1})
    # I can now mine iron_ore using the stone_pickaxe
    mine({'iron_ore': 1}, {'stone_pickaxe': 1})
    # I need to craft a furnace using the cobblestone
    craft({'furnace': 1}, {'cobblestone': 8})
    # Finally, I can smelt the iron_ore using the furnace to obtain iron_ingot
    smelt({'iron_ingot': 1}, {'iron_ore': 1, 'furnace': 1})
    return 'iron_ingot
```

The first argument of each line is a dict of items to be obtained, and the second argument is a dict of items to be used.
"""

ONE_SHOT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: How to obtain iron_ingot?\ninventory = {'diamond_axe'}""",
    },
    {
        "role": "assistant",
        "content": """def obtain_iron_ingot(inventory):
    # first I need to mine 3 logs using the diamond_axe
    mine({'log': 3}, {'diamond_axe': 1})
    # then I can craft 11 planks using the logs
    craft({'planks': 11}, {'log': 3})
    # I will also need sticks, so I can craft 6 sticks using the planks
    craft({'stick': 6}, {'planks': 2})
    # I will then need a crafting table to craft the wooden_pickaxe
    craft({'crafting_table': 1}, {'planks': 4})
    # I can now craft a wooden_pickaxe using the planks, sticks and crafting_table
    craft({'wooden_pickaxe': 1}, {'planks': 3, 'stick': 2, 'crafting_table': 1})
    # I need to mine 11 cobblestone using the wooden_pickaxe
    mine({'cobblestone': 11}, {'wooden_pickaxe': 1})
    # Using the cobblestone, sticks and crafting_table, I can craft a stone_pickaxe
    craft({'stone_pickaxe': 1}, {'cobblestone': 3, 'stick': 2, 'crafting_table': 1})
    # I can now mine iron_ore using the stone_pickaxe
    mine({'iron_ore': 1}, {'stone_pickaxe': 1})
    # I need to craft a furnace using the cobblestone
    craft({'furnace': 1}, {'cobblestone': 8})
    # Finally, I can smelt the iron_ore using the furnace to obtain iron_ingot
    smelt({'iron_ingot': 1}, {'iron_ore': 1, 'furnace': 1})
    return 'iron_ingot""",
    },
]


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

There are four types of actions
- think
- mine
- craft
- smelt

If you wish to think, you can output a string with your thought. For example:
{
"thought": "I need to gather some logs to craft planks using the diamond_axe."
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
        "content": """{"thought": "To solve this task I need to smelt an iron_ingot."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"thought": "First I need to gather some logs to craft planks using the diamond_axe."}""",
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
        "content": """{"thought": "Now I can craft planks using the logs. I need 11 planks to craft sticks and a crafting_table."}""",
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
        "content": """{"thought": "Now I can craft sticks using the planks. I need 4 sticks to craft a wooden_pickaxe and then the stone_pickaxe."}""",
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
        "content": """{"thought": "Now I can craft a crafting_table using the planks."}""",
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
        "content": """{"thought": "Now I can craft a wooden_pickaxe using the planks, sticks and crafting_table."}""",
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
        "content": """{"thought": "Now I need to mine 11 cobblestone using the wooden_pickaxe to be able to craft a stone_pickaxe and a furnace."}""",
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
        "content": """{"thought": "Now I can craft a stone_pickaxe using the cobblestone, sticks and crafting_table."}""",
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
        "content": """{"thought": "Now I can mine iron_ore using the stone_pickaxe."}""",
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
        "content": """{"thought": "Now I can craft a furnace using the cobblestone to smelt the iron_ore."}""",
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
        "content": """{"thought": "Now I can smelt the iron_ore using the furnace to obtain iron_ingot."}""",
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
        "content": """{"thought": "I have successfully obtained iron_ingot."}""",
    },
]
