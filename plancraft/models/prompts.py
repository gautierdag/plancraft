REACT_SYSTEM_PROMPT = """You are crafting in Minecraft.
You need to decide on the next action.

You must output an action in JSON format like the following:
{
    "slot_from": 1,
    "slot_to": 1, 
    "quantity": 1 
}

There are two types of actions
- move
- smelt

To assist with planning, you first generate some thoughts before answering. For example:
{
    "thought": "To craft an acacia_fence, I first need to craft acacia_planks."
}

The first 10 slots in the inventory are reserved for crafting. 

[1, 2, 3] 
[4, 5, 6] -> [0]
[7, 8, 9]

The crafting matrix is a 3x3 grid, and the output is sent to slot 0.
You cannot move items into output slot 0.
The remaining slots (10-46) are for storing items.

Do not generate errors. Always generate valid JSON objects following the format above.
"""

REACT_EXAMPLE = [
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory='[{"slot": 27,"type": "diorite", "quantity": 1},{"slot": 39,"type": "cobblestone", "quantity": 1}]'""",
    },
    {
        "role": "assistant",
        "content": """{"thought": "To solve this task I need to craft andesite using 1 diorite and 1 cobblestone side by side."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"slot_from": 27, "slot_to": 4, "quantity": 1}""",
    },
    {
        "role": "user",
        "content": """TASK: Craft an item of type: andesite\ninventory=[{"slot": 4, "type": "diorite", "quantity": 1},{"slot": 39, "type": "cobblestone", "quantity": 1}]""",
    },
    {
        "role": "assistant",
        "content": """{"thought": "Now I need to move the cobblestone into position."}""",
    },
    {"role": "user", "content": "OK"},
    {
        "role": "assistant",
        "content": """{"slot_from": 39, "slot_to": 5, "quantity": 1}""",
    },
]
