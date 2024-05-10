REACT_SYSTEM_PROMPT = """You are a helper AI agent in Minecraft.
You need to decide on the next action to accomplish the crafting TASK in Minecraft.

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
        "content": """{"thought": "I have successfully obtained iron ingot."}""",
    },
]
