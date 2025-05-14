import base64
import csv
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from loguru import logger
from PIL import Image as PILImage
from plancraft.config import PlancraftExample
from plancraft.environment.actions import (
    MoveAction,
    SmeltAction,
    StopAction,
)
from plancraft.environment.env import (
    PlancraftEnvironment,
    get_objective_str,
    target_and_inventory_to_text_obs,
)
from plancraft.simple import get_plancraft_examples

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent


class PlancraftMCPWrapper:
    def __init__(
        self,
        example: PlancraftExample,
        max_steps: int = 30,
        resolution: str = "high",
        use_text_inventory: bool = True,
    ):
        self.max_steps = max_steps
        # whether to convert the inventory to text observation
        # if False, only the objective string is returned
        self.use_text_inventory = use_text_inventory
        self.current_step = 0
        self.stopped = False
        self.success = False
        self.example = example
        self.resolution = resolution
        self.environment = PlancraftEnvironment(
            example.slotted_inventory, resolution=self.resolution
        )

    def check_done(self, inventory: dict, target: str):
        """
        Check that target object is obtained
        """
        for slot, item in inventory.items():
            # ensure the target is in the inventory (not in slot 0)
            if target == item["type"] and slot != 0:
                return True
        return False

    def step(
        self, action: Optional[StopAction | MoveAction | SmeltAction] = None
    ) -> tuple[dict[str, Any], bool]:
        # Handle already stopped case
        if self.stopped:
            return (
                {"text": "Plancraft environment is terminated"},
                True,
            )

        # Handle initial step
        if not action:
            observation = self.environment.step()
            observation["target"] = self.example.target
            if self.use_text_inventory:
                text = target_and_inventory_to_text_obs(
                    target=self.example.target, inventory=observation["inventory"]
                )
            else:
                text = get_objective_str(self.example.target)
            observation["text"] = text
            return observation, self.stopped

        # Handle max steps reached
        if self.current_step > self.max_steps:
            self.stopped = True
            return (
                {"text": f"Max steps ({self.max_steps}) reached"},
                self.stopped,
            )

        self.current_step += 1
        # Handle stop action
        if isinstance(action, StopAction):
            self.stopped = True
            #  success is True if example was truly impossible
            self.success = self.example.impossible
            observation = {
                "text": "Plancraft environment is terminate due to stop action"
            }
        else:
            observation = self.environment.step(action)
            observation["target"] = self.example.target

            # Generate text observation
            if self.use_text_inventory:
                text = target_and_inventory_to_text_obs(
                    target=self.example.target, inventory=observation["inventory"]
                )
            else:
                text = get_objective_str(self.example.target)

            observation["text"] = text

            self.success = self.check_done(
                observation["inventory"], self.example.target
            )

        return (
            observation,
            self.stopped,
        )


@dataclass
class PlancraftContext:
    """Context for the Plancraft environment."""

    env: Optional[Any] = None
    examples: list[PlancraftExample] = field(default_factory=list)

    # Default environment settings
    max_steps: int = 30
    resolution: Literal["high", "low"] = "high"
    use_text_inventory: bool = True
    terminated: bool = False
    use_images: bool = False

    # history and statistics
    history: list[dict] = field(default_factory=list)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[PlancraftContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    try:
        logger.info("Starting up")
        examples = get_plancraft_examples(split="train")
        yield PlancraftContext(examples=examples)
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down")


# Pass lifespan to server
app = FastMCP("Plancraft", lifespan=app_lifespan)


@app.prompt()
def plancraft_environment_instructions() -> str:
    return """You are crafting in Minecraft. You need to decide on the next action.

Crafting Grid: The crafting table is organized into a 3x3 grid. Each slot in the grid has a unique identifier:
    - Top row: [A1] [A2] [A3]
    - Middle row: [B1] [B2] [B3]
    - Bottom row: [C1] [C2] [C3]

The output of the crafting process is placed in a designated output slot labeled [0] You cannot move or smelt items directly into slot [0]

Inventory Slots: The remaining inventory slots (outside of the crafting grid) are used for storing items. These slots are labeled as [I1] to [I36]

Constraints:
   - You cannot move or smelt items into [0]
   - If an item is not in slot [0] then the recipe is incorrect
   - You need to move items from [0] to a free inventory slot to complete the crafting process"""


def observation_to_tool_result(
    observation: dict,
    terminated: bool = False,
    success: bool = False,
    add_instructions=False,
    use_images=False,
) -> CallToolResult:
    content = []
    if add_instructions:
        instructions = plancraft_environment_instructions()
        content.append(TextContent(text=instructions, type="text"))

    text_content = observation["text"]
    # Add success message if the task was completed successfully
    if success:
        text_content = f"SUCCESS! You have completed the task: {text_content}"
    # Add termination message if the task was terminated but not successful
    elif terminated:
        text_content = f"Task terminated: {text_content}"

    content.append(TextContent(text=text_content, type="text"))

    if use_images:
        # numpy array to PIL
        pil_image = PILImage.fromarray(observation["image"])
        # Save the image to a BytesIO buffer in PNG format
        import io

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode the properly formatted PNG data
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        content.append(
            ImageContent(type="image", data=base64_data, mimeType="image/png")
        )
    return CallToolResult(
        content=content,
    )


def add_action_to_history(
    action: MoveAction | SmeltAction | StopAction, result: CallToolResult, ctx: Context
) -> None:
    """Add action call to history - storing only text content for simplicity"""
    # Extract only the text content from the result
    text_content = ""
    for content in result.content:
        if hasattr(content, "text"):
            text_content = content.text
            break
    ctx.request_context.lifespan_context.history.append(
        {"action": str(action), "observation": text_content}
    )


def save_result_if_terminated(environment, terminated: bool, ctx: Context) -> None:
    """Helper function to save results if the environment is terminated"""
    if terminated:
        logger.info(f"Task completed or terminated in {environment.current_step} steps")

        result_data = {
            "example_id": environment.example.id,
            "success": environment.success,
            "steps": environment.current_step,
            "history": ctx.request_context.lifespan_context.history.copy(),
        }

        # save the result data to a CSV file
        pwd = os.path.dirname(os.path.abspath(__file__))
        os.makedirs("results", exist_ok=True)
        result_file = os.path.join(pwd, "results", "results.csv")
        with open(result_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(result_data)

        # Reset history after storing results
        ctx.request_context.lifespan_context.history = []


def correct_slot_format(slot: str):
    # helper function to correct the slot format
    # Claude seems unable to generate slots with brackets
    if "[" not in slot and "]" not in slot:
        return f"[{slot}]"


@app.tool(
    name="smelt",
    description="Smelt items in the Plancraft environment. You must specify the slot to smelt from and the slot to smelt to and quantity.",
)
def smelt(from_slot: str, to_slot: str, quantity: int, ctx: Context) -> CallToolResult:
    """Smelt items in the Plancraft environment. You must specify the slot to smelt from and the slot to smelt to and quantity."""
    try:
        environment = ctx.request_context.lifespan_context.env
        use_images = ctx.request_context.lifespan_context.use_images

        if environment.stopped:
            content = "Plancraft environment is terminated, first call start_new_task to start a new task"
            return CallToolResult(
                content=[TextContent(text=content, type="text")], isError=True
            )

        smelt_action = SmeltAction(
            slot_from=correct_slot_format(from_slot),
            slot_to=correct_slot_format(to_slot),
            quantity=quantity,
        )
        obs, terminated = environment.step(smelt_action)

        # Generate the result with appropriate success/termination messages
        result = observation_to_tool_result(
            obs,
            terminated=terminated,
            success=environment.success,
            use_images=use_images,
        )
        add_action_to_history(smelt_action, result, ctx)
        logger.info(f"Step {environment.current_step}: {obs['text']}")

        # Save result if the task is terminated
        save_result_if_terminated(environment, terminated, ctx)

        return result
    except Exception as e:
        return CallToolResult(
            content=[TextContent(text=str(e), type="text")], isError=True
        )


@app.tool(
    name="move",
    description="Move items in the Plancraft environment. You must specify the slot to move from and the slot to move to and quantity.",
)
def move(from_slot: str, to_slot: str, quantity: int, ctx: Context) -> CallToolResult:
    """
    Move items in the Plancraft environment. You must specify the slot to move from and the slot to move to and quantity.
    """
    try:
        environment = ctx.request_context.lifespan_context.env
        use_images = ctx.request_context.lifespan_context.use_images
        if environment.stopped:
            content = "Plancraft environment is terminated, first call start_new_task to start a new task"
            return CallToolResult(
                content=[TextContent(text=content, type="text")], isError=True
            )

        move_action = MoveAction(
            slot_from=correct_slot_format(from_slot),
            slot_to=correct_slot_format(to_slot),
            quantity=quantity,
        )
        obs, terminated = environment.step(move_action)

        # Generresult with appropriate success/termination messages
        result = observation_to_tool_result(
            obs,
            terminated=terminated,
            success=environment.success,
            use_images=use_images,
        )
        add_action_to_history(move_action, result, ctx)
        logger.info(f"Step {environment.current_step}: {obs['text']}")

        # Save result if the task is terminated
        save_result_if_terminated(environment, terminated, ctx)

        return result

    except Exception as e:
        return CallToolResult(
            content=[TextContent(text=str(e), type="text")], isError=True
        )


@app.tool(
    name="impossible",
    description="Declare the current task impossible. This will end the current task.",
)
def impossible_task(reason: str, ctx: Context) -> CallToolResult:
    """Declare the current task impossible. This will end the current task."""
    try:
        environment = ctx.request_context.lifespan_context.env
        use_images = ctx.request_context.lifespan_context.use_images
        stop_action = StopAction(reason=reason)
        obs, terminated = environment.step(stop_action)

        # Generate the ith appropriate success/termination messages
        # For impossible action, it's successful if the task was truly impossible
        result = observation_to_tool_result(
            obs,
            terminated=terminated,
            success=environment.success,
            use_images=use_images,
        )
        add_action_to_history(stop_action, result, ctx)
        logger.info(f"Step {environment.current_step}: {obs['text']}")

        # Save result if the task is terminated
        save_result_if_terminated(environment, terminated, ctx)

        return result
    except Exception as e:
        return CallToolResult(
            content=[TextContent(text=str(e), type="text")], isError=True
        )


@app.tool(
    name="start_plancraft_task",
    description="Start a new Plancraft environment (default use_images=False)",
)
def start_plancraft(use_images: bool, ctx: Context) -> CallToolResult:
    """Tool that uses initialized resources"""
    # Check if there's an existing environment and if it wasn't terminated yet
    current_env = ctx.request_context.lifespan_context.env
    if current_env and not current_env.stopped:
        # Save the current environment state in results before discarding
        logger.info("Discarding existing incomplete task")
        # Reset history when starting a new task
        ctx.request_context.lifespan_context.history = []

    ctx.request_context.lifespan_context.use_images = use_images

    random_idx = random.randint(
        0, len(ctx.request_context.lifespan_context.examples) - 1
    )
    example: PlancraftExample = ctx.request_context.lifespan_context.examples[
        random_idx
    ]
    # initialize the environment
    env = PlancraftMCPWrapper(
        example=example,
        max_steps=ctx.request_context.lifespan_context.max_steps,
        resolution=ctx.request_context.lifespan_context.resolution,
        use_text_inventory=ctx.request_context.lifespan_context.use_text_inventory,
    )
    logger.info(f"Environment initialized with example {example.id}")
    ctx.request_context.lifespan_context.env = env

    obs, _ = env.step()
    logger.info(f"Step {env.current_step}: {obs['text']}")

    result = observation_to_tool_result(obs, add_instructions=True)
    return result


if __name__ == "__main__":
    app.run()
