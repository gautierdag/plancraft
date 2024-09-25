import json

from pydantic import BaseModel, model_validator, field_validator

import numpy as np
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.agent.action import Action


class InventoryCommandAction(Action):
    """
    Handler which lets agents programmatically interact with an open container

    Using this - agents can move a chosen quantity of items from one slot to another.
    """

    def to_string(self):
        return "inventory_command"

    def xml_template(self) -> str:
        return str("<InventoryCommands/>")

    def __init__(self):
        self._command = "inventory_command"
        # first argument is the slot to take from
        # second is the slot to put into
        # third is the count to take
        super().__init__(
            self.command,
            spaces.Tuple(
                (
                    spaces.Discrete(46),
                    spaces.Discrete(46),
                    spaces.Discrete(64),
                )
            ),
        )

    def from_universal(self, x):
        return np.array([0, 0, 0], dtype=np.int32)


class SmeltCommandAction(Action):
    """
    An action handler for smelting an item
    We assume smelting is immediate.
    @TODO: might be interesting to explore using the smelting time as an additional planning parameter.

    Using this agents can smelt items in their inventory.
    """

    def __init__(self):
        self._command = "smelt"
        # first argument is the slot to take from
        # second is the slot to put into
        # third is the count to smelt
        super().__init__(
            self.command,
            spaces.Tuple(
                (
                    spaces.Discrete(46),
                    spaces.Discrete(46),
                    spaces.Discrete(64),
                )
            ),
        )

    def to_string(self):
        return "smelt"

    def xml_template(self) -> str:
        return str("<SmeltCommands/>")

    def from_universal(self, x):
        return np.array([0, 0, 0], dtype=np.int32)


class InventoryResetAction(Action):
    def __init__(self):
        self._command = "inventory_reset"
        super().__init__(self._command, spaces.Text([1]))

    def to_string(self) -> str:
        return "inventory_reset"

    def to_hero(self, inventory_items: list[dict]):
        return "{} {}".format(self._command, json.dumps(inventory_items))

    def xml_template(self) -> str:
        return "<InventoryResetCommands/>"

    def from_universal(self, x):
        return []


def convert_to_slot_index(slot: str) -> int:
    slot = slot.strip()
    grid_map = {
        "[0]": 0,
        "[A1]": 1,
        "[A2]": 2,
        "[A3]": 3,
        "[B1]": 4,
        "[B2]": 5,
        "[B3]": 6,
        "[C1]": 7,
        "[C2]": 8,
        "[C3]": 9,
    }
    if slot in grid_map:
        return grid_map[slot]
    else:
        return int(slot[2:-1]) + 9


def convert_from_slot_index(slot_index: int) -> str:
    grid_map = {
        0: "[0]",
        1: "[A1]",
        2: "[A2]",
        3: "[A3]",
        4: "[B1]",
        5: "[B2]",
        6: "[B3]",
        7: "[C1]",
        8: "[C2]",
        9: "[C3]",
    }
    if slot_index < 10:
        return grid_map[slot_index]
    else:
        return f"[I{slot_index-9}]"


class SymbolicMoveAction(BaseModel):
    """ "Moves an item from one slot to another"""

    slot_from: int
    slot_to: int
    quantity: int
    action_type: str = "move"

    @field_validator("action_type", mode="before")
    def fix_action_type(cls, value) -> str:
        return "move"

    @field_validator("slot_from", "slot_to", mode="before")
    def transform_str_to_int(cls, value) -> int:
        # if value is a string like [A1] or [I1], convert it to an integer
        if isinstance(value, str):
            try:
                return convert_to_slot_index(value)
            except ValueError:
                raise AttributeError(
                    "slot_from and slot_to must be [0] or [A1] to [C3] or [I1] to [I36]"
                )
        return value

    @field_validator("quantity", mode="before")
    def transform_quantity(cls, value) -> int:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise AttributeError("quantity must be an integer")
        return value

    @model_validator(mode="after")
    def validate(self):
        if self.slot_from == self.slot_to:
            raise AttributeError("slot_from and slot_to must be different")
        if self.slot_from < 0 or self.slot_from > 45:
            raise AttributeError("slot_from must be between 0 and 45")
        if self.slot_to < 1 or self.slot_to > 45:
            raise AttributeError("slot_to must be between 1 and 45")
        if self.quantity < 1 or self.quantity > 64:
            raise AttributeError("quantity must be between 1 and 64")

    def to_action_dict(self) -> dict:
        return {
            "inventory_command": [self.slot_from, self.slot_to, self.quantity],
        }


class SymbolicSmeltAction(BaseModel):
    """Smelts an item and moves the result into a new slot"""

    slot_from: int
    slot_to: int
    quantity: int
    action_type: str = "smelt"

    @field_validator("action_type", mode="before")
    def fix_action_type(cls, value) -> str:
        return "smelt"

    @field_validator("slot_from", "slot_to", mode="before")
    def transform_str_to_int(cls, value) -> int:
        # if value is a string like [A1] or [I1], convert it to an integer
        if isinstance(value, str):
            try:
                return convert_to_slot_index(value)
            except ValueError:
                raise AttributeError(
                    "slot_from and slot_to must be [0] or [A1] to [C3] or [I1] to [I36]"
                )
        return value

    @field_validator("quantity", mode="before")
    def transform_quantity(cls, value) -> int:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise AttributeError("quantity must be an integer")
        return value

    @model_validator(mode="after")
    def validate(self):
        if self.slot_from == self.slot_to:
            raise AttributeError("slot_from and slot_to must be different")
        if self.slot_from < 0 or self.slot_from > 45:
            raise AttributeError("slot_from must be between 0 and 45")
        if self.slot_to < 1 or self.slot_to > 45:
            raise AttributeError("slot_to must be between 1 and 45")
        if self.quantity < 1 or self.quantity > 64:
            raise AttributeError("quantity must be between 1 and 64")

    def to_action_dict(self) -> dict:
        return {
            "smelt": [self.slot_from, self.slot_to, self.quantity],
        }


class ThinkAction(BaseModel):
    """Think about the answer before answering"""

    thought: str

    def to_action_dict(self) -> dict:
        return {}


class SearchAction(BaseModel):
    """Searches for a relevant document in the wiki"""

    search_string: str

    def to_action_dict(self) -> dict:
        return {
            "search": self.search_string,
        }


class RealActionInteraction(BaseModel):
    mouse_direction_x: float = 0
    mouse_direction_y: float = 0
    right_click: bool = False
    left_click: bool = False

    @field_validator("mouse_direction_x", "mouse_direction_y")
    def prevent_zero(cls, v):
        if v > 10:
            return 10
        elif v < -10:
            return -10
        return v

    def to_action_dict(self) -> dict:
        return {
            "camera": [self.mouse_direction_x, self.mouse_direction_y],
            "use": int(self.right_click),
            "attack": int(self.left_click),
        }


class StopAction(BaseModel):
    """
    Action that model can take to stop planning - decide impossible to continue
    Note: also known as the "impossible" action
    """

    reason: str = ""


class NoOp(SymbolicMoveAction):
    """No operation action - special instance of move"""

    def __init__(self):
        super().__init__(slot_from=0, slot_to=1, quantity=1)
        self.slot_to = 0

    def __call__(self, *args, **kwargs):
        return None

    def __str__(self):
        return "NoOp"


# when symbolic action is true, can either move objects around or smelt
SymbolicAction = SymbolicMoveAction | SymbolicSmeltAction

# when symbolic action is false, then need to use mouse to move things around, but can use smelt action
RealAction = RealActionInteraction | SymbolicSmeltAction


class PydanticSymbolicAction(BaseModel):
    root: SymbolicMoveAction | SymbolicSmeltAction
