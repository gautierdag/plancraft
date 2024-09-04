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


class SymbolicMoveAction(BaseModel):
    """ "Moves an item from one slot to another"""

    slot_from: int
    slot_to: int
    quantity: int

    @field_validator("slot_from", "slot_to", "quantity", mode="before")
    def transform_str_to_int(cls, value) -> int:
        return int(value)

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

    @field_validator("slot_from", "slot_to", "quantity", mode="before")
    def transform_str_to_int(cls, value) -> int:
        return int(value)

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
    Acction that model can take to stop planning - decide impossible to continue
    """

    reason: str = ""


# when symbolic action is true, can either move objects around or smelt
SymbolicAction = SymbolicMoveAction | SymbolicSmeltAction

# when symbolic action is false, then need to use mouse to move things around, but can use smelt action
RealAction = RealActionInteraction | SymbolicSmeltAction


class PydanticSymbolicAction(BaseModel):
    root: SymbolicMoveAction | SymbolicSmeltAction
