from pydantic import BaseModel, field_validator, model_validator


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


class MoveAction(BaseModel):
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
        return self

    def __str__(self):
        return f"move: from {convert_from_slot_index(self.slot_from)} to {convert_from_slot_index(self.slot_to)} with quantity {self.quantity}"


class SmeltAction(BaseModel):
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
        return self

    def __str__(self):
        return f"smelt: from {convert_from_slot_index(self.slot_from)} to {convert_from_slot_index(self.slot_to)} with quantity {self.quantity}"


class StopAction(BaseModel):
    """
    Action that model can take to stop planning - decide impossible to continue
    Note: also known as the "impossible" action
    """

    reason: str = ""

    def __str__(self):
        return f"impossible: {self.reason}"


# when symbolic action is true, can either move objects around or smelt
SymbolicAction = MoveAction | SmeltAction
