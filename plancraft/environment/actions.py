import abc
import re
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


# [A1], [A2], [A3], [B1], [B2], [B3], [C1], [C2], [C3], [I1]-[I36]
SLOT_REGEX_PATTERN = r"\[([ABC][0-9]|I[1-9]|I[12][0-9]|I3[0-6])\]"
# [0], [A1], [A2], [A3], [B1], [B2], [B3], [C1], [C2], [C3], [I1]-[I36]
SLOT_REGEX_PATTERN_WITH_CRAFTING_SLOT = r"\[(0|[ABC][0-9]|I[1-9]|I[12][0-9]|I3[0-6])\]"
# 1-64
QUANTITY_REGEX_PATTERN = r"([1-9]|[1-5][0-9]|6[0-4])"


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
        # check if it is an inventory slot
        if len(slot) < 4 or len(slot) > 5 or slot[1] != "I" or slot[2] == "0":
            raise ValueError("Invalid slot index")
        # convert I1 to 10, I2 to 11, I3 to 12, ..., I36 to 45
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
    elif slot_index < 46:
        return f"[I{slot_index-9}]"
    raise ValueError("Invalid slot index")


class ActionHandlerBase(abc.ABC):
    @property
    @abc.abstractmethod
    def prompt_description(self) -> str:
        """
        Return the prompt description for the model
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prompt_format_example(self) -> str:
        """
        Return the prompt format example for the model
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def regex_pattern(self) -> str:
        """
        Return the regex pattern for the model
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def action_name(self) -> str:
        """
        Return the action name for the model
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def match(self, generated_text: str, **kwargs) -> Optional[BaseModel | str]:
        """
        Match the generated text to the action/tool
        """
        raise NotImplementedError()


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
                    "[Source] and [Target] must be [0] or [A1] to [C3] or [I1] to [I36]"
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
            raise AttributeError("[Source] and [Target] must be different")
        if self.slot_from < 0 or self.slot_from > 45:
            raise AttributeError(
                "[Source] must be [0] or [A1] to [C3] or [I1] to [I36]"
            )
        if self.slot_to == 0:
            raise AttributeError("You cannot move items into [0]")
        if self.slot_to < 1 or self.slot_to > 45:
            raise AttributeError("[Target] must be [A1] to [C3] or [I1] to [I36]")
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
                    "[Source] and [Target] must be [0] or [A1] to [C3] or [I1] to [I36]"
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
            raise AttributeError("[Source] and [Target] must be different")
        if self.slot_from < 0 or self.slot_from > 45:
            raise AttributeError(
                "[Source] must be [0] or [A1] to [C3] or [I1] to [I36]"
            )
        if self.slot_to == 0:
            raise AttributeError("You cannot smelt items into [0]")
        if self.slot_to < 1 or self.slot_to > 45:
            raise AttributeError("[Target] must be [A1] to [C3] or [I1] to [I36]")
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


class MoveActionHandler(ActionHandlerBase):
    @property
    def prompt_description(self) -> str:
        return "Transfer a specific quantity of an item from one slot to another"

    @property
    def prompt_format_example(self) -> str:
        return "`move: from [Source] to [Target] with quantity N`"

    @property
    def regex_pattern(self) -> str:
        return rf"move: from {SLOT_REGEX_PATTERN_WITH_CRAFTING_SLOT} to {SLOT_REGEX_PATTERN} with quantity {QUANTITY_REGEX_PATTERN}"

    @property
    def action_name(self) -> str:
        return "move"

    def match(self, generated_text: str, **kwargs) -> Optional[MoveAction | str]:
        """
        Parse the raw model response to a MoveAction
        """
        try:
            action_match = re.search(f"({self.action_name}):", generated_text)
            if not action_match:
                return
            slot_from = re.search(r" from (\[[ABCI]?\d+\])", generated_text).group(1)
            slot_to = re.search(r" to (\[[ABCI]?\d+\])", generated_text).group(1)
            quantity = re.search(r"with quantity (\d+)", generated_text).group(1)
            action = MoveAction(
                slot_from=slot_from,
                slot_to=slot_to,
                quantity=quantity,
            )
            return action
        except AttributeError as e:
            return f"Format Error: {e}. Correct format: {self.prompt_format_example}"


class SmeltActionHandler(ActionHandlerBase):
    @property
    def prompt_description(self) -> str:
        return "Smelt an item in a furnace and moves the output to a specific slot"

    @property
    def prompt_format_example(self) -> str:
        return "`smelt: from [Source] to [Target] with quantity N`"

    @property
    def regex_pattern(self) -> str:
        return rf"smelt: from {SLOT_REGEX_PATTERN} to {SLOT_REGEX_PATTERN} with quantity {QUANTITY_REGEX_PATTERN}"

    @property
    def action_name(self) -> str:
        return "smelt"

    def match(self, generated_text: str, **kwargs) -> Optional[SmeltAction | str]:
        """
        Parse the raw model response to a SmeltAction
        """

        try:
            action_match = re.search(f"({self.action_name}):", generated_text)
            if not action_match:
                return
            slot_from = re.search(r" from (\[[ABCI]?\d+\])", generated_text).group(1)
            slot_to = re.search(r" to (\[[ABCI]?\d+\])", generated_text).group(1)
            quantity = re.search(r"with quantity (\d+)", generated_text).group(1)
            action = SmeltAction(
                slot_from=slot_from,
                slot_to=slot_to,
                quantity=quantity,
            )
            return action
        except AttributeError as e:
            return f"Format Error: {e}. Correct format: {self.prompt_format_example}"


class ImpossibleActionHandler(ActionHandlerBase):
    @property
    def prompt_description(self) -> str:
        return "Stop task if it is certain that it is impossible with given inventory"

    @property
    def prompt_format_example(self) -> str:
        return "`impossible: <reason>`"

    @property
    def regex_pattern(self) -> str:
        return r"impossible: .+"

    @property
    def action_name(self) -> str:
        return "impossible"

    def match(self, generated_text, **kwargs) -> Optional[StopAction]:
        """
        Parse the raw model response to a StopAction
        """
        try:
            action_match = re.search(f"({self.action_name}):", generated_text)
            if not action_match:
                return
            reason = re.search(r"impossible: (.*)", generated_text).group(1)
            return StopAction(reason=reason)
        except AttributeError as e:
            return f"Format Error: {e}. Correct format: {self.prompt_format_example}"


class ThinkActionHandler(ActionHandlerBase):
    @property
    def prompt_description(self) -> str:
        return "Generate thoughts to help you decide on the next action"

    @property
    def prompt_format_example(self) -> str:
        return "`think: <thought message>`"

    @property
    def regex_pattern(self) -> str:
        return r"think: .+"

    @property
    def action_name(self) -> str:
        return "think"

    def match(self, generated_text, **kwargs) -> Optional[str]:
        """
        Parse the raw model response to a ThinkAction
        """
        try:
            action_match = re.search(f"({self.action_name}):", generated_text)
            if not action_match:
                return
            return "Ok"
        except AttributeError as e:
            return f"Format Error: {e}. Correct format: {self.prompt_format_example}"
