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
