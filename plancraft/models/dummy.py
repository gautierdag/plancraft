import random

from plancraft.environment.actions import (
    MoveAction,
)
from plancraft.models.base import PlancraftBaseModel


class DummyModel(PlancraftBaseModel):
    """
    Dummy model returns actions that do random action
    """

    def __init__(self, cfg=None):
        pass

    def reset(self):
        pass

    def random_select(self, observation):
        # randomly pick an item from the inventory
        item_indices = set()
        for slot, item in observation["inventory"].items():
            if item["quantity"] > 0:
                item_indices.add(slot)
        all_slots_to = set(range(1, 46))
        empty_slots = all_slots_to - item_indices

        random_slot_from = random.choice(list(item_indices))
        random_slot_to = random.choice(list(empty_slots))

        return MoveAction(
            slot_from=random_slot_from, slot_to=random_slot_to, quantity=1
        )

    def step(self, observation: dict, **kwargs) -> str:
        return str(self.random_select(observation))
