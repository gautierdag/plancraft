import random

from plancraft.environment.actions import (
    MoveAction,
)
from plancraft.models.base import PlancraftBaseModel, PlancraftModelOutput


class DummyModel(PlancraftBaseModel):
    """
    Dummy model returns actions that do random action
    """

    def __init__(self, cfg=None):
        pass

    def reset(self):
        pass

    def random_select(self, observation):
        # randomly pick an item that has quantity 1 from the inventory
        item_indices = set()
        for slot, item in observation["inventory"].items():
            if item["quantity"] == 1:
                item_indices.add(slot)
        all_slots_to = set(range(1, 46))
        empty_slots = all_slots_to - item_indices

        # if not item with quantity == 1, randomly pick any item
        if len(item_indices) == 0:
            item_indices = set(observation["inventory"].keys())

        # move the item to a random empty slot
        random_slot_from = random.choice(list(item_indices))
        random_slot_to = random.choice(list(empty_slots))

        return MoveAction(
            slot_from=random_slot_from, slot_to=random_slot_to, quantity=1
        )

    def step(self, observation: dict, **kwargs) -> PlancraftModelOutput:
        return PlancraftModelOutput(action=str(self.random_select(observation)))

    def batch_step(
        self, observations: list[dict], **kwargs
    ) -> list[PlancraftModelOutput]:
        return [self.step(observation) for observation in observations]
