import random

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    RealActionInteraction,
    SymbolicMoveAction,
    SymbolicSmeltAction,
)
from plancraft.models.base import ABCModel, History


class DummyModel(ABCModel):
    """
    Dummy model returns actions that do random action
    """

    def __init__(self, cfg: EvalConfig):
        self.symbolic_move_action = cfg.plancraft.environment.symbolic_action_space
        self.histories = [
            History(objective="") for _ in range(cfg.plancraft.batch_size)
        ]

    def random_select(self, observation):
        if observation is None or "inventory" not in observation:
            return SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)
        # randomly pick an item from the inventory
        item_indices = set()
        for item in observation["inventory"]:
            if item["quantity"] > 0:
                item_indices.add(item["index"])
        all_slots_to = set(range(1, 46))
        empty_slots = all_slots_to - item_indices

        random_slot_from = random.choice(list(item_indices))
        random_slot_to = random.choice(list(empty_slots))

        return SymbolicMoveAction(
            slot_from=random_slot_from, slot_to=random_slot_to, quantity=1
        )

    def step(
        self, batch_observations: list[dict]
    ) -> list[SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction]:
        out_actions = []
        for observation, history in zip(batch_observations, self.histories):
            if observation is None:
                out_actions.append(None)
                continue

            # add observation to history
            history.add_observation_to_history(observation)

            # get action
            if self.symbolic_move_action:
                action = self.random_select(observation)
            else:
                action = RealActionInteraction()
            out_actions.append(action)

            # add action to history
            history.add_action_to_history(action)

        return out_actions
