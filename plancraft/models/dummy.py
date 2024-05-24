from plancraft.models.base import ABCModel, History

from plancraft.environments.actions import (
    SymbolicMoveAction,
    RealActionInteraction,
    SymbolicSmeltAction,
)
from plancraft.config import Config


class DummyModel(ABCModel):
    """
    Dummy model returns actions that do nothing - use to test
    """

    def __init__(self, cfg: Config):
        self.symbolic_move_action = cfg.plancraft.environment.symbolic_action_space
        self.histories = [
            History(objective="") for _ in range(cfg.plancraft.batch_size)
        ]

    def step(
        self, batch_observations: list[dict]
    ) -> list[SymbolicMoveAction | RealActionInteraction | SymbolicSmeltAction]:
        if self.symbolic_move_action:
            [SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)] * len(
                batch_observations
            )
        return [RealActionInteraction()] * len(batch_observations)
