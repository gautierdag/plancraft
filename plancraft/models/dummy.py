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
        out_actions = []
        for observation, history in zip(batch_observations, self.histories):
            if observation is None:
                out_actions.append(None)
                continue

            # add observation to history
            history.add_observation_to_history(observation)

            # get action
            if self.symbolic_move_action:
                action = SymbolicMoveAction(slot_from=0, slot_to=0, quantity=1)
            else:
                action = RealActionInteraction()
            out_actions.append(action)

            # add action to history
            history.add_action_to_history(action)

        return out_actions
