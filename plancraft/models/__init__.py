from plancraft.models.base import ABCModel

from plancraft.config import Config
from plancraft.models.dummy import DummyModel
from plancraft.models.react import ReactModel


def get_model(cfg: Config, dummy=False) -> ABCModel:
    """
    Factory get model
    """
    if dummy:
        return DummyModel(
            symbolic_move_action=cfg.plancraft.environment.symbolic_action_space
        )
    return ReactModel(cfg)
