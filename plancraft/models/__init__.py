from plancraft.models.base import ABCModel

from plancraft.config import EvalConfig
from plancraft.models.dummy import DummyModel
from plancraft.models.react import ReactModel
from plancraft.models.oracle import OracleModel
from plancraft.models.act import ActModel


def get_model(cfg: EvalConfig) -> ABCModel:
    """
    Factory get model (default: ReactModel)
    """
    if cfg.plancraft.mode == "dummy":
        return DummyModel(cfg)
    elif cfg.plancraft.mode == "oracle":
        return OracleModel(cfg)
    elif cfg.plancraft.mode == "act":
        return ActModel(cfg)
    else:
        return ReactModel(cfg)
