from plancraft.config import EvalConfig
from plancraft.models.act import ActModel
from plancraft.models.base import ABCModel
from plancraft.models.dummy import DummyModel
from plancraft.models.oracle import OracleModel


def get_model(cfg: EvalConfig) -> ABCModel:
    """
    Factory get model (default: ActModel)
    """
    if cfg.plancraft.mode == "dummy":
        return DummyModel(cfg)
    elif cfg.plancraft.mode == "oracle":
        return OracleModel(cfg)
    return ActModel(cfg)
