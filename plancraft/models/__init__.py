from plancraft.models.base import ABCModel

from plancraft.config import Config
from plancraft.models.dummy import DummyModel
from plancraft.models.react import ReactModel
from plancraft.models.oracle import OracleModel


def get_model(cfg: Config, llm_model=None) -> ABCModel:
    """
    Factory get model
    """
    if cfg.plancraft.mode == "dummy":
        return DummyModel(cfg)
    if cfg.plancraft.mode == "oracle":
        return OracleModel(cfg)

    return ReactModel(cfg, llm=llm_model)
