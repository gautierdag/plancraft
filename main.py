import logging
import warnings

import hydra

from plancraft.config import EvalConfig
from plancraft.evaluator import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def flatten_cfg(cfg):
    # for some reason hydra wraps file paths from config path
    if len(cfg) == 1:
        return flatten_cfg(cfg[list(cfg.keys())[0]])
    return cfg


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = EvalConfig(**flatten_cfg(dict(cfg)))
    evaluator = Evaluator(cfg)
    evaluator.eval_all()
    evaluator.close()


if __name__ == "__main__":
    main()
