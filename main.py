import warnings

import hydra

from loguru import logger
from plancraft.config import EvalConfig
from plancraft.evaluator import Evaluator

warnings.filterwarnings("ignore")


def flatten_cfg(cfg):
    # for some reason hydra wraps file paths from config path
    if len(cfg) == 1:
        return flatten_cfg(cfg[list(cfg.keys())[0]])
    return cfg


@hydra.main(config_path="configs", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = EvalConfig(**flatten_cfg(dict(cfg)))
    evaluator = Evaluator(cfg)
    evaluator.eval_all_seeds()


if __name__ == "__main__":
    main()
