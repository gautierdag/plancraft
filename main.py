import logging
import warnings

import hydra

from plancraft.config import EvalConfig
from plancraft.evaluator import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = EvalConfig(**dict(cfg))
    evaluator = Evaluator(cfg)
    evaluator.eval_all()
    evaluator.close_envs()


if __name__ == "__main__":
    main()
