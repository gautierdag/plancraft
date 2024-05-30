import logging
import warnings

import hydra

from plancraft.config import Config
from plancraft.evaluator import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = Config(**dict(cfg))
    evaluator = Evaluator(cfg)
    evaluator.eval_all()


if __name__ == "__main__":
    main()
