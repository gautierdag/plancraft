import torch

from plancraft.config import EvalConfig
from plancraft.environment.planner import get_subplans
from plancraft.models.base import PlancraftBaseModel
from plancraft.models.bbox_model import IntegratedBoundingBoxModel


class OracleModel(PlancraftBaseModel):
    """
    Oracle model returns actions that solve the task optimally
    """

    def __init__(self, cfg: EvalConfig):
        self.subplans = []
        self.use_fasterrcnn = cfg.plancraft.use_fasterrcnn

        self.bbox_model = None
        if self.use_fasterrcnn:
            # fasterrcnn is not multimodal model but a separate model
            self.bbox_model = IntegratedBoundingBoxModel.from_pretrained(
                "gautierdag/plancraft-fasterrcnn"
            )
            self.bbox_model.eval()
            if torch.cuda.is_available():
                self.bbox_model.cuda()

    def reset(self):
        self.subplans = []

    def step(self, observation: dict, **kwargs) -> str:
        # get action
        if len(self.subplans) == 0:
            subplans, _ = get_subplans(observation)
            # flatten subplans since they are nested for each subgoal
            flattened_subplans = [item for sublist in subplans for item in sublist]
            self.subplans = flattened_subplans

        action = self.subplans.pop(0)
        return action

    def batch_step(self, observations: list[dict], **kwargs) -> list:
        # Need to fully isolate state between examples
        actions = []
        for observation in observations:
            self.reset()
            action = self.step(observation)
            actions.append(action)
        return actions
