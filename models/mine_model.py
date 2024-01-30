import random

import torch
from torch.distributions import Categorical
import numpy as np

from models.goal_model import get_goal_model
from models.mineclip import MineCLIP
from models.utils import get_torch_device


def accquire_goal_embeddings(clip_path: str, goal: str):
    clip_cfg = {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": [160, 256],
    }
    clip_model = MineCLIP(**clip_cfg)
    clip_model.load_ckpt(clip_path, strict=True)
    device = get_torch_device()
    clip_model = clip_model.to(device)
    with torch.no_grad():
        return clip_model.encode_text([goal]).cpu().numpy()


class MineAgent:
    """
    Shell agent for goal: mine_cobblestone, mine_stone, mine_coal, mine_iron_ore, mine_diamond
    """

    def __init__(
        self,
        cfg: dict,
        no_op: np.array = np.array([0, 0, 0, 5, 5, 0, 0, 0]),
        max_ranking: int = 15,
    ) -> None:
        self.script_goals = ["cobblestone", "stone", "coal", "iron_ore", "diamond"]

        self.no_op = no_op  # no_op is numpy array
        action_space = [3, 3, 4, 11, 11, 8, 1, 1]
        self.model = get_goal_model(cfg, action_space)
        self.model.eval()
        self.max_ranking = max_ranking

        # rely_goals = [val for val in self.goal_mapping_dict.values()]
        self.embedding_dict = {}
        self.clip_path = cfg["pretrains"]["clip_path"]

    def get_action(self, goal: str, states: dict) -> tuple[int, torch.Tensor]:
        if goal in self.script_goals:
            print(f"[INFO]: goal is {goal}")
            act = self.no_op.copy()
            if random.randint(0, 20) == 0:
                act[4] = 1
            if random.randint(0, 20) == 0:
                act[0] = 1
            if goal in ["stone", "coal", "cobblestone"]:
                if states["compass"][-1][1] < 83:
                    act[3] = 9
                    return self.max_ranking, act
                else:
                    act[5] = 3
                    return self.max_ranking, act
            elif goal in ["iron_ore", "diamond"]:
                if goal == "iron_ore":
                    depth = 30
                elif goal == "diamond":
                    depth = 10
                if states["gps"][-1][1] * 100 > depth:
                    if states["compass"][-1][1] < 80:
                        act[3] = 9
                        return self.max_ranking, act
                    else:
                        act[5] = 3
                        return self.max_ranking, act
                else:
                    if states["compass"][-1][1] > 50:
                        act[3] = 1
                        return self.max_ranking, act
                    elif states["compass"][-1][1] < 40:
                        act[3] = 9
                        return self.max_ranking, act
                    else:
                        act[0] = 1
                        act[5] = 3
                        return self.max_ranking, act
            else:
                raise NotImplementedError
        else:
            if goal not in self.embedding_dict:
                # accquire goal embedding
                self.embedding_dict[goal] = accquire_goal_embeddings(
                    self.clip_path, goal
                )

            goal_embedding = self.embedding_dict[goal]
            goals = torch.from_numpy(goal_embedding).repeat(
                len(states["prev_action"]), 1
            )

            # Neural Network Agent
            action_preds, mid_info = self.model.get_action(
                goals=goals,
                states=states,
                horizons=None,
            )

            # Split the action prediction tensor into separate tensors for each categorical action.
            # TODO: BROKEN
            split_action_preds = torch.split(
                action_preds[:, -1], self.model.action_space
            )

            action_dists = [Categorical(logits=preds) for preds in split_action_preds]

            # To sample actions, you can use the sample() method of each distribution.
            sampled_actions = torch.stack([dist.sample() for dist in action_dists])

            # Assuming the goal ranking logic remains the same
            goal_ranking = mid_info["pred_horizons"][0, -1].argmax(-1)

            return goal_ranking, sampled_actions
