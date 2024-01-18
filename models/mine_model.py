import random

import torch
import numpy as np

from models.goal_model import get_goal_model
from models.mineclip import MineCLIP


def accquire_goal_embeddings(clip_path, goal_list, device="cuda"):
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
    clip_model = clip_model.to(device)
    res = {}
    with torch.no_grad():
        for goal in goal_list:
            print(goal)
            res[goal] = clip_model.encode_text([goal]).cpu().numpy()
    return res


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
        self.max_ranking = max_ranking

    def get_action(
        self, goal: str, goals: torch.Tensor, states: dict
    ) -> tuple[int, torch.Tensor]:
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
            raise NotImplementedError("Only support mining goals.")
            # Neural Network Agent
            # action_preds, mid_info = self.model.get_action(
            #     goals=goals,
            #     states=states,
            #     horizons=None,
            # )

            # from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical

            # action_dist = TorchMultiCategorical(
            #     action_preds[:, -1], None, self.model.action_space
            # )

            # Split the action prediction tensor into separate tensors for each categorical action.
            # split_action_preds = torch.split(
            #     action_preds[:, -1], [space.n for space in self.model.action_space]
            # )

            # # Create a Categorical distribution for each action space.
            # action_dists = [Categorical(logits=preds) for preds in split_action_preds]

            # # To sample actions, you can use the sample() method of each distribution.
            # sampled_actions = [dist.sample() for dist in action_dists]

            # action = action_dist.sample().squeeze(0)
            # goal_ranking = mid_info["pred_horizons"][0, -1].argmax(-1)
            # return goal_ranking, action
