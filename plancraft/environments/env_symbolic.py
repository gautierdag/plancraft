from gym import Env
import numpy as np

from minerl.herobraine.hero import spaces, mc

from plancraft.environments.actions import InventoryCommandAction
from plancraft.environments.recipes import RECIPES


class SymbolicPlancraft(Env):
    def __init__(self):
        self.action_space = InventoryCommandAction()
        self.observation_space = spaces.Dict(
            spaces={
                k: spaces.Box(
                    low=0,
                    high=2304,
                    shape=(),
                    dtype=np.int32,
                    normalizer_scale="log",
                )
                for k in mc.ALL_ITEMS
            }
        )
        self.state = {}

    def step(self, action):
        # action is a dict with keys "slot", "slot_to", "count"
        slot, slot_to, count = action
        item = self.state[slot]
        if item == "air":
            return self.state, 0, False, {}
        if count > self.state[slot]:
            return self.state, 0, False, {}
        if item not in RECIPES:
            return self.state, 0, False, {}
        if slot_to not in self.state:
            self.state[slot_to] = item
            self.state[slot] -= count
            return self.state, 1, False, {}
        if self.state[slot_to] != RECIPES[item]:
            return self.state, 0, False, {}
        self.state[slot_to] = item
        self.state[slot] -= count
        return self.state, 1, False, {}

    def reset(self):
        self.state = {}
        return self.state

    def render(self):
        print(f"state: {self.state}")
