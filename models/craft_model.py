from itertools import chain
import numpy as np

from minedojo.sim.mc_meta.mc import ALL_CRAFT_SMELT_ITEMS


class CraftAgent:
    """
    Craft agent based on 'craft' action space.
    """

    def __init__(self, env, no_op: np.array):
        self.env = env
        self.no_op = no_op
        self.history = {
            "craft_w_table": None,
            "craft_wo_table": None,
            "smelt_w_furnace": None,
        }

    def no_op(self, times=20):
        for i in range(times):
            act = self.no_op.copy()
            yield act

    def take_forward(self, times=3):
        for _ in range(times):
            yield self.no_op.copy()

    def index_slot(self, goal):
        #! accquire info
        _, _, _, info = self.env.step(self.no_op.copy())
        slot = -1
        for item in info["inventory"]:
            if goal == item["name"]:
                slot = item["index"]
                break
        return slot

    def equip(self, goal):
        _, _, _, info = self.env.step(self.no_op.copy())
        for item in info["inventory"]:
            if item["name"] == goal and item["index"] > 0:
                act = self.no_op.copy()
                act[5] = 5
                act[7] = item["index"]
                yield act
                return

    def pillar_jump(self, stepping_stone="cobblestone"):
        for act in chain(
            self.look_to(-85),
            self.attack(40),
            self.place_down(stepping_stone),
            self.place_down(stepping_stone),
            self.place_down(stepping_stone),
        ):
            yield act

    def go_surface(self):
        while True:
            obs, reward, done, info = self.env.step(self.no_op.copy())
            if info["can_see_sky"]:
                return
            candidates = ["dirt", "stone", "cobblestone"]
            insufficient = True
            for stepping_stone in candidates:
                quantity = sum(
                    [
                        item["quantity"]
                        for item in info["inventory"]
                        if item["name"] == stepping_stone
                    ]
                )
                if quantity >= 1:
                    insufficient = False
                    for act in self.pillar_jump(stepping_stone):
                        yield act
                    break
            if insufficient:
                return

    def use(self):
        act = self.no_op.copy()
        act[5] = 1
        yield act
        yield self.no_op.copy()

    def look_to(self, deg=0):
        #! accquire info
        obs, _, _, _ = self.env.step(self.no_op.copy())
        while obs["compass"][1] < deg:
            act = self.no_op.copy()
            act[3] = 10
            act[5] = 3
            yield act
            obs, _, _, _ = self.env.step(self.no_op.copy())
        while obs["compass"][1] > deg:
            act = self.no_op.copy()
            act[5] = 3
            act[3] = 0
            yield act
            obs, _, _, _ = self.env.step(self.no_op.copy())

    def jump(self):
        act = self.no_op.copy()
        act[2] = 1
        yield act
        yield self.no_op.copy()

    def place(self, goal):
        slot = self.index_slot(goal)
        if slot == -1:
            return False
        act = self.no_op.copy()
        act[5] = 6
        act[7] = slot
        yield act

    def place_down(self, goal):
        if self.index_slot(goal) == -1:
            return None
        for act in chain(
            self.look_to(deg=87),
            self.attack(2),
            self.jump(),
            self.place(goal),
            self.use(),
        ):
            yield act

    def attack(self, times=20):
        for i in range(times):
            act = self.no_op.copy()
            act[5] = 3
            yield act
        yield self.no_op.copy()

    def recycle(self, goal, times=20):
        for i in range(times):
            act = self.no_op.copy()
            act[5] = 3
            obs, reward, done, info = self.env.step(act)
            if any([item["name"] == goal for item in info["inventory"]]):
                break
        yield self.no_op.copy()
        for act in chain(
            self.look_to(0),
            self.take_forward(3),
        ):
            yield act

    def craft_wo_table(self, goal):
        act = self.no_op.copy()
        act[5] = 4
        act[6] = ALL_CRAFT_SMELT_ITEMS.index(goal)
        yield act

    def forward(self, times=5):
        for i in range(times):
            act = self.no_op.copy()
            act[0] = 1
            yield act

    def craft_w_table(self, goal):
        if self.index_slot("crafting_table") == -1:
            return None
        for act in chain(
            self.forward(5),
            self.look_to(-87),
            self.attack(40),
            self.place_down("crafting_table"),
            self.craft_wo_table(goal),
            self.recycle("crafting_table", 200),
        ):
            # print(f"{goal}: {act}")
            yield act

    def smelt_w_furnace(self, goal):
        if self.index_slot("furnace") == -1:
            return None
        for act in chain(
            self.look_to(-87),
            self.attack(40),
            self.place_down("furnace"),
            self.craft_wo_table(goal),
            self.recycle("furnace", 200),
        ):
            yield act

    def smelt_wo_furnace(self, goal):
        for act in self.craft_wo_table(goal):
            yield act

    def get_action(self, preconditions, goal_type, goal):
        if goal_type == "craft":
            use_crafting_table = "crafting_table" in preconditions
            if use_crafting_table:
                if self.history["craft_w_table"] is None:
                    self.history["craft_w_table"] = self.craft_w_table(goal)
                try:
                    act = next(self.history["craft_w_table"])
                    return act, False
                except:
                    self.history["craft_w_table"] = None
                    return self.no_op.copy(), True
            else:
                if self.history["craft_wo_table"] is None:
                    self.history["craft_wo_table"] = self.craft_wo_table(goal)
                try:
                    act = next(self.history["craft_wo_table"])
                    return act, False
                except:
                    self.history["craft_wo_table"] = None
                    return self.no_op.copy(), True
        elif goal_type == "smelt":
            if self.history["smelt_w_furnace"] is None:
                self.history["smelt_w_furnace"] = self.smelt_w_furnace(goal)
            try:
                act = next(self.history["smelt_w_furnace"])
                return act, False
            except:
                self.history["smelt_w_furnace"] = None
                return self.no_op.copy(), True
