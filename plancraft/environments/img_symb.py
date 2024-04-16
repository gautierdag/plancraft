from typing import List, Sequence

import gym
import numpy as np
from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero import handlers, mc, spaces
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.agent.action import Action
from minerl.herobraine.hero.handlers.translation import TranslationHandler

MINUTE = 20 * 60


class CraftingTableOnly(Handler):
    def to_string(self):
        return "start_with_crafting_table"

    def xml_template(self) -> str:
        return "<CraftingTableOnly>true</CraftingTableOnly>"


class TimeoutWrapper(gym.Wrapper):
    """Timeout wrapper"""

    def __init__(self, env):
        super().__init__(env)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

    def reset(self):
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info


class InventoryObservation(TranslationHandler):
    """
    Handles GUI Workbench Observations for selected items
    """

    def to_string(self):
        return "inventory"

    def xml_template(self) -> str:
        return str("""<ObservationFromFullInventory flat="false"/>""")

    def __init__(self, item_list, _other="other"):
        item_list = sorted(item_list)
        super().__init__(
            spaces.Dict(
                spaces={
                    k: spaces.Box(
                        low=0,
                        high=2304,
                        shape=(),
                        dtype=np.int32,
                        normalizer_scale="log",
                    )
                    for k in item_list
                }
            )
        )
        self.num_items = len(item_list)
        self.items = item_list

    def add_to_mission_spec(self, mission_spec):
        pass

    def from_hero(self, info):
        """
        Converts the Hero observation into a one-hot of the inventory items
        for a given inventory container. Ignores variant / color
        :param obs:
        :return:
        """
        print(info["inventory"])
        return info["inventory"]

    def from_universal(self, obs):
        raise NotImplementedError(
            "from_universal not implemented in InventoryObservation"
        )


class InventoryCommandAction(Action):
    """
    Handler which lets agents programmatically interact with an open container

    Using this - agents can move a chosen quantity of items from one slot to another.
    """

    def to_string(self):
        return "inventory_command"

    def xml_template(self) -> str:
        return str("<InventoryCommands/>")

    def __init__(self):
        self._command = "inventory_command"

        # first argument is the slot to take from, second is the slot to put into
        # third is the count to take
        super().__init__(
            self.command,
            spaces.Box(low=0, high=64, shape=[3], dtype=np.int32),
        )

    def from_universal(self, x):
        #         if 'custom_action' in x and 'cameraYaw' in x['custom_action'] and 'cameraPitch' in x['custom_action']:
        #     delta_pitch = x['custom_action']['cameraPitch']
        #     delta_yaw = x['custom_action']['cameraYaw']
        #     assert not np.isnan(np.sum(x['custom_action']['cameraYaw'])), "NAN in action!"
        #     assert not np.isnan(np.sum(x['custom_action']['cameraPitch'])), "NAN in action!"
        #     return np.array([-delta_pitch, -delta_yaw], dtype=np.float32)
        # else:
        #     return np.array([0.0, 0.0], dtype=np.float32)

        return np.array([0, 0, 0], dtype=np.int32)


PLANCRAFT_GYM_ENTRY_POINT = "crafting_gym:_plancraft_gym_entrypoint"


class PlancraftBaseEnvSpec(HumanControlEnvSpec):
    def __init__(
        self,
        name,
        max_episode_steps=2 * MINUTE,
        inventory: Sequence[dict] = (),
        preferred_spawn_biome: str = "plains",
    ):
        self.inventory = inventory  # Used by minerl.util.docs to construct Sphinx docs.
        self.preferred_spawn_biome = preferred_spawn_biome
        super().__init__(
            name=name,
            max_episode_steps=max_episode_steps,
            # smaller resolution to maximize crafting gui
            resolution=[260, 180],
            cursor_size_range=[16, 16],
        )

    def is_from_folder(self, folder: str) -> bool:
        return False

    def create_agent_start(self) -> List[handlers.Handler]:
        base_agent_start_handlers = super().create_agent_start()
        return base_agent_start_handlers + [
            handlers.SimpleInventoryAgentStart(self.inventory),
            handlers.PreferredSpawnBiome(self.preferred_spawn_biome),
            handlers.DoneOnDeath(),
            CraftingTableOnly(),
        ]

    def create_observables(self) -> List[TranslationHandler]:
        return [
            handlers.POVObservation(self.resolution),
            InventoryObservation(mc.ALL_ITEMS),
        ]

    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        # Camera and mouse
        actionables = [
            handlers.KeybasedCommandAction(v, v)
            for k, v in mc.KEYMAP.items()
            # if "mouse" in k
        ] + [
            handlers.CameraAction(),
            handlers.ChatAction(),
            InventoryCommandAction(),
        ]
        return actionables

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return []

    def create_server_world_generators(self) -> List[handlers.Handler]:
        # TODO the original biome forced is not implemented yet. Use this for now.
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> List[handlers.Handler]:
        return [
            handlers.ServerQuitFromTimeUp((self.max_episode_steps * mc.MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_decorators(self) -> List[handlers.Handler]:
        return []

    def create_server_initial_conditions(self) -> List[handlers.Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def create_mission_handlers(self):
        # Implements abstractmethod
        return ()

    def create_monitors(self):
        # Implements abstractmethod
        return ()

    def create_rewardables(self):
        # Implements abstractmethod
        return ()

    def determine_success_from_rewards(self, rewards: list) -> bool:
        """Implements abstractmethod.

        Plancraft environment have no rewards, so this is always False."""
        return False

    def get_docstring(self):
        return self.__class__.__doc__

    def _entry_point(self, fake: bool = False):
        return PLANCRAFT_GYM_ENTRY_POINT


def _plancraft_gym_entrypoint(
    env_spec: "PlancraftBaseEnvSpec",
    fake: bool = False,
) -> _singleagent._SingleAgentEnv:
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = TimeoutWrapper(env)
    return env
