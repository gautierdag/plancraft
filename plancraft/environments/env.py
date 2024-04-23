from typing import Literal, Sequence, Union

import numpy as np
from minerl.env import _singleagent

from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero import handlers, mc, spaces
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.agent.action import Action
from minerl.herobraine.hero.handlers.agent.start import InventoryAgentStart
from minerl.herobraine.hero.handlers.translation import TranslationHandler


PLANCRAFT_MODE = Literal[
    "symbolic_symbolic",  # symbolic env and symbolic actions
    "image_symbolic",  # image env and symbolic actions
    "image_real",  # image env and real (mouse + keyboard) actions
]
MINUTE = 20 * 60


class CustomInventoryAgentStart(InventoryAgentStart):
    def __init__(self, inventory: list[dict[str, Union[str, int]]]):
        super().__init__({item["slot"]: item for item in inventory})


class CraftingTableOnly(Handler):
    def to_string(self):
        return "start_with_crafting_table"

    def xml_template(self) -> str:
        return "<CraftingTableOnly>true</CraftingTableOnly>"


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
        return np.array([0, 0, 0], dtype=np.int32)


class PlancraftBaseEnvSpec(HumanControlEnvSpec):
    def __init__(
        self,
        mode: PLANCRAFT_MODE = "image_real",
        max_episode_steps=2 * MINUTE,
        inventory: Sequence[dict] = (),
        preferred_spawn_biome: str = "plains",
        resolution=[260, 180],
    ):
        self.inventory = inventory
        self.preferred_spawn_biome = preferred_spawn_biome
        self.mode = mode

        if "symbolic" in mode:
            cursor_size = 1
        else:
            cursor_size = 16

        name = f"plancraft-{mode}-v0"
        super().__init__(
            name=name,
            max_episode_steps=max_episode_steps,
            resolution=resolution,
            cursor_size_range=[cursor_size, cursor_size],
        )

    def create_agent_start(self) -> list[Handler]:
        base_agent_start_handlers = super().create_agent_start()
        return base_agent_start_handlers + [
            CustomInventoryAgentStart(self.inventory),
            handlers.PreferredSpawnBiome(self.preferred_spawn_biome),
            handlers.DoneOnDeath(),
            CraftingTableOnly(),
        ]

    def create_observables(self) -> list[TranslationHandler]:
        return [
            handlers.POVObservation(self.resolution),
            InventoryObservation(mc.ALL_ITEMS),
        ]

    def create_server_world_generators(self) -> list[Handler]:
        # TODO the original biome forced is not implemented yet. Use this for now.
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> list[Handler]:
        return [
            handlers.ServerQuitFromTimeUp((self.max_episode_steps * mc.MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_initial_conditions(self) -> list[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def create_actionables(self) -> list[TranslationHandler]:
        """
        Symbolic env can move items around in the inventory using function
        Real env can use camera/keyboard
        """
        # Camera and mouse
        if "symbolic" in self.name:
            return [InventoryCommandAction()]
        elif "real" in self.name:
            return [
                handlers.KeybasedCommandAction(v, v) for k, v in mc.KEYMAP.items()
            ] + [handlers.CameraAction()]
        else:
            raise ValueError("Invalid mode")

    def is_from_folder(self, folder: str) -> bool:
        return False

    def create_agent_handlers(self) -> list[Handler]:
        return []

    def create_mission_handlers(self):
        return []

    def create_monitors(self):
        return []

    def create_rewardables(self):
        return []

    def create_server_decorators(self) -> list[Handler]:
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return False

    def get_docstring(self):
        return self.__class__.__doc__


class PlancraftEnv(_singleagent._SingleAgentEnv):
    def __init__(
        self,
        inventory: list[dict],
        mode: PLANCRAFT_MODE = "image_real",
        preferred_spawn_biome="plains",
        resolution=[260, 180],
    ):
        preferred_spawn_biome = "plains"
        env_spec = PlancraftBaseEnvSpec(
            mode=mode,
            preferred_spawn_biome=preferred_spawn_biome,
            inventory=inventory,
            resolution=resolution,
        )
        super(PlancraftEnv, self).__init__(env_spec=env_spec)
        self.reset()
