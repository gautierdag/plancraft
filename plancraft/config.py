from typing import Literal, Union, Optional

from pydantic import BaseModel

DatasetSplit = Literal[
    "train", "val", "val.small", "val.small.easy", "test", "test.small"
]


class EnvironmentConfig(BaseModel):
    symbolic: bool
    symbolic_observation_space: bool
    symbolic_action_space: bool
    preferred_spawn_biome: str = "plains"
    resolution: list[int] = [512, 512]


class PlancraftConfig(BaseModel):
    model: str
    num_generations: int
    mode: str
    output_dir: str
    max_steps: int
    quantize: Literal[False, "int4", "int8"]
    environment: EnvironmentConfig
    split: DatasetSplit = "val.small"
    batch_size: int = 1
    max_message_window: int = 30  # max number of messages to keep in dialogue history
    resume: bool = True  # resume inference


class WandbConfig(BaseModel):
    project: str
    entity: str
    mode: str


class LaunchConfig(BaseModel):
    command: str
    job_name: str
    gpu_limit: int
    gpu_product: str
    cpu_request: int
    ram_request: str
    interactive: bool = False
    namespace: str = "informatics"
    env_vars: dict[str, dict[str, str]]


class EvalConfig(BaseModel):
    plancraft: PlancraftConfig
    wandb: WandbConfig
    launch: LaunchConfig


class PlancraftExample(BaseModel):
    target: str
    inventory: dict[str, int]
    slotted_inventory: list[dict[str, Union[str, int]]]
    num_distractors: int
    impossible: bool
    optimal_path_length: Optional[int]
    optimal_path: Optional[list[str]]
    inventory_trace: Optional[list[dict[str, int]]]
    items_used: Optional[int]
    unique_items_used: Optional[int]
    complexity: Optional[int]
    complexity_bin: int
    unseen_in_train: bool
    unseen_in_val: bool
    split: DatasetSplit
    id: str
