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
    tokenizer: str
    num_generations: int
    mode: Literal["react", "act", "oracle", "dummy"] = "react"
    output_dir: str
    max_steps: int
    quantize: Literal[False, "int4", "int8"]
    environment: EnvironmentConfig
    split: DatasetSplit = "val.small"
    batch_size: int = 1
    max_message_window: int = 30  # max number of messages to keep in dialogue history (30 is around 8k llama3 tokens)
    hot_cache: bool = True  # whether to cache the dialogue history between steps
    resume: bool = True  # resume inference
    few_shot: bool = True  # whether to use few-shot prompt
    system_prompt: bool = True  # whether to use system prompt


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


class TrainingArgs(BaseModel):
    base_model: str = "llama3"
    trace_mode: str = "oa"
    push_to_hub: bool = False

    # uses less space but not working with multi-gpu training..
    qlora: bool = False

    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_r: int = 64
    # training data args
    seed: int = 42
    # model args
    batch_size: int = 1
    max_seq_length: int = 8142
    max_message_window: int = 100
    only_assistant: bool = True

    # training args
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    num_workers: int = 1


class TrainConfig(BaseModel):
    training: TrainingArgs
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
