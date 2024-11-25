from typing import Literal, Optional, Union

from pydantic import BaseModel, model_validator

try:
    from plancraft.environments.recipes import RECIPES
except ImportError:
    RECIPES = {}

DatasetSplit = Literal[
    "train", "val", "val.small", "val.small.easy", "test", "test.small"
]


class EnvironmentConfig(BaseModel):
    resolution: Literal["low", "medium", "high"] = "high"


class PlancraftConfig(BaseModel):
    model: str
    adapter: str = ""
    tokenizer: str
    num_generations: int
    mode: Literal["react", "act", "oracle", "dummy"] = "react"
    output_dir: str
    max_steps: int = 30  # max number of steps (smelt/move) to take in the environment before stopping
    quantize: Literal[False, "int4", "int8"]
    environment: EnvironmentConfig
    split: DatasetSplit = "val.small"
    max_message_window: int = 30  # max number of messages to keep in dialogue history (30 is around 8k llama3 tokens)
    hot_cache: bool = True  # whether to cache the dialogue history between steps
    resume: bool = True  # resume inference
    few_shot: bool = True  # whether to use few-shot prompt
    system_prompt: bool = True  # whether to use system prompt
    valid_actions: list[str] = ["move", "smelt", "think", "search", "impossible"]
    use_maskrcnn: bool = False  # whether to use maskrcnn for multimodal parsing

    # observations
    use_text_inventory: bool = True  # whether to include inventory in text
    use_images: bool = False  # whether to include images in multimodal content
    use_multimodal_content_format: bool = (
        False  # whether to use multimodal content format
    )

    @model_validator(mode="after")
    def validate(self):
        assert set(
            self.valid_actions
        ).issubset(
            {"move", "smelt", "think", "search", "impossible"}
        ), "valid_actions should be subset of {'move', 'smelt', 'think', 'search', 'impossible'}"

        if self.use_images:
            assert (
                not self.environment.symbolic
            ), "Set environment.symbolic to False when using images"

        return self


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

    recipe_type: Optional[str] = ""

    # post processing set recipe type
    def model_post_init(self, __context):
        recipe_types = set()
        if self.optimal_path is None:
            self.recipe_type = "impossible"
            return
        for step in self.optimal_path:
            for r in RECIPES[step]:
                recipe_types.add(r.recipe_type)
        if len(recipe_types) == 1:
            self.recipe_type = recipe_types.pop()
        else:
            self.recipe_type = "mixed"
