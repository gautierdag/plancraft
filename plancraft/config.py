from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    symbolic: bool
    symbolic_observation_space: bool
    symbolic_action_space: bool


class PlancraftConfig(BaseModel):
    model: str
    num_generations: int
    mode: str
    max_steps: int
    guidance: bool
    quantize: bool
    environment: EnvironmentConfig
    split: str


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


class Config(BaseModel):
    plancraft: PlancraftConfig
    wandb: WandbConfig
    launch: LaunchConfig
