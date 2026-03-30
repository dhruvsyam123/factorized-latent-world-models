from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class EnvConfig:
    grid_size: int = 10
    render_size: int = 84
    max_steps: int = 40
    max_hazards: int = 2
    max_distractors: int = 2
    task_types: tuple[str, ...] = ("reach", "key_door", "push")


@dataclass
class DatasetConfig:
    passive_train_transitions: int = 200_000
    passive_val_transitions: int = 10_000
    labelled_train_transitions: int = 10_000
    labelled_val_transitions: int = 2_000
    policy_mix: tuple[str, ...] = ("random", "goal_seek", "push_helper")
    seed: int = 7


@dataclass
class ModelConfig:
    encoder_type: str = "entity"
    image_channels: int = 3
    factor_dim: int = 128
    latent_action_dim: int = 16
    hidden_dim: int = 256
    num_attention_heads: int = 4
    decoder_channels: int = 64
    max_entities: int = 8
    entity_feature_dim: int = 14
    num_actions: int = 5
    reconstruction_weight: float = 1.0
    entity_weight: float = 5.0
    latent_weight: float = 0.1
    slot_iterations: int = 3
    slot_mlp_dim: int = 256


@dataclass
class TrainConfig:
    batch_size: int = 64
    stage1_epochs: int = 15
    stage2_epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    device: str = "cpu"
    freeze_world_model_stage2: bool = True
    num_workers: int = 0


@dataclass
class EvalConfig:
    episodes: int = 100
    planner_horizon: int = 12
    planner_iterations: int = 4
    planner_population: int = 64
    planner_elite_count: int = 8


@dataclass
class ProjectConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def default_project_config() -> ProjectConfig:
    return ProjectConfig()
