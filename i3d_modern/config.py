from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    subset: str
    num_classes: int
    split_file: str
    clip_frames: int = 64
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 6
    num_workers: int = 2
    epochs: int = 400
    max_steps: int = 64000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-8
    adam_eps: float = 1e-3
    grad_accum_steps: int = 1
    amp: bool = True
    cache_mode: str = "metadata"
    cache_dir: str | None = None
    seed: int = 0
    eval_interval: int = 1
    checkpoint_interval: int = 1
    train_splits: list[str] = field(default_factory=lambda: ["train", "val"])
    val_splits: list[str] = field(default_factory=lambda: ["test"])

    def resolved_split_file(self, config_path: str | Path) -> Path:
        path = Path(self.split_file)
        if path.is_absolute():
            return path

        config_relative = Path(config_path).resolve().parent / path
        if config_relative.exists():
            return config_relative

        cwd_relative = Path.cwd() / path
        if cwd_relative.exists():
            return cwd_relative

        return config_relative


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return TrainConfig(**raw)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return {
        field_name: getattr(config, field_name)
        for field_name in TrainConfig.__dataclass_fields__
    }
