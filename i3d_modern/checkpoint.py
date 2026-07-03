from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    config: dict,
    epoch: int,
    global_step: int,
    best_metric: float,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
        },
        out,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def load_model_state(model: torch.nn.Module, checkpoint: dict) -> None:
    state = checkpoint.get("model", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
