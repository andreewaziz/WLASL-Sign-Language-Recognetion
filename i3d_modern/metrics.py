from __future__ import annotations

import json
from pathlib import Path

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1, 5, 10)) -> dict[str, int]:
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return {f"top{k}": int(correct[: min(k, max_k)].reshape(-1).float().sum().item()) for k in topk}


def append_jsonl(path: str | Path, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
