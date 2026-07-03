from __future__ import annotations

from pathlib import Path


def load_class_names(path: str | Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, gloss = line.split(None, 1)
            labels[int(idx)] = gloss
    return labels
