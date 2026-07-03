from __future__ import annotations

import random

import numpy as np


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            frames = transform(frames)
        return frames


class RandomCrop:
    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        _, h, w, _ = frames.shape
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"Cannot crop {self.size} from frames with shape {frames.shape}")
        top = random.randint(0, h - th) if h != th else 0
        left = random.randint(0, w - tw) if w != tw else 0
        return frames[:, top : top + th, left : left + tw, :]


class CenterCrop:
    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        _, h, w, _ = frames.shape
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"Cannot crop {self.size} from frames with shape {frames.shape}")
        top = int(round((h - th) / 2.0))
        left = int(round((w - tw) / 2.0))
        return frames[:, top : top + th, left : left + tw, :]


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.flip(frames, axis=2).copy()
        return frames


def build_transforms(train: bool, image_size: int) -> Compose:
    if train:
        return Compose([RandomCrop(image_size), RandomHorizontalFlip()])
    return Compose([CenterCrop(image_size)])
