from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import decoded_cache_path, get_frame_count, load_metadata, metadata_cache_path, save_metadata
from .transforms import build_transforms


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    label: int
    subset: str
    path: Path
    frame_start: int
    frame_end: int
    frame_count: int

    @property
    def duration(self) -> int:
        if len(self.video_id) == 6:
            return max(1, min(self.frame_count - self.frame_start, self.frame_end - self.frame_start))
        return max(1, self.frame_count)

    @property
    def segment_start(self) -> int:
        return self.frame_start if len(self.video_id) == 6 else 0


def load_split(split_file: str | Path) -> dict:
    with Path(split_file).open("r", encoding="utf-8") as f:
        return json.load(f)


def num_classes_from_split(split_file: str | Path) -> int:
    data = load_split(split_file)
    return len({int(item["action"][0]) for item in data.values()})


def resolve_video_path(data_root: str | Path, video_id: str) -> Path:
    return Path(data_root) / f"{video_id}.mp4"


def build_metadata(
    split_file: str | Path,
    data_root: str | Path,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> tuple[dict[str, dict], list[str]]:
    data = load_split(split_file)
    cache_path = metadata_cache_path(cache_dir or Path(data_root) / ".i3d_cache", split_file, data_root)
    metadata = {} if refresh else load_metadata(cache_path)
    missing: list[str] = []
    changed = False

    for video_id in data:
        path = resolve_video_path(data_root, video_id)
        cached = metadata.get(video_id)
        if cached and cached.get("path") == str(path) and Path(cached["path"]).exists():
            continue

        if not path.exists():
            missing.append(video_id)
            metadata.pop(video_id, None)
            changed = True
            continue

        metadata[video_id] = {"path": str(path), "frame_count": get_frame_count(path)}
        changed = True

    if changed or not cache_path.exists():
        save_metadata(cache_path, metadata)
    return metadata, missing


def make_records(
    split_file: str | Path,
    data_root: str | Path,
    splits: Iterable[str],
    cache_dir: str | Path | None = None,
    cache_mode: str = "metadata",
    min_frames: int = 1,
    refresh_cache: bool = False,
) -> tuple[list[VideoRecord], list[str]]:
    split_names = set(splits)
    data = load_split(split_file)
    metadata, missing = (
        build_metadata(split_file, data_root, cache_dir, refresh=refresh_cache)
        if cache_mode in {"metadata", "decoded"}
        else ({}, [])
    )
    records: list[VideoRecord] = []

    for video_id, item in data.items():
        if item["subset"] not in split_names:
            continue
        path = resolve_video_path(data_root, video_id)
        if not path.exists():
            missing.append(video_id)
            continue

        frame_count = int(metadata.get(video_id, {}).get("frame_count") or get_frame_count(path))
        if frame_count < min_frames:
            continue

        label, frame_start, frame_end = item["action"]
        records.append(
            VideoRecord(
                video_id=video_id,
                label=int(label),
                subset=str(item["subset"]),
                path=path,
                frame_start=max(0, int(frame_start)),
                frame_end=max(0, int(frame_end)),
                frame_count=frame_count,
            )
        )

    return records, sorted(set(missing))


def resize_short_side(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape[:2]
    short = min(h, w)
    if short == target:
        return frame
    scale = target / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def read_video_frames(path: str | Path, start: int, num_frames: int, resize_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_short_side(frame, resize_size)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return np.asarray(frames, dtype=np.uint8)


def pad_frames(frames: np.ndarray, total_frames: int) -> np.ndarray:
    if frames.shape[0] >= total_frames:
        return frames[:total_frames]
    pad_count = total_frames - frames.shape[0]
    pad = np.repeat(frames[-1:,...], pad_count, axis=0)
    return np.concatenate([frames, pad], axis=0)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32)
    frames = (frames / 255.0) * 2.0 - 1.0
    return torch.from_numpy(frames.transpose(3, 0, 1, 2))


class WLASLVideoDataset(Dataset):
    def __init__(
        self,
        split_file: str | Path,
        data_root: str | Path,
        splits: Iterable[str],
        clip_frames: int = 64,
        image_size: int = 224,
        resize_size: int = 256,
        train: bool = True,
        cache_mode: str = "metadata",
        cache_dir: str | Path | None = None,
        refresh_cache: bool = False,
    ) -> None:
        if cache_mode not in {"none", "metadata", "decoded"}:
            raise ValueError("cache_mode must be one of: none, metadata, decoded")
        self.records, self.missing_video_ids = make_records(
            split_file=split_file,
            data_root=data_root,
            splits=splits,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            min_frames=1,
            refresh_cache=refresh_cache,
        )
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.resize_size = resize_size
        self.train = train
        self.cache_mode = cache_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transforms = build_transforms(train=train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.records)

    def _sample_start(self, record: VideoRecord) -> int:
        duration = max(1, record.duration)
        if duration > self.clip_frames:
            max_offset = duration - self.clip_frames
            offset = random.randint(0, max_offset) if self.train else max_offset // 2
        else:
            offset = 0
        return record.segment_start + offset

    def _read_frames(self, record: VideoRecord, start: int) -> np.ndarray:
        if self.cache_mode == "decoded" and self.cache_dir is not None:
            cache_path = decoded_cache_path(self.cache_dir, record.video_id)
            if cache_path.exists():
                full = np.load(cache_path)
            else:
                full = read_video_frames(record.path, 0, record.frame_count, self.resize_size)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, full)
            frames = full[start : start + self.clip_frames]
            if frames.size == 0:
                frames = full[-1:]
            return frames
        return read_video_frames(record.path, start, self.clip_frames, self.resize_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        start = self._sample_start(record)
        frames = self._read_frames(record, start)
        frames = pad_frames(frames, self.clip_frames)
        frames = self.transforms(frames)
        return frames_to_tensor(frames), torch.tensor(record.label, dtype=torch.long), record.video_id
