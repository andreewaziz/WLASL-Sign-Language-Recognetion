from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2


def default_cache_dir(output_dir: str | Path | None, data_root: str | Path) -> Path:
    if output_dir:
        return Path(output_dir) / "cache"
    return Path(data_root) / ".i3d_cache"


def metadata_cache_path(cache_dir: str | Path, split_file: str | Path, data_root: str | Path) -> Path:
    key = hashlib.sha1(f"{Path(split_file).resolve()}::{Path(data_root).resolve()}".encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"metadata_{Path(split_file).stem}_{key}.json"


def get_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def load_metadata(path: str | Path) -> dict[str, dict]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(path: str | Path, metadata: dict[str, dict]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def decoded_cache_path(cache_dir: str | Path, video_id: str) -> Path:
    return Path(cache_dir) / "decoded" / f"{video_id}.npy"
