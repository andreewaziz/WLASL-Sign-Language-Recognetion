from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from i3d_modern.cache import get_frame_count
from i3d_modern.checkpoint import load_checkpoint, load_model_state
from i3d_modern.data import frames_to_tensor, pad_frames, read_video_frames
from i3d_modern.labels import load_class_names
from i3d_modern.model import build_i3d
from i3d_modern.runtime import get_device
from i3d_modern.transforms import CenterCrop

from person_crop import crop_video_around_person


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class PredictionItem:
    rank: int
    class_id: int
    gloss: str
    probability: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "class_id": self.class_id,
            "gloss": self.gloss,
            "probability": self.probability,
        }


class I3DSignPredictor:
    """Small inference interface around the WLASL RGB I3D model.

    The class loads the model once during initialization and exposes
    `predict_video(...)` as the main public method for Streamlit or scripts.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        class_list_path: str | Path | None = None,
        split_file: str | Path | None = None,
        num_classes: int = 2000,
        clip_frames: int = 64,
        image_size: int = 224,
        resize_size: int = 256,
        device: str | None = None,
    ) -> None:
        self.root = Path(__file__).resolve().parent
        self.checkpoint_path = self._resolve_path(checkpoint_path or "wlasl_i3d_pretrained.pt")
        self.class_list_path = self._resolve_path(class_list_path or "wlasl_class_list.txt")
        self.split_file = self._resolve_path(split_file or "data/splits/nslt_2000.json", must_exist=False)
        self.num_classes = num_classes
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.resize_size = resize_size
        self.device = get_device(None if device in {None, "auto"} else device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
        if not self.class_list_path.exists():
            raise FileNotFoundError(f"Class list not found: {self.class_list_path}")

        self.labels = load_class_names(self.class_list_path)
        self.split_lookup = self._load_split_lookup(self.split_file)
        self.model = build_i3d(self.num_classes, device=self.device)
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        load_model_state(self.model, checkpoint)
        self.model.eval()

    def _resolve_path(self, path: str | Path, must_exist: bool = True) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        locations = [
            self.root / candidate,
            self.root / "data" / "splits" / candidate,
        ]
        for location in locations:
            if location.exists():
                return location

        fallback = self.root / candidate
        if must_exist:
            return fallback
        return fallback

    def _load_split_lookup(self, split_file: Path) -> dict[str, dict[str, Any]]:
        if not split_file.exists():
            return {}
        with split_file.open("r", encoding="utf-8") as f:
            split_data = json.load(f)

        lookup: dict[str, dict[str, Any]] = {}
        for video_id, item in split_data.items():
            action = item.get("action", [])
            if not action:
                continue
            class_id = int(action[0])
            lookup[str(video_id)] = {
                "video_id": str(video_id),
                "class_id": class_id,
                "gloss": self.labels.get(class_id, str(class_id)),
                "subset": item.get("subset"),
                "frame_start": int(action[1]) if len(action) > 1 else None,
                "frame_end": int(action[2]) if len(action) > 2 else None,
            }
        return lookup

    def get_actual_label(self, video_path: str | Path) -> dict[str, Any] | None:
        video_id = Path(video_path).stem
        return self.split_lookup.get(video_id)

    def predict_video(
        self,
        video_path: str | Path,
        top_k: int = 10,
        auto_crop: bool = False,
    ) -> dict[str, Any]:
        original_path = Path(video_path)
        if not original_path.exists():
            raise FileNotFoundError(f"Video not found: {original_path}")

        if auto_crop:
            cropped_path = original_path.with_name(
                original_path.stem + "_cropped" + original_path.suffix
            )
            path = crop_video_around_person(original_path, cropped_path)
        else:
            path = original_path

        actual = self.get_actual_label(original_path)

        try:
            frame_count = get_frame_count(path)
        except Exception:
            frame_count = None

        if actual and actual.get("frame_start") is not None:
            start_frame = int(actual["frame_start"])
        elif frame_count is not None:
            start_frame = max(0, (frame_count - self.clip_frames) // 2)
        else:
            start_frame = 0

        frames = read_video_frames(
            path,
            start=start_frame,
            num_frames=self.clip_frames,
            resize_size=self.resize_size,
        )
        frames = pad_frames(frames, self.clip_frames)
        frames = CenterCrop(self.image_size)(frames)
        inputs = frames_to_tensor(frames).unsqueeze(0).to(self.device)

        with torch.no_grad():
            per_frame_logits = self.model(inputs)
            logits = per_frame_logits.max(dim=2).values
            probabilities = torch.softmax(logits, dim=1)[0]
            k = min(top_k, probabilities.numel())
            values, indices = probabilities.topk(k)

        predictions = [
            PredictionItem(
                rank=rank + 1,
                class_id=int(idx),
                gloss=self.labels.get(int(idx), str(int(idx))),
                probability=float(prob),
            ).as_dict()
            for rank, (prob, idx) in enumerate(zip(values.cpu(), indices.cpu()))
        ]

        actual = self.get_actual_label(original_path)
        predicted = predictions[0] if predictions else None
        is_correct = (
            bool(actual and predicted and int(actual["class_id"]) == int(predicted["class_id"]))
            if actual
            else None
        )

        frame_count: int | None
        try:
            frame_count = get_frame_count(path)
        except Exception:
            frame_count = None

        return {
            "video_path": str(original_path),
            "cropped_video_path": str(path) if auto_crop else None,
            "video_id": original_path.stem,
            "frame_count": frame_count,
            "device": str(self.device),
            "num_classes": self.num_classes,
            "top_k": predictions,
            "predicted": predicted,
            "actual": actual,
            "is_correct": is_correct,
        }

    @staticmethod
    def list_dataset_videos(folder: str | Path) -> list[Path]:
        root = Path(folder)
        if not root.exists():
            return []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )