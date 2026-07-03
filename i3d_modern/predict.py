from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, load_model_state
from .data import frames_to_tensor, pad_frames, read_video_frames
from .labels import load_class_names
from .model import build_i3d
from .runtime import get_device
from .transforms import CenterCrop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run I3D prediction on one video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--num-classes", type=int, default=2000)
    parser.add_argument("--class-list", default="data/splits/wlasl_class_list.txt")
    parser.add_argument("--clip-frames", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    labels = load_class_names(args.class_list)

    frames = read_video_frames(args.video, start=0, num_frames=args.clip_frames, resize_size=args.resize_size)
    frames = pad_frames(frames, args.clip_frames)
    frames = CenterCrop(args.image_size)(frames)
    inputs = frames_to_tensor(frames).unsqueeze(0).to(device)

    model = build_i3d(args.num_classes, device=device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    load_model_state(model, checkpoint)
    model.eval()

    logits = model(inputs).max(dim=2).values
    probs = torch.softmax(logits, dim=1)[0]
    k = min(args.top_k, probs.numel())
    values, indices = probs.topk(k)
    result = [
        {"rank": rank + 1, "class_id": int(idx), "gloss": labels.get(int(idx), str(int(idx))), "probability": float(prob)}
        for rank, (prob, idx) in enumerate(zip(values.cpu(), indices.cpu()))
    ]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
