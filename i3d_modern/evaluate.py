from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cache import default_cache_dir
from .checkpoint import load_checkpoint, load_model_state
from .config import load_config
from .data import WLASLVideoDataset
from .model import build_i3d
from .runtime import get_device, set_seed
from .train import make_loader, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate modern RGB I3D on WLASL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--limit-val-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_file = config.resolved_split_file(args.config)
    device = get_device(args.device)
    set_seed(config.seed)

    output_dir = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent
    cache_dir = Path(config.cache_dir) if config.cache_dir else default_cache_dir(output_dir, args.data_root)
    dataset = WLASLVideoDataset(
        split_file=split_file,
        data_root=args.data_root,
        splits=config.val_splits,
        clip_frames=config.clip_frames,
        image_size=config.image_size,
        resize_size=config.resize_size,
        train=False,
        cache_mode=config.cache_mode,
        cache_dir=cache_dir,
    )
    if not dataset:
        raise RuntimeError(f"Empty validation dataset for data_root={args.data_root}")
    loader = make_loader(dataset, config.batch_size, config.num_workers, False, device)
    model = build_i3d(config.num_classes, device=device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    load_model_state(model, checkpoint)

    metrics = validate(
        model=model,
        loader=loader,
        device=device,
        num_classes=config.num_classes,
        amp=config.amp,
        limit_batches=args.limit_val_batches,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
