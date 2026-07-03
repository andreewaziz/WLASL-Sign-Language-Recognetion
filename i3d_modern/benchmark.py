from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .cache import default_cache_dir
from .config import load_config
from .data import WLASLVideoDataset
from .runtime import get_device, set_seed
from .train import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WLASL I3D dataloader throughput")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--output", default="runs/benchmark")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_file = config.resolved_split_file(args.config)
    device = get_device(args.device)
    set_seed(config.seed)
    cache_dir = Path(config.cache_dir) if config.cache_dir else default_cache_dir(args.output, args.data_root)
    dataset = WLASLVideoDataset(
        split_file=split_file,
        data_root=args.data_root,
        splits=config.train_splits,
        clip_frames=config.clip_frames,
        image_size=config.image_size,
        resize_size=config.resize_size,
        train=True,
        cache_mode=config.cache_mode,
        cache_dir=cache_dir,
    )
    loader = make_loader(dataset, config.batch_size, config.num_workers, True, device)

    count = 0
    start = time.time()
    for batch_idx, (inputs, _, _) in enumerate(loader, start=1):
        if batch_idx > args.batches:
            break
        if device.type == "cuda":
            inputs = inputs.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        count += inputs.size(0)
    elapsed = max(time.time() - start, 1e-9)
    print(
        json.dumps(
            {
                "batches": min(args.batches, batch_idx if "batch_idx" in locals() else 0),
                "samples": count,
                "seconds": elapsed,
                "samples_per_second": count / elapsed,
                "cache_mode": config.cache_mode,
                "num_workers": config.num_workers,
                "batch_size": config.batch_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
