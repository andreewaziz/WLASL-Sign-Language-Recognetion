from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .cache import default_cache_dir, decoded_cache_path
from .config import load_config
from .data import make_records, read_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare metadata or decoded-frame cache for WLASL I3D")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="runs/cache")
    parser.add_argument("--decoded", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_file = config.resolved_split_file(args.config)
    cache_dir = Path(config.cache_dir) if config.cache_dir else default_cache_dir(args.output, args.data_root)
    records, missing = make_records(
        split_file=split_file,
        data_root=args.data_root,
        splits=sorted(set(config.train_splits + config.val_splits)),
        cache_dir=cache_dir,
        cache_mode="metadata",
        refresh_cache=False,
    )
    print(f"Metadata ready in {cache_dir}. records={len(records)} missing={len(missing)}")

    if args.decoded:
        for record in tqdm(records, desc="decode"):
            out = decoded_cache_path(cache_dir, record.video_id)
            if out.exists():
                continue
            frames = read_video_frames(record.path, 0, record.frame_count, config.resize_size)
            out.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.save(out, frames)


if __name__ == "__main__":
    main()
