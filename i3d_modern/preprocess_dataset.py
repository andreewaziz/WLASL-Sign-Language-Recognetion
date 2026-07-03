from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_tools import add_common_dataset_args, preprocess_dataset, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw WLASL videos and extract per-instance MP4 clips")
    add_common_dataset_args(parser)
    parser.add_argument("--raw-dir")
    parser.add_argument("--mp4-dir")
    parser.add_argument("--videos-dir")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, verbose=args.verbose)
    data_dir = Path(args.data_dir)
    stats = preprocess_dataset(
        index_path=args.index,
        raw_dir=Path(args.raw_dir) if args.raw_dir else data_dir / "raw_videos",
        mp4_dir=Path(args.mp4_dir) if args.mp4_dir else data_dir / "raw_videos_mp4",
        videos_dir=Path(args.videos_dir) if args.videos_dir else data_dir / "videos",
        skip_convert=args.skip_convert,
        overwrite=args.overwrite,
        ffmpeg=args.ffmpeg,
        fps=args.fps,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
