from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_tools import add_common_dataset_args, download_dataset, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw WLASL videos into the modern I3D data folder")
    add_common_dataset_args(parser)
    parser.add_argument("--raw-dir")
    parser.add_argument("--youtube-downloader", default="yt-dlp")
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep-min", type=float, default=0.5)
    parser.add_argument("--sleep-max", type=float, default=1.5)
    parser.add_argument("--log-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, verbose=args.verbose)
    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else data_dir / "raw_videos"
    stats = download_dataset(
        index_path=args.index,
        raw_dir=raw_dir,
        youtube_downloader=args.youtube_downloader,
        skip_youtube=args.skip_youtube,
        skip_direct=args.skip_direct,
        overwrite=args.overwrite,
        limit=args.limit,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
