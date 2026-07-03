from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_tools import add_common_dataset_args, find_missing_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write missing WLASL clip IDs for the modern I3D data folder")
    add_common_dataset_args(parser)
    parser.add_argument("--videos-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    videos_dir = Path(args.videos_dir) if args.videos_dir else data_dir / "videos"
    output = Path(args.output) if args.output else data_dir / "missing.txt"
    missing = find_missing_ids(args.index, videos_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"missing={len(missing)} output={output}")


if __name__ == "__main__":
    main()
