from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_tools import download_dataset, find_missing_ids, preprocess_dataset, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, convert, preprocess, and report WLASL videos")
    parser.add_argument("--index", default=str(Path(__file__).resolve().parents[1] / "data" / "WLASL_v0.3.json"))
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--youtube-downloader", default="yt-dlp")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, verbose=args.verbose)
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw_videos"
    mp4_dir = data_dir / "raw_videos_mp4"
    videos_dir = data_dir / "videos"

    result: dict[str, object] = {}
    if not args.skip_download:
        result["download"] = download_dataset(
            index_path=args.index,
            raw_dir=raw_dir,
            youtube_downloader=args.youtube_downloader,
            skip_youtube=args.skip_youtube,
            skip_direct=args.skip_direct,
            overwrite=args.overwrite,
            limit=args.limit,
        )
    result["preprocess"] = preprocess_dataset(
        index_path=args.index,
        raw_dir=raw_dir,
        mp4_dir=mp4_dir,
        videos_dir=videos_dir,
        skip_convert=args.skip_convert,
        overwrite=args.overwrite,
        ffmpeg=args.ffmpeg,
        fps=args.fps,
        limit=args.limit,
    )
    missing = find_missing_ids(args.index, videos_dir)
    missing_path = data_dir / "missing.txt"
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    result["missing"] = {"count": len(missing), "output": str(missing_path)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
