from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


KAGGLE_DATASET = "risangbaskoro/wlasl-processed"


def find_videos_dir(dataset_path: str | Path) -> Path:
    root = Path(dataset_path)
    direct = root / "videos"
    if direct.exists() and direct.is_dir():
        return direct

    named = [path for path in root.rglob("videos") if path.is_dir()]
    if named:
        return named[0]

    if list(root.glob("*.mp4")):
        return root

    candidate_dirs = []
    for path in root.rglob("*"):
        if path.is_dir():
            count = sum(1 for _ in path.glob("*.mp4"))
            if count:
                candidate_dirs.append((count, path))
    if candidate_dirs:
        return sorted(candidate_dirs, reverse=True)[0][1]

    raise FileNotFoundError(f"Could not find a videos directory under {root}")


def find_index_file(dataset_path: str | Path) -> Path | None:
    root = Path(dataset_path)
    direct = root / "WLASL_v0.3.json"
    if direct.exists():
        return direct
    matches = list(root.rglob("WLASL_v0.3.json"))
    return matches[0] if matches else None


def count_videos(path: str | Path) -> int:
    root = Path(path)
    return sum(1 for suffix in ("*.mp4", "*.mkv", "*.webm", "*.mov") for _ in root.glob(suffix))


def make_link_or_copy(source: Path, target: Path, copy: bool = False) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, target)
        return
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError:
        # Windows without symlink permission often lands here. Copying a 5GB
        # dataset by surprise is worse than asking the caller to choose --copy.
        raise OSError(f"Could not symlink {target} -> {source}. Re-run with --copy if you really want a full copy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and locate the Kaggle processed WLASL dataset")
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--link-root", default="data/kaggle_videos")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--no-link", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub first: pip install kagglehub") from exc

    dataset_path = Path(kagglehub.dataset_download(args.dataset))
    videos_dir = find_videos_dir(dataset_path)
    index_file = find_index_file(dataset_path)

    linked_root = None
    if not args.no_link:
        linked_root = Path(args.link_root)
        make_link_or_copy(videos_dir, linked_root, copy=args.copy)

    result = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "videos_dir": str(videos_dir),
        "data_root": str(linked_root or videos_dir),
        "index_file": str(index_file) if index_file else None,
        "video_count": count_videos(linked_root or videos_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
