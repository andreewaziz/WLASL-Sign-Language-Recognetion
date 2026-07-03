from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


KINETICS_ID = "1JgTRHGBRCHyHRT_rAF0fOjnfiFefXkEd"
WLASL_I3D_ID = "1jALimVOB69ifYkeT0Pe297S1z4U3jC48"


def gdown_download(file_id: str, output: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown first: pip install gdown") from exc
    url = f"https://drive.google.com/uc?id={file_id}"
    output.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(output), quiet=False, fuzzy=True)


def unpack_if_zip(path: Path, extract_dir: Path) -> Path:
    if zipfile.is_zipfile(path):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        return extract_dir
    return path.parent


def find_one(root: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {patterns} under {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download I3D weights referenced by the WLASL README")
    parser.add_argument("--kind", choices=["kinetics-rgb", "wlasl-pretrained"], required=True)
    parser.add_argument("--output-dir", default="weights")
    parser.add_argument("--work-dir", default="/content/i3d_weight_download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "kinetics-rgb":
        archive = work_dir / "i3d_kinetics_weights"
        gdown_download(KINETICS_ID, archive)
        extracted = unpack_if_zip(archive, work_dir / "kinetics")
        source = find_one(extracted, ["rgb_imagenet.pt"])
        target = output_dir / "rgb_imagenet.pt"
    else:
        archive = work_dir / "wlasl_i3d_pretrained"
        gdown_download(WLASL_I3D_ID, archive)
        extracted = unpack_if_zip(archive, work_dir / "wlasl_pretrained")
        source = find_one(extracted, ["FINAL_nslt_2000*.pt", "*.pt"])
        target = output_dir / "wlasl_i3d_pretrained.pt"

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(json.dumps({"kind": args.kind, "source": str(source), "target": str(target)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
