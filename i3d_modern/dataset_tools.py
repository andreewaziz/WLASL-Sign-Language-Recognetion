from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


LOGGER = logging.getLogger("i3d_modern.dataset")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass(frozen=True)
class WLASLInstance:
    gloss: str
    url: str
    video_id: str
    frame_start: int
    frame_end: int

    @property
    def is_youtube(self) -> bool:
        return is_youtube_url(self.url)


def load_index(index_path: str | Path) -> list[dict]:
    with Path(index_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_instances(index_path: str | Path) -> Iterable[WLASLInstance]:
    for entry in load_index(index_path):
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            yield WLASLInstance(
                gloss=gloss,
                url=inst["url"],
                video_id=inst["video_id"],
                frame_start=int(inst["frame_start"]),
                frame_end=int(inst["frame_end"]),
            )


def is_youtube_url(url: str) -> bool:
    return "youtube" in url or "youtu.be" in url


def youtube_id_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        video_id = parsed.path.strip("/").split("/")[0]
    elif "youtube" in host:
        query = urllib.parse.parse_qs(parsed.query)
        video_id = query.get("v", [""])[0]
        if not video_id:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[-1] if parts else ""
    else:
        video_id = ""
    return video_id or url.rstrip("/")[-11:]


def extension_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".swf", ".avi"}:
        return suffix
    return ".mp4"


def request_bytes(url: str, referer: str | None = None, timeout: int = 60) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, None, headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_direct_video(instance: WLASLInstance, raw_dir: str | Path, overwrite: bool = False) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    suffix = ".swf" if "aslpro" in instance.url else extension_from_url(instance.url)
    output = raw_path / f"{instance.video_id}{suffix}"
    if output.exists() and not overwrite:
        LOGGER.info("exists: %s", output)
        return output
    referer = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi" if "aslpro" in instance.url else None
    LOGGER.info("downloading direct video %s -> %s", instance.video_id, output)
    data = request_bytes(instance.url, referer=referer)
    output.write_bytes(data)
    return output


def check_youtube_downloader(binary: str) -> None:
    result = subprocess.run([binary, "--version"], text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"{binary} is not available. Install yt-dlp or pass --youtube-downloader.")


def download_youtube_video(instance: WLASLInstance, raw_dir: str | Path, downloader: str = "yt-dlp", overwrite: bool = False) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    video_id = youtube_id_from_url(instance.url)
    existing = list(raw_path.glob(f"{video_id}.*"))
    if existing and not overwrite:
        LOGGER.info("exists: %s", existing[0])
        return existing[0]

    output_template = str(raw_path / "%(id)s.%(ext)s")
    cmd = [downloader, instance.url, "-o", output_template]
    if overwrite:
        cmd.append("--force-overwrites")
    LOGGER.info("downloading youtube video %s", instance.url)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"YouTube download failed for {instance.url}")
    outputs = list(raw_path.glob(f"{video_id}.*"))
    if not outputs:
        raise RuntimeError(f"YouTube download finished but no output found for {video_id}")
    return outputs[0]


def download_dataset(
    index_path: str | Path,
    raw_dir: str | Path,
    *,
    youtube_downloader: str = "yt-dlp",
    skip_youtube: bool = False,
    skip_direct: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
    sleep_min: float = 0.5,
    sleep_max: float = 1.5,
) -> dict[str, int]:
    if not skip_youtube:
        check_youtube_downloader(youtube_downloader)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    for idx, instance in enumerate(iter_instances(index_path), start=1):
        if limit is not None and idx > limit:
            break
        if instance.is_youtube and skip_youtube:
            stats["skipped"] += 1
            continue
        if not instance.is_youtube and skip_direct:
            stats["skipped"] += 1
            continue
        try:
            if instance.is_youtube:
                download_youtube_video(instance, raw_dir, downloader=youtube_downloader, overwrite=overwrite)
            else:
                download_direct_video(instance, raw_dir, overwrite=overwrite)
            stats["downloaded"] += 1
            time.sleep(random.uniform(sleep_min, sleep_max))
        except Exception as exc:
            LOGGER.error("failed video_id=%s url=%s error=%s", instance.video_id, instance.url, exc)
            stats["failed"] += 1
    return stats


def convert_raw_videos(raw_dir: str | Path, mp4_dir: str | Path, overwrite: bool = False, ffmpeg: str = "ffmpeg") -> dict[str, int]:
    raw_path = Path(raw_dir)
    mp4_path = Path(mp4_dir)
    mp4_path.mkdir(parents=True, exist_ok=True)
    stats = {"converted": 0, "copied": 0, "skipped": 0, "failed": 0}

    for src in sorted(raw_path.glob("*")):
        if not src.is_file():
            continue
        dst = mp4_path / f"{src.stem}.mp4"
        if dst.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        if src.suffix.lower() == ".mp4":
            shutil.copyfile(src, dst)
            stats["copied"] += 1
            continue

        cmd = [
            ffmpeg,
            "-y" if overwrite else "-n",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
            str(dst),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode == 0:
            stats["converted"] += 1
        else:
            LOGGER.error("ffmpeg failed for %s: %s", src, result.stderr.strip())
            stats["failed"] += 1
    return stats


def extract_clip(src_video: str | Path, dst_video: str | Path, start_frame: int, end_frame: int, fps: int = 25, overwrite: bool = False) -> bool:
    src = Path(src_video)
    dst = Path(dst_video)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return True
    if end_frame <= 0:
        shutil.copyfile(src, dst)
        return True

    cap = cv2.VideoCapture(str(src))
    try:
        if not cap.isOpened():
            LOGGER.error("could not open source video: %s", src)
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        writer = None
        try:
            for frame_idx in range(max(0, end_frame - start_frame + 1)):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
        if writer is None:
            LOGGER.error("no frames extracted from %s", src)
            return False
        return True
    finally:
        cap.release()


def preprocess_dataset(
    index_path: str | Path,
    raw_dir: str | Path,
    mp4_dir: str | Path,
    videos_dir: str | Path,
    *,
    skip_convert: bool = False,
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    fps: int = 25,
    limit: int | None = None,
) -> dict[str, int]:
    stats = {"created": 0, "skipped": 0, "missing_raw": 0, "failed": 0}
    if not skip_convert:
        convert_stats = convert_raw_videos(raw_dir, mp4_dir, overwrite=overwrite, ffmpeg=ffmpeg)
        LOGGER.info("conversion stats: %s", convert_stats)

    out_dir = Path(videos_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, instance in enumerate(iter_instances(index_path), start=1):
        if limit is not None and idx > limit:
            break
        src_stem = youtube_id_from_url(instance.url) if instance.is_youtube else instance.video_id
        src = Path(mp4_dir) / f"{src_stem}.mp4"
        dst = out_dir / f"{instance.video_id}.mp4"
        if dst.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        if not src.exists():
            stats["missing_raw"] += 1
            continue

        if instance.is_youtube:
            start_frame = max(0, instance.frame_start - 1)
            end_frame = instance.frame_end - 1
            ok = extract_clip(src, dst, start_frame, end_frame, fps=fps, overwrite=overwrite)
        else:
            shutil.copyfile(src, dst)
            ok = True
        if ok:
            stats["created"] += 1
        else:
            stats["failed"] += 1
    return stats


def find_missing_ids(index_path: str | Path, videos_dir: str | Path) -> list[str]:
    filenames = {path.name for path in Path(videos_dir).glob("*.mp4")}
    missing: list[str] = []
    for instance in iter_instances(index_path):
        if f"{instance.video_id}.mp4" not in filenames:
            missing.append(instance.video_id)
    return missing


def setup_logging(log_file: str | Path | None = None, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s")


def default_index_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "WLASL_v0.3.json"


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default=str(default_index_path()))
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--verbose", action="store_true")
