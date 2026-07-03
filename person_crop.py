"""
person_crop.py

Detects the person in a recorded video and writes a re-cropped copy of it,
so the framing (how much of the frame the person fills) matches the WLASL
training clips more closely. WLASL clips are pre-cropped tightly around the
signer; a raw webcam recording usually is not, and that framing mismatch is
the most common reason a model trained on WLASL predicts incorrectly on
webcam footage even when the sign performed is correct.

Usage:
    from person_crop import crop_video_around_person
    cropped_path = crop_video_around_person(input_path, output_path)
    # then run prediction on cropped_path instead of the raw recording
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False


def _collect_landmark_boxes(
    input_path: Path,
    sample_every: int = 2,
    visibility_threshold: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    """Run MediaPipe Pose over the video and return normalized (xmin, ymin, xmax, ymax)
    boxes for every sampled frame where a person was detected."""
    boxes: list[tuple[float, float, float, float]] = []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    xs, ys = [], []
                    for lm in result.pose_landmarks.landmark:
                        if lm.visibility >= visibility_threshold:
                            xs.append(lm.x)
                            ys.append(lm.y)
                    if xs and ys:
                        boxes.append((min(xs), min(ys), max(xs), max(ys)))
            frame_idx += 1
    finally:
        pose.close()
        cap.release()

    return boxes


def _aggregate_box(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    """Combine per-frame boxes into a single stable box for the whole clip,
    using the 5th/95th percentile to ignore momentary detection noise."""
    if not boxes:
        return None
    arr = np.array(boxes)
    xmin = float(np.percentile(arr[:, 0], 5))
    ymin = float(np.percentile(arr[:, 1], 5))
    xmax = float(np.percentile(arr[:, 2], 95))
    ymax = float(np.percentile(arr[:, 3], 95))
    return xmin, ymin, xmax, ymax


def crop_video_around_person(
    input_path: str | Path,
    output_path: str | Path,
    padding_ratio: float = 0.35,
    sample_every: int = 2,
    fallback_center_crop: float = 0.8,
) -> Path:
    """Detect the person across the video and write a cropped copy.

    A single crop box is computed for the whole clip (rather than per-frame)
    to avoid jitter, matching how WLASL source clips use one bounding box per
    video. Padding is added around the detected person so hands raised above
    the shoulders are not cut off.

    If no person is detected at all (e.g. mediapipe unavailable, or nobody
    in frame), falls back to a plain centered crop so the function always
    produces an output video rather than failing.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.release()

    box = None
    if MEDIAPIPE_AVAILABLE:
        boxes = _collect_landmark_boxes(input_path, sample_every=sample_every)
        box = _aggregate_box(boxes)

    if box is None:
        margin = (1.0 - fallback_center_crop) / 2.0
        xmin, ymin, xmax, ymax = margin, margin, 1.0 - margin, 1.0 - margin
    else:
        xmin, ymin, xmax, ymax = box
        box_w = xmax - xmin
        box_h = ymax - ymin
        xmin = max(0.0, xmin - box_w * padding_ratio)
        xmax = min(1.0, xmax + box_w * padding_ratio)
        ymin = max(0.0, ymin - box_h * padding_ratio)
        ymax = min(1.0, ymax + box_h * padding_ratio)

    px_xmin = int(xmin * width)
    px_xmax = int(xmax * width)
    px_ymin = int(ymin * height)
    px_ymax = int(ymax * height)

    px_xmin, px_xmax = sorted((max(0, px_xmin), min(width, px_xmax)))
    px_ymin, px_ymax = sorted((max(0, px_ymin), min(height, px_ymax)))
    if px_xmax - px_xmin < 10 or px_ymax - px_ymin < 10:
        px_xmin, px_ymin, px_xmax, px_ymax = 0, 0, width, height

    cap = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (px_xmax - px_xmin, px_ymax - px_ymin),
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[px_ymin:px_ymax, px_xmin:px_xmax]
            writer.write(cropped)
    finally:
        writer.release()
        cap.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Cropped video was not written correctly: {output_path}")

    return output_path