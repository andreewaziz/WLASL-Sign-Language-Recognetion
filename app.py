from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st
import torch

from i3d_predictor import I3DSignPredictor, VIDEO_EXTENSIONS

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
DATASET_VIDEO_DIR = APP_DIR / "dataset_videos"
UPLOAD_DIR = APP_DIR / "uploads"
RECORDING_DIR = APP_DIR / "recordings"

for folder in (DATASET_VIDEO_DIR, UPLOAD_DIR, RECORDING_DIR):
    folder.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="WLASL I3D Sign Recognizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      :root {
        --bg: #f6f7f9;
        --surface: #ffffff;
        --surface-2: #f0f2f5;
        --text: #111827;
        --muted: #667085;
        --border: #d9dee7;
        --accent: #0f766e;
        --accent-2: #115e59;
        --danger: #b42318;
        --ok: #067647;
      }

      .stApp {
        background: var(--bg);
        color: var(--text);
      }

      .block-container {
        max-width: 1280px;
        padding-top: 28px;
        padding-bottom: 48px;
      }

      [data-testid="stSidebar"] {
        background: #15171c;
        border-right: 1px solid #262a33;
      }

      [data-testid="stSidebar"] * {
        color: #eef1f5;
      }

            /* Improve contrast for form controls in the sidebar */
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] select,
            [data-testid="stSidebar"] .stTextInput > div > input,
            [data-testid="stSidebar"] .stSelectbox > div > div,
            [data-testid="stSidebar"] .stNumberInput > div > input {
                background: #0b1220 !important;
                color: #eef1f5 !important;
                border: 1px solid #26303a !important;
                border-radius: 8px !important;
            }

            /* Style code / path displays in the sidebar for readability */
            [data-testid="stSidebar"] pre,
            [data-testid="stSidebar"] code,
            [data-testid="stSidebar"] .stCodeBlock {
                background: #0b1220 !important;
                color: #eef1f5 !important;
                border: 1px solid #26303a !important;
                border-radius: 8px !important;
                padding: 8px !important;
                overflow: auto !important;
            }

      [data-testid="stSidebar"] .stSelectbox label,
      [data-testid="stSidebar"] .stSlider label {
        color: #c8ced8;
      }

      .app-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 24px;
        margin-bottom: 22px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--border);
      }

      .app-title {
        margin: 0;
        color: var(--text);
        font-size: 30px;
        line-height: 1.15;
        font-weight: 720;
        letter-spacing: 0;
      }

      .app-subtitle {
        margin-top: 8px;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.5;
        max-width: 760px;
      }

      .status-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 0 0 22px;
      }

      .status-card,
      .result-card,
      .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
      }

      .status-card span,
      .result-card span,
      .info-card span {
        display: block;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 5px;
      }

      .status-card strong,
      .result-card strong,
      .info-card strong {
        color: var(--text);
        font-size: 17px;
        line-height: 1.35;
        word-break: break-word;
      }

      .result-card.ok {
        border-color: #9ee4c5;
        background: #f1fbf6;
      }

      .result-card.bad {
        border-color: #f2b8b5;
        background: #fff5f4;
      }

      .muted-text {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
      }

      .file-path {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        color: #394150;
        font-size: 12px;
        word-break: break-all;
      }

      div.stButton > button {
        border-radius: 8px;
        border: 1px solid #cfd5df;
        background: #ffffff;
        color: #111827;
        font-weight: 620;
      }

      div.stButton > button[kind="primary"] {
        border-color: var(--accent);
        background: var(--accent);
        color: #ffffff;
      }

      div.stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent-2);
      }

      div.stButton > button[kind="primary"]:hover {
        color: #ffffff;
        background: var(--accent-2);
      }

      [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid var(--border);
      }

      [data-testid="stTabs"] [role="tab"] {
        border-radius: 0;
        padding: 12px 14px;
        color: #475467;
        font-weight: 620;
      }

      [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
      }

      .video-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 12px;
      }

      .video-card-name {
        color: var(--text);
        font-size: 13px;
        font-weight: 650;
        margin: 6px 0 8px;
        word-break: break-word;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


class CameraRecorder(VideoProcessorBase):
    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.recording = False
        self.max_frames = 180
        self.lock = threading.Lock()

    def start_recording(self, max_frames: int) -> None:
        with self.lock:
            self.frames = []
            self.max_frames = max_frames
            self.recording = True

    def stop_recording(self) -> list[Any]:
        with self.lock:
            self.recording = False
            return list(self.frames)

    def recv(self, frame: Any) -> Any:
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            if self.recording and len(self.frames) < self.max_frames:
                self.frames.append(image.copy())
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "video.mp4"


def unique_video_path(folder: Path, suffix: str = ".mp4", prefix: str = "video") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder / f"{prefix}_{stamp}{suffix.lower()}"


def save_uploaded_file(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXTENSIONS:
        suffix = ".mp4"
    filename = safe_filename(Path(uploaded_file.name).stem) + suffix
    output_path = UPLOAD_DIR / filename
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def write_recorded_video(frames: list[Any], output_path: Path, fps: int = 15) -> Path:
    if not frames:
        raise ValueError("No frames were recorded.")

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Recording was not written correctly: {output_path}")
    return output_path


@st.cache_resource(show_spinner=False)
def load_predictor(device: str) -> I3DSignPredictor:
    return I3DSignPredictor(device=device)


def get_available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def prediction_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result["top_k"]:
        rows.append(
            {
                "Rank": item["rank"],
                "Class": item["class_id"],
                "Gloss": item["gloss"],
                "Probability": f"{item['probability'] * 100:.2f}%",
            }
        )
    return pd.DataFrame(rows)


def render_result(result: dict[str, Any], key: str = "") -> None:
    predicted = result.get("predicted")
    actual = result.get("actual")
    is_correct = result.get("is_correct")

    if not predicted:
        st.warning("No prediction returned.")
        return

    status_class = "ok" if is_correct is True else "bad" if is_correct is False else ""
    status_text = "Correct" if is_correct is True else "Different" if is_correct is False else "No ground truth"
    actual_label = actual["gloss"] if actual else "Unknown"
    confidence = predicted["probability"] * 100

    st.markdown("#### Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="result-card {status_class}">
              <span>Predicted label</span>
              <strong>{predicted["gloss"]}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="result-card">
              <span>Actual label</span>
              <strong>{actual_label}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="result-card {status_class}">
              <span>Status</span>
              <strong>{status_text} | {confidence:.2f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(predicted["probability"], 0.0), 1.0), text=f"Confidence: {confidence:.2f}%")
    st.dataframe(prediction_dataframe(result), use_container_width=True, hide_index=True)


def predict_and_render(
    predictor: I3DSignPredictor,
    video_path: Path,
    top_k: int,
    key: str,
    auto_crop: bool = False,
) -> None:
    st.video(str(video_path))
    actual = predictor.get_actual_label(video_path)
    if actual:
        st.markdown(
            f"""
            <div class="info-card">
              <span>Dataset label</span>
              <strong>{actual["gloss"]} | class {actual["class_id"]} | {actual.get("subset", "unknown")}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="info-card">
              <span>Dataset label</span>
              <strong>No matching WLASL split entry found for this filename</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<p class="file-path">{video_path}</p>', unsafe_allow_html=True)
    result_state_key = f"result_{key}"

    if st.button("Run prediction", key=key, type="primary", use_container_width=True):
        spinner_text = (
            "Detecting person and cropping, then running I3D inference..."
            if auto_crop
            else "Running I3D inference..."
        )
        with st.spinner(spinner_text):
            result = predictor.predict_video(video_path, top_k=top_k, auto_crop=auto_crop)
        st.session_state[result_state_key] = result

    if result_state_key in st.session_state:
        result = st.session_state[result_state_key]
        if auto_crop and result.get("cropped_video_path"):
            st.caption("Prediction ran on an auto-cropped version of this recording, re-framed around the detected person.")
        render_result(result, key=key)


def render_dataset_picker(predictor: I3DSignPredictor, top_k: int) -> None:
    videos = predictor.list_dataset_videos(DATASET_VIDEO_DIR)
    st.markdown("#### Dataset video folder")
    st.markdown(
        f'<p class="muted-text">Put dataset clips in <span class="file-path">{DATASET_VIDEO_DIR}</span>. The app scans this folder recursively.</p>',
        unsafe_allow_html=True,
    )

    if not videos:
        st.info("No videos found yet. Add `.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm` files to `dataset_videos`.")
        return

    search = st.text_input("Search videos", placeholder="Type part of a filename or label")
    filtered = []
    for path in videos:
        actual = predictor.get_actual_label(path)
        label = actual["gloss"] if actual else ""
        haystack = f"{path.name} {label}".lower()
        if not search or search.lower() in haystack:
            filtered.append(path)

    sort_mode = st.radio("Sort", ["Name", "Newest first", "Oldest first"], horizontal=True)
    if sort_mode == "Newest first":
        filtered = sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)
    elif sort_mode == "Oldest first":
        filtered = sorted(filtered, key=lambda p: p.stat().st_mtime)

    st.caption(f"{len(filtered)} of {len(videos)} videos shown")
    if not filtered:
        st.warning("No videos match the search.")
        return

    selected_name = st.selectbox("Selected video", [str(path.relative_to(DATASET_VIDEO_DIR)) for path in filtered])
    selected_path = DATASET_VIDEO_DIR / selected_name

    preview_count = min(6, len(filtered))
    st.markdown("#### Quick preview")
    columns = st.columns(3)

    def _select_video(p: Path) -> None:
        st.session_state["selected_dataset_video"] = str(p)

    cache = st.session_state.setdefault("dataset_pred_cache", {})

    preview_paths: list[Path] = []
    with st.spinner("Selecting preview videos (checking predictions)..."):
        for path in filtered:
            if len(preview_paths) >= preview_count:
                break
            key = str(path)
            info = cache.get(key)
            if info is None:
                try:
                    res = predictor.predict_video(path, top_k=1)
                    is_correct = bool(res.get("is_correct"))
                    cache[key] = {"is_correct": is_correct, "predicted": res.get("predicted")}
                    info = cache[key]
                except Exception:
                    cache[key] = {"is_correct": False}
                    info = cache[key]
            if info.get("is_correct"):
                preview_paths.append(path)

    if len(preview_paths) < preview_count:
        preview_paths = filtered[:preview_count]

    for index, path in enumerate(preview_paths):
        with columns[index % 3]:
            st.markdown('<div class="video-card">', unsafe_allow_html=True)
            st.video(str(path))
            actual = predictor.get_actual_label(path)
            label = f"{actual['gloss']} | " if actual else ""
            st.markdown(
                f'<div class="video-card-name">{label}{path.name}</div>',
                unsafe_allow_html=True,
            )
            btn_key = f"use_dataset_{index}_{path.name}"
            st.button(
                "Use this video",
                key=btn_key,
                on_click=_select_video,
                args=(path,),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if "selected_dataset_video" in st.session_state:
        selected_path = Path(st.session_state["selected_dataset_video"])

    st.markdown("#### Prediction")
    predict_and_render(predictor, selected_path, top_k, key="predict_dataset")


def render_camera_recorder(predictor: I3DSignPredictor, top_k: int) -> None:
    st.markdown("#### Camera recorder")
    st.markdown(
        '<p class="muted-text">Camera recording works on localhost or HTTPS. Start the stream, record a short sign, then save and run prediction.</p>',
        unsafe_allow_html=True,
    )

    if not WEBRTC_AVAILABLE:
        st.error("Camera recording requires `streamlit-webrtc` and `av`. Install the app requirements first.")
        st.code("pip install -r requirements.txt", language="bash")
        return

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    ctx = webrtc_streamer(
        key="wlasl-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=CameraRecorder,
        async_processing=True,
    )

    duration = st.slider("Max recording length", min_value=2, max_value=12, value=5, step=1)
    fps = st.slider("Saved video FPS", min_value=10, max_value=30, value=15, step=5)
    max_frames = duration * fps

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start recording", type="primary", use_container_width=True):
            if ctx.video_processor:
                ctx.video_processor.start_recording(max_frames=max_frames)
                st.session_state["camera_status"] = "Recording"
            else:
                st.warning("Start the camera stream first.")
    with c2:
        if st.button("Stop and save", use_container_width=True):
            if ctx.video_processor:
                frames = ctx.video_processor.stop_recording()
                if frames:
                    output_path = unique_video_path(RECORDING_DIR, ".mp4", "camera")
                    write_recorded_video(frames, output_path, fps=fps)
                    st.session_state["recorded_video_path"] = str(output_path)
                    st.session_state["camera_status"] = f"Saved {len(frames)} frames"
                else:
                    st.warning("No frames were captured. Start recording and wait a few seconds.")
            else:
                st.warning("Start the camera stream first.")

    st.caption(st.session_state.get("camera_status", "Camera idle"))
    if "recorded_video_path" in st.session_state:
        recorded_path = Path(st.session_state["recorded_video_path"])
        st.markdown("#### Recorded video")
        predict_and_render(predictor, recorded_path, top_k, key="predict_camera", auto_crop=True)


def main() -> None:
    st.markdown(
        """
        <div class="app-header">
          <div>
            <h1 class="app-title">WLASL I3D Sign Recognizer</h1>
            <div class="app-subtitle">
              Run the pretrained RGB I3D WLASL2000 model on uploaded clips,
              selected dataset videos, or short webcam recordings.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Model")
        device = st.selectbox("Device", get_available_devices(), index=0)
        top_k = st.slider("Top-k results", min_value=1, max_value=20, value=10, step=1)
        st.markdown("---")
        st.markdown("### Paths")
        st.caption("Dataset browser folder")
        st.code(str(DATASET_VIDEO_DIR), language="text")
        st.caption("Checkpoint")
        st.code(str(APP_DIR / "wlasl_i3d_pretrained.pt"), language="text")

    try:
        with st.spinner("Loading I3D model..."):
            predictor = load_predictor(device)
    except Exception as exc:
        st.error("The model could not be loaded.")
        st.exception(exc)
        return

    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-card"><span>Model</span><strong>RGB I3D | WLASL2000</strong></div>
          <div class="status-card"><span>Device</span><strong>{predictor.device}</strong></div>
          <div class="status-card"><span>Classes</span><strong>{predictor.num_classes}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_tab, dataset_tab, camera_tab = st.tabs(["Upload video", "Dataset folder", "Camera recording"])

    with upload_tab:
        st.markdown("#### Upload video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS)],
        )
        if uploaded_file is None:
            st.info("Upload a short sign-language clip to run prediction.")
        else:
            video_path = save_uploaded_file(uploaded_file)
            predict_and_render(predictor, video_path, top_k, key="predict_upload")

    with dataset_tab:
        render_dataset_picker(predictor, top_k)

    with camera_tab:
        render_camera_recorder(predictor, top_k)


if __name__ == "__main__":
    main()