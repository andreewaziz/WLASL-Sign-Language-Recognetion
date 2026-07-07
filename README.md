# I3D Streamlit Sign Recognition App

Standalone Streamlit UI for the packed WLASL2000 RGB I3D model.

## What Is Included

- `app.py`: Streamlit interface.
- `i3d_predictor.py`: reusable model wrapper class.
- `i3d_modern/`: copied inference/model code.
- `wlasl_i3d_pretrained.pt`: pretrained WLASL2000 I3D checkpoint.
- `wlasl_class_list.txt`: class id to gloss labels.
- `data/splits/nslt_2000.json`: split file used to show actual labels when a dataset video filename matches a WLASL video id.
- `dataset_videos/`: put WLASL clips here for the dataset picker.
- `uploads/`: uploaded videos are saved here.
- `recordings/`: webcam recordings are saved here.

## Run

From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Features

1. Upload a video and run top-k prediction.
2. Browse videos placed in `dataset_videos/`, preview them, and run prediction.
3. Open the camera, record a short sign video, save it, and run prediction.

## Dataset Video Folder

Copy WLASL processed videos into:

```text
dataset_videos/
```

The app scans this folder recursively for:

```text
.mp4, .mov, .avi, .mkv, .webm
```

If the filename stem matches a WLASL video id, for example `00335.mp4`, the UI
shows the actual label from `data/splits/nslt_2000.json`.

## Predictor API

Use the model wrapper directly:

```python
from i3d_predictor import I3DSignPredictor

predictor = I3DSignPredictor()
result = predictor.predict_video("dataset_videos/00335.mp4", top_k=10)
print(result["predicted"])
print(result["top_k"])
```

The model loads once in `I3DSignPredictor.__init__`. The public prediction
method is:

```python
predict_video(video_path, top_k=10)
```

## Camera Notes

Camera recording uses `streamlit-webrtc`. Browser camera access normally works
on `localhost` or HTTPS. If the camera panel does not start, check that:

- dependencies were installed from `requirements.txt`,
- the browser has camera permission,
- no other app is using the webcam.
