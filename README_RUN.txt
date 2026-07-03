Packed WLASL2000 I3D inference bundle.

Setup:
  pip install -r requirements.txt

Predict one video:
  python predict_one.py --video path/to/video.mp4 --top-k 10

Files:
  wlasl_i3d_pretrained.pt   model checkpoint
  wlasl2000.yaml            config used for WLASL2000
  wlasl_class_list.txt      class id to gloss labels
  i3d_modern/               model and inference code
  predict_one.py            standalone prediction script

Note:
  This is a PyTorch model. Joblib is not the right format for this.