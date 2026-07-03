from pathlib import Path
import argparse
import json
import torch

from i3d_modern.checkpoint import load_checkpoint, load_model_state
from i3d_modern.data import frames_to_tensor, pad_frames, read_video_frames
from i3d_modern.labels import load_class_names
from i3d_modern.model import build_i3d
from i3d_modern.runtime import get_device
from i3d_modern.transforms import CenterCrop

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    device = get_device(args.device)

    labels = load_class_names(root / "wlasl_class_list.txt")
    model = build_i3d(2000, device=device)

    checkpoint = load_checkpoint(root / "wlasl_i3d_pretrained.pt", device)
    load_model_state(model, checkpoint)
    model.eval()

    frames = read_video_frames(args.video, start=0, num_frames=64, resize_size=256)
    frames = pad_frames(frames, 64)
    frames = CenterCrop(224)(frames)

    inputs = frames_to_tensor(frames).unsqueeze(0).to(device)

    logits = model(inputs).max(dim=2).values
    probs = torch.softmax(logits, dim=1)[0]

    values, indices = probs.topk(min(args.top_k, probs.numel()))
    result = [
        {
            "rank": rank + 1,
            "class_id": int(idx),
            "gloss": labels.get(int(idx), str(int(idx))),
            "probability": float(prob),
        }
        for rank, (prob, idx) in enumerate(zip(values.cpu(), indices.cpu()))
    ]

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()