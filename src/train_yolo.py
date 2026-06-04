"""
Train YOLO26 instance segmentation on VizWiz.
Run convert_vizwiz_to_yolo.py once before this to build data_yolo/.

Tuned for GTX 1060 6GB: yolo26s-seg, imgsz=512, batch=4, AMP on.
Lower batch or use yolo26n-seg if OOM. Raise batch to 8 if VRAM holds.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_yolo/vizwiz.yaml",
                        help="path to dataset yaml produced by convert_vizwiz_to_yolo.py")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                        help="yolo26n-seg.pt (smallest) | yolo26s-seg.pt | yolo26m-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' to force CPU")
    parser.add_argument("--project", default="runs/yolo_vizwiz")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run convert_vizwiz_to_yolo.py first."
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=True,
        project=args.project,
        name=args.name,
        patience=20,
        save_period=10,
    )


if __name__ == "__main__":
    main()
