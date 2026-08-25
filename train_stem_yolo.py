"""Train the fruit-stem YOLO model used by StemDetector.detect_yolo / detect_hybrid.

Usage:
    pip install ultralytics
    python train_stem_yolo.py                      # sensible defaults
    python train_stem_yolo.py --epochs 150 --model yolov8s.pt --imgsz 960

The trained weights land at runs/detect/stem_detector/weights/best.pt.
Copy that file to models/best.pt (the path StemDetector() looks for by
default) once you're happy with it:

    cp runs/detect/stem_detector/weights/best.pt models/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="dataset/data.yaml",
        help="Path to the YOLO data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Base checkpoint to fine-tune. yolov8n.pt is fastest to train "
             "and is a reasonable starting point for a single-class, "
             "small-object task like this; step up to yolov8s.pt if you "
             "have >500 labeled images and want a small accuracy bump.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz", type=int, default=960,
        help="Training resolution. Stems/crowns are small relative to the "
             "whole fruit, so this is deliberately higher than YOLO's usual "
             "640 default - it noticeably helps small-object recall here.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU 0, or 'cpu'")
    parser.add_argument("--name", default="stem_detector")
    parser.add_argument(
        "--patience", type=int, default=25,
        help="Stop early if validation metrics don't improve for this many epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run this script from your project root "
            f"(alongside the dataset/ folder), or pass --data with the full "
            f"path to your data.yaml."
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # Stems have a real, consistent orientation relative to the fruit in
        # most product photos (protruding sideways, or a crown at a
        # consistent-ish spot). Heavy rotation augmentation would train the
        # model on orientations it will rarely see and can hurt small-object
        # accuracy, so it's toned down from the YOLO defaults; flips are left
        # on since a mirrored stem is still a valid stem.
        degrees=10.0,
        fliplr=0.5,
        flipud=0.1,
        mosaic=1.0,
        # Small objects benefit from not being shrunk further by aggressive
        # scale augmentation.
        scale=0.3,
    )

    metrics = model.val()
    print("\nValidation metrics:")
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")

    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Copy them into place with:")
    print(f"  cp {best} models/best.pt")


if __name__ == "__main__":
    main()
