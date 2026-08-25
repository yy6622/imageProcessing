from pathlib import Path

from ultralytics import YOLO


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

IMAGE_PATH = CURRENT_DIR / "image.png"

model = YOLO(
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "fruit_yolo_v3-5"
    / "weights"
    / "best.pt"
)

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    iou=0.5,
    agnostic_nms=True,
    save=True,
    show=True
)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())

        print(
            "Class:",
            model.names[cls_id],
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )