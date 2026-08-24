from pathlib import Path

from ultralytics import YOLO


# Folder containing this train_yolo.py
CURRENT_DIR = Path(__file__).resolve().parent

# Main project folder
PROJECT_ROOT = CURRENT_DIR.parent

# Dataset
DATA_PATH = (
    CURRENT_DIR
    / "yolo_dataset"
    / "roboflow_export"
    / "data.yaml"
)

# Pretrained YOLO model
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"


print("Dataset path:", DATA_PATH)
print("Model path:", MODEL_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"data.yaml not found at: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"yolov8n.pt not found at: {MODEL_PATH}"
    )


model = YOLO(str(MODEL_PATH))

model.train(
    data=str(DATA_PATH),
    epochs=50,
    imgsz=640,
    patience=10,
    name="fruit_yolo_v4"
)