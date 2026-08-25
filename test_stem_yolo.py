from ultralytics import YOLO

model = YOLO(
    "runs/detect/fruit_stem_detector_v3/weights/best.pt"
)

results = model.predict(
    source="dataset/se1.jpg",
    conf=0.10,
    save=True
)

print("V3 test completed!")