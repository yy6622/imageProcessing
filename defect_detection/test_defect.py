from pathlib import Path

import cv2 as cv
from ultralytics import YOLO

from defect_detection import detect_defect
from ripeness_detection import classify_ripeness


# -------------------------------------------------
# Paths
# -------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

IMAGE_PATH = CURRENT_DIR / "minor.png"

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "fruit_yolo_v3-5"
    / "weights"
    / "best.pt"
)


# -------------------------------------------------
# Load YOLO model
# -------------------------------------------------

model = YOLO(str(MODEL_PATH))


# -------------------------------------------------
# Load image
# -------------------------------------------------

image = cv.imread(str(IMAGE_PATH))

if image is None:
    print(f"Image not found: {IMAGE_PATH}")
    exit()


# Copy image for final display
display_image = image.copy()


# -------------------------------------------------
# YOLO Detection
# -------------------------------------------------

results = model.predict(
    source=image,
    conf=0.50,
    iou=0.45,
    agnostic_nms=True
)


# -------------------------------------------------
# Defect + Ripeness Detection
# -------------------------------------------------

detected = False

for result in results:

    for i, box in enumerate(result.boxes):

        detected = True

        # -----------------------------------------
        # YOLO information
        # -----------------------------------------

        cls_id = int(
            box.cls.item()
        )

        fruit_type = model.names[
            cls_id
        ]

        confidence = float(
            box.conf.item()
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # -----------------------------------------
        # Crop fruit ROI
        # -----------------------------------------

        roi = image[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:
            continue


        # -----------------------------------------
        # Defect Detection
        # -----------------------------------------

        output, defect_mask, defect_percentage = detect_defect(
            roi,
            fruit_type
        )


        # -----------------------------------------
        # Ripeness Detection
        # -----------------------------------------

        ripeness_result = classify_ripeness(
            roi,
            fruit_type,
            defect_percentage
        )

        ripeness = ripeness_result[
            "ripeness"
        ]


        # -----------------------------------------
        # Print Results
        # -----------------------------------------

        print(
            f"Fruit #{i + 1}: "
            f"{fruit_type}"
        )

        print(
            f"YOLO confidence: "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Defect percentage: "
            f"{defect_percentage:.2f}%"
        )

        print(
            f"Ripeness: "
            f"{ripeness}"
        )

        print(
            "-------------------------"
        )


        # -----------------------------------------
        # Put defect result back into main image
        # -----------------------------------------

        display_image[
            y1:y2,
            x1:x2
        ] = output


        # -----------------------------------------
        # Draw YOLO Bounding Box
        # -----------------------------------------

        cv.rectangle(
            display_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -----------------------------------------
        # Result Label
        # -----------------------------------------

        label = (
            f"{fruit_type.title()} | "
            f"{ripeness} | "
            f"Defect: {defect_percentage:.1f}% | "
            f"Conf: {confidence * 100:.1f}%"
        )


        # Put label above bounding box
        text_y = max(
            y1 - 10,
            25
        )

        cv.putText(
            display_image,
            label,
            (x1, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv.LINE_AA
        )


# -------------------------------------------------
# No Detection
# -------------------------------------------------

if not detected:

    print(
        "No fruit detected."
    )


# -------------------------------------------------
# Show ONLY ONE Window
# -------------------------------------------------

cv.namedWindow(
    "Fruit Quality Result",
    cv.WINDOW_NORMAL
)

cv.resizeWindow(
    "Fruit Quality Result",
    1000,
    700
)

cv.imshow(
    "Fruit Quality Result",
    display_image
)

cv.waitKey(0)
cv.destroyAllWindows()