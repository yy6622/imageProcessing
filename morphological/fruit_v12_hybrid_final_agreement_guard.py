from pathlib import Path
from collections import Counter, defaultdict

import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_MODEL_PATH = PROJECT_ROOT / "runs/detect/fruit_individual_v12/yolo11n_fruit_individual_v12/weights/best.pt"
FEATURE_MODEL_PATH = PROJECT_ROOT / "geometry_texture_model_v12.pkl"
FEATURE_NAMES_PATH = PROJECT_ROOT / "geometry_texture_feature_names_v12.pkl"
SIZE_THRESHOLD_PATH = PROJECT_ROOT / "fruit_size_thresholds.csv"

IMAGE_PATH = PROJECT_ROOT / "dataset" / "se5.jpg"
OUTPUT_PATH = PROJECT_ROOT / "fruit_v12_hybrid_agreement_guard_result.png"

CLASS_NAMES = ["apple", "banana", "orange", "mango", "strawberry"]

YOLO_CONF = 0.05
YOLO_IOU = 0.65
YOLO_IMGSZ = 640
MAX_DET = 100

MIN_BOX_AREA_RATIO = 0.0025
MAX_BOX_ASPECT_RATIO = 7.0

DUPLICATE_IOU = 0.60
DUPLICATE_CONTAINMENT = 0.88

YOLO_HIGH_CONF = 0.70
YOLO_LOW_CONF = 0.30

# Extremely low YOLO detections are treated as possible noise.
# Example:
#   true orange: YOLO 11.83% + feature 57.54% -> KEEP / rescue
#   false banana: YOLO 6.41% + feature 34.86% -> REJECT
HARD_REJECT_YOLO_CONF = 0.20

# General feature confidence required to rescue a very-low YOLO detection.
HARD_RESCUE_FEATURE_CONF = 0.55

# Mango is the weakest class in the Geometry + Texture classifier,
# so require stronger evidence before Mango can rescue/override.
MANGO_RESCUE_MIN_CONF = 0.70

# For YOLO detections between 20% and 30%, feature override requires
# stronger confidence than before.
LOW_YOLO_FEATURE_OVERRIDE_MIN_CONF = 0.55

YOLO_MID_WEIGHT = 0.55
FEATURE_MID_WEIGHT = 0.45

# ============================================================
# UNKNOWN / OTHER FRUIT REJECTION
# ============================================================

UNKNOWN_CLASS_NAME = "unknown"

# Medium-confidence fusion result must reach this score.
# Otherwise the object is treated as unsupported/unknown.
UNKNOWN_MIN_FUSION_SCORE = 0.45

# If YOLO and Geometry+Texture disagree, and both are below
# this confidence, classify as Unknown instead of forcing one
# of the five supported classes.
UNKNOWN_DISAGREE_CONF = 0.60

# ============================================================
# AGREEMENT-AWARE ACCEPTANCE
# ============================================================
#
# If YOLO and Geometry+Texture predict the SAME class,
# we can accept lower-confidence cases more easily because
# two independent model components agree.
#
# Rules:
#   YOLO >= 30% and same class -> accept
#   YOLO 10%-30% and feature >= 40% -> accept
#   YOLO < 10% and feature >= 45% -> accept
#
AGREE_YOLO_DIRECT_MIN = 0.30
AGREE_LOW_YOLO_MIN = 0.10
AGREE_FEATURE_MIN_10_30 = 0.40
AGREE_FEATURE_MIN_UNDER_10 = 0.45

# Extremely weak detections should not be counted at all.
# Unknown objects may still be drawn for user feedback.
DRAW_UNKNOWN_OBJECTS = True

GROUP_S_MAX = 0.82
GROUP_M_MAX = 0.95
GROUP_L_MAX = 1.15


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a, b):
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(a, b):
    inter = intersection_area(a, b)
    smaller = min(box_area(a), box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def valid_box(box, image_w, image_h):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    area_ratio = (w * h) / max(1, image_w * image_h)
    if area_ratio < MIN_BOX_AREA_RATIO:
        return False

    aspect = max(w / h, h / w)
    if aspect > MAX_BOX_ASPECT_RATIO:
        return False

    return True


def remove_duplicate_detections(detections):
    detections = sorted(
        detections,
        key=lambda d: d["yolo_confidence"],
        reverse=True
    )

    kept = []

    for candidate in detections:
        duplicate = False

        for existing in kept:
            iou = box_iou(candidate["box"], existing["box"])
            contain = containment_ratio(candidate["box"], existing["box"])

            if candidate["yolo_class"] == existing["yolo_class"]:
                if iou >= DUPLICATE_IOU or contain >= DUPLICATE_CONTAINMENT:
                    duplicate = True
                    break
            else:
                if iou >= 0.72 or contain >= 0.95:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(candidate)

    return kept


def create_foreground_mask(crop):
    if crop is None or crop.size == 0:
        return None, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    colourful = cv2.inRange(
        hsv,
        np.array([0, 25, 20], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    non_white = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    mask = cv2.bitwise_and(colourful, non_white)

    if np.count_nonzero(mask) < 100:
        mask = colourful.copy()

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) <= 0:
        return None, None

    final_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(final_mask, [contour], -1, 255, -1)

    return final_mask, contour


def extract_geometrical_features(contour, crop_shape):
    crop_h, crop_w = crop_shape[:2]

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if area <= 0 or perimeter <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return None

    aspect_ratio = float(w / h)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    extent = float(area / max(1.0, w * h))

    crop_area = float(max(1, crop_w * crop_h))
    area_ratio = float(area / crop_area)

    diagonal = float(max(1.0, np.hypot(crop_w, crop_h)))
    perimeter_ratio = float(perimeter / diagonal)

    equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))
    short_side = float(max(1, min(crop_w, crop_h)))
    equivalent_diameter_ratio = float(equivalent_diameter / short_side)

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect

    major_axis = float(max(rw, rh, 1.0))
    minor_axis = float(max(min(rw, rh), 1.0))
    long_side = float(max(1, max(crop_w, crop_h)))

    major_axis_ratio = float(major_axis / long_side)
    minor_axis_ratio = float(minor_axis / short_side)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    return {
        "geo_aspect_ratio": aspect_ratio,
        "geo_circularity": circularity,
        "geo_extent": extent,
        "geo_area_ratio": area_ratio,
        "geo_perimeter_ratio": perimeter_ratio,
        "geo_equivalent_diameter_ratio": equivalent_diameter_ratio,
        "geo_major_axis_ratio": major_axis_ratio,
        "geo_minor_axis_ratio": minor_axis_ratio,
        "geo_solidity": solidity,
    }

def extract_texture_features(crop, mask):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    foreground = mask > 0

    if np.count_nonzero(foreground) < 80:
        return None

    fruit_pixels = gray[foreground]
    mean_intensity = float(np.mean(fruit_pixels))
    std_intensity = float(np.std(fruit_pixels))

    isolated_gray = gray.copy()
    isolated_gray[~foreground] = int(round(mean_intensity))

    isolated_gray = cv2.resize(
        isolated_gray,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    quantized = (isolated_gray // 8).astype(np.uint8)
    quantized = np.clip(quantized, 0, 31)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))

    correlation_values = np.nan_to_num(
        graycoprops(glcm, "correlation"),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    correlation = float(np.mean(correlation_values))

    glcm_average = np.mean(glcm, axis=(2, 3))
    nz = glcm_average[glcm_average > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))

    return {
        "tex_contrast": contrast,
        "tex_energy": energy,
        "tex_homogeneity": homogeneity,
        "tex_correlation": correlation,
        "tex_entropy": entropy,
        "tex_mean_intensity": mean_intensity,
        "tex_std_intensity": std_intensity,
    }

def extract_all_features(crop, feature_names):
    mask, contour = create_foreground_mask(crop)

    if mask is None or contour is None:
        return None, None, None

    geometry = extract_geometrical_features(contour, crop.shape)

    if geometry is None:
        return None, None, None

    features = {}
    features.update(geometry)
    texture = extract_texture_features(crop, mask)

    if texture is None:
        return None, None, None

    features.update(texture)

    missing = [name for name in feature_names if name not in features]

    if missing:
        raise ValueError(
            "Current feature model expects missing columns:\n"
            + "\n".join(missing)
        )

    ordered = {
        name: float(features[name])
        for name in feature_names
    }

    return ordered, mask, contour


def predict_feature_class(feature_model, feature_names, features):
    X = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = str(feature_model.predict(X)[0]).strip().lower()

    probabilities = {
        name: 0.0
        for name in CLASS_NAMES
    }

    confidence = 1.0

    if hasattr(feature_model, "predict_proba"):
        probs = feature_model.predict_proba(X)[0]
        model_classes = [
            str(c).strip().lower()
            for c in feature_model.classes_
        ]

        for class_name, probability in zip(model_classes, probs):
            if class_name in probabilities:
                probabilities[class_name] = float(probability)

        confidence = float(probabilities.get(prediction, max(probs)))
    else:
        probabilities[prediction] = 1.0

    return prediction, confidence, probabilities


def fuse_predictions(
    yolo_class,
    yolo_confidence,
    feature_class,
    feature_confidence,
    feature_probabilities
):
    """
    Agreement-aware Hybrid Class Decision.

    Main logic:

    A) If YOLO and Feature classifier AGREE:
       - YOLO >= 30%
         -> accept common class
       - YOLO 10%-30% and Feature >= 40%
         -> accept common class
       - YOLO < 10% and Feature >= 45%
         -> accept common class

    B) If they DISAGREE:
       - high YOLO >= 70% -> trust YOLO
       - very low YOLO < 20% -> feature rescue only if strong
       - low YOLO 20%-30% -> feature override only if strong
       - medium YOLO 30%-70% -> strict unknown/fusion logic

    C) Mango guard:
       Mango remains the weakest Geometry+Texture class,
       so Mango rescue/override requires >= 70%.
    """

    # ========================================================
    # STEP 1: AGREEMENT-AWARE ACCEPTANCE
    # ========================================================
    if yolo_class == feature_class:

        # Strong enough YOLO and both agree.
        if yolo_confidence >= AGREE_YOLO_DIRECT_MIN:
            return (
                yolo_class,
                float(
                    max(
                        yolo_confidence,
                        feature_confidence,
                    )
                ),
                "agreement_accept_yolo_30plus",
                False,
            )

        # YOLO 10%-30%, feature at least 40%.
        if (
            yolo_confidence >= AGREE_LOW_YOLO_MIN
            and
            feature_confidence >= AGREE_FEATURE_MIN_10_30
        ):
            return (
                yolo_class,
                float(
                    max(
                        yolo_confidence,
                        feature_confidence,
                    )
                ),
                "agreement_accept_low_yolo",
                False,
            )

        # YOLO below 10%, feature at least 45%.
        if (
            yolo_confidence < AGREE_LOW_YOLO_MIN
            and
            feature_confidence >= AGREE_FEATURE_MIN_UNDER_10
        ):
            return (
                yolo_class,
                float(
                    max(
                        yolo_confidence,
                        feature_confidence,
                    )
                ),
                "agreement_accept_very_low_yolo",
                False,
            )

        # Same class but still too weak.
        return (
            UNKNOWN_CLASS_NAME,
            float(
                max(
                    yolo_confidence,
                    feature_confidence,
                )
            ),
            "unknown_agreement_but_too_weak",
            False,
        )

    # ========================================================
    # STEP 2: DISAGREEMENT CASES
    # ========================================================

    # --------------------------------------------------------
    # CASE 1: HIGH-CONFIDENCE YOLO
    # --------------------------------------------------------
    if yolo_confidence >= YOLO_HIGH_CONF:
        return (
            yolo_class,
            float(yolo_confidence),
            "high_confidence_yolo",
            False,
        )

    # --------------------------------------------------------
    # CASE 2: VERY LOW YOLO (<20%)
    # --------------------------------------------------------
    if yolo_confidence < HARD_REJECT_YOLO_CONF:

        required_feature_conf = (
            MANGO_RESCUE_MIN_CONF
            if feature_class == "mango"
            else HARD_RESCUE_FEATURE_CONF
        )

        if feature_confidence >= required_feature_conf:
            return (
                feature_class,
                float(feature_confidence),
                (
                    "mango_rescue_very_low_yolo"
                    if feature_class == "mango"
                    else "feature_rescue_very_low_yolo"
                ),
                False,
            )

        return (
            UNKNOWN_CLASS_NAME,
            float(
                max(
                    yolo_confidence,
                    feature_confidence,
                )
            ),
            "unknown_very_low_yolo_disagreement",
            False,
        )

    # --------------------------------------------------------
    # CASE 3: LOW YOLO (20%-30%)
    # --------------------------------------------------------
    if yolo_confidence < YOLO_LOW_CONF:

        required_feature_conf = (
            MANGO_RESCUE_MIN_CONF
            if feature_class == "mango"
            else LOW_YOLO_FEATURE_OVERRIDE_MIN_CONF
        )

        if feature_confidence >= required_feature_conf:
            return (
                feature_class,
                float(feature_confidence),
                (
                    "mango_override_low_yolo"
                    if feature_class == "mango"
                    else "feature_override_low_yolo"
                ),
                False,
            )

        return (
            UNKNOWN_CLASS_NAME,
            float(
                max(
                    yolo_confidence,
                    feature_confidence,
                )
            ),
            "unknown_low_confidence_disagreement",
            False,
        )

    # --------------------------------------------------------
    # CASE 4: MEDIUM YOLO (30%-70%)
    # --------------------------------------------------------

    # If both disagree and neither is strong, Unknown.
    if (
        yolo_confidence < UNKNOWN_DISAGREE_CONF
        and
        feature_confidence < UNKNOWN_DISAGREE_CONF
    ):
        return (
            UNKNOWN_CLASS_NAME,
            float(
                max(
                    yolo_confidence,
                    feature_confidence,
                )
            ),
            "unknown_medium_confidence_disagreement",
            False,
        )

    scores = {
        class_name: 0.0
        for class_name in CLASS_NAMES
    }

    scores[yolo_class] += (
        YOLO_MID_WEIGHT
        *
        yolo_confidence
    )

    for class_name in CLASS_NAMES:
        scores[class_name] += (
            FEATURE_MID_WEIGHT
            *
            feature_probabilities.get(
                class_name,
                0.0
            )
        )

    final_class = max(
        scores,
        key=scores.get
    )

    final_confidence = float(
        scores[final_class]
    )

    if final_confidence < UNKNOWN_MIN_FUSION_SCORE:
        return (
            UNKNOWN_CLASS_NAME,
            final_confidence,
            "unknown_low_fusion_score",
            False,
        )

    return (
        final_class,
        final_confidence,
        "weighted_mid_confidence_fusion",
        False,
    )

def load_size_thresholds():
    if not SIZE_THRESHOLD_PATH.exists():
        return {}

    df = pd.read_csv(SIZE_THRESHOLD_PATH)
    thresholds = {}

    for _, row in df.iterrows():
        fruit_class = str(row["fruit_class"]).strip().lower()

        thresholds[fruit_class] = {
            "feature": str(row["feature"]).strip(),
            "S": float(row["S_max"]),
            "M": float(row["M_max"]),
            "L": float(row["L_max"]),
        }

    return thresholds


def extract_size_geometry(contour, image_shape):
    image_h, image_w = image_shape[:2]

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    x, y, w, h = cv2.boundingRect(contour)

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect

    major_axis = float(max(rw, rh, 1.0))
    minor_axis = float(max(min(rw, rh), 1.0))

    equivalent_diameter = (
        float(np.sqrt(4 * area / np.pi))
        if area > 0
        else 0.0
    )

    long_side = max(image_w, image_h)
    short_side = min(image_w, image_h)

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "width_px": float(w),
        "height_px": float(h),
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "major_axis_ratio": major_axis / max(1, long_side),
        "minor_axis_ratio": minor_axis / max(1, short_side),
        "equivalent_diameter_px": equivalent_diameter,
        "equivalent_diameter_ratio": equivalent_diameter / max(1, short_side),
    }


def classify_dataset_size(fruit_class, geometry, thresholds):
    if fruit_class not in thresholds:
        return "Unknown", None, None

    rule = thresholds[fruit_class]
    feature_name = rule["feature"]

    if feature_name not in geometry:
        return "Unknown", feature_name, None

    value = float(geometry[feature_name])

    if value <= rule["S"]:
        size = "S"
    elif value <= rule["M"]:
        size = "M"
    elif value <= rule["L"]:
        size = "L"
    else:
        size = "XL"

    return size, feature_name, value


def apply_group_relative_size(objects):
    groups = defaultdict(list)

    for obj in objects:
        if obj["final_class"] == UNKNOWN_CLASS_NAME:
            continue

        groups[obj["final_class"]].append(obj)

    for fruit_class, group in groups.items():
        valid = [
            obj
            for obj in group
            if obj.get("size_value") is not None
        ]

        if len(valid) < 2:
            continue

        values = np.array(
            [obj["size_value"] for obj in valid],
            dtype=float
        )

        median = float(np.median(values))

        if median <= 0:
            continue

        for obj in valid:
            ratio = obj["size_value"] / median

            if ratio < GROUP_S_MAX:
                size = "S"
            elif ratio < GROUP_M_MAX:
                size = "M"
            elif ratio <= GROUP_L_MAX:
                size = "L"
            else:
                size = "XL"

            obj["size"] = size
            obj["size_method"] = "group_relative"
            obj["relative_ratio"] = float(ratio)


def draw_result(image, objects):
    output = image.copy()

    for index, obj in enumerate(objects, start=1):
        x1, y1, x2, y2 = obj["box"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        contour = obj["contour_global"]

        cv2.drawContours(
            output,
            [contour],
            -1,
            (255, 0, 255),
            2
        )

        if obj["final_class"] == UNKNOWN_CLASS_NAME:
            label = (
                f"{index}. Unknown/Other "
                f"{obj['final_confidence'] * 100:.1f}%"
            )
        else:
            label = (
                f"{index}. {obj['final_class']} "
                f"{obj['final_confidence'] * 100:.1f}% "
                f"[{obj['size']}]"
            )

        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


def main():
    print_section(
        "V12 YOLO + PURE GEOMETRICAL + TEXTURE + AGREEMENT-AWARE UNKNOWN GUARD"
    )

    for path in [
        YOLO_MODEL_PATH,
        FEATURE_MODEL_PATH,
        FEATURE_NAMES_PATH,
        IMAGE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("Loading V12 YOLO...")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    print("V12 YOLO loaded.")

    print()
    print("Loading PURE Geometrical + Texture V12 feature model...")
    feature_model = joblib.load(FEATURE_MODEL_PATH)
    feature_names = list(joblib.load(FEATURE_NAMES_PATH))
    print("Feature model loaded.")

    print()
    print("Feature columns:")
    for name in feature_names:
        print(f"  - {name}")

    size_thresholds = load_size_thresholds()

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image:\n{IMAGE_PATH}"
        )

    image_h, image_w = image.shape[:2]

    print()
    print(f"Test image: {IMAGE_PATH}")
    print(f"Image size: {image_w} x {image_h}")

    print_section("RUNNING V12 YOLO LOCALISATION")

    result = yolo_model.predict(
        source=image,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=MAX_DET,
        verbose=False
    )[0]

    print(f"Raw YOLO detections: {len(result.boxes)}")

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_w, x2)
        y2 = min(image_h, y2)

        current_box = (x1, y1, x2, y2)

        if not valid_box(
            current_box,
            image_w,
            image_h
        ):
            continue

        detections.append({
            "box": current_box,
            "yolo_class": str(
                yolo_model.names[class_id]
            ).strip().lower(),
            "yolo_confidence": confidence,
        })

    print(f"After basic filtering: {len(detections)}")

    detections = remove_duplicate_detections(detections)

    print(f"After duplicate removal: {len(detections)}")

    objects = []

    for detection_index, detection in enumerate(
        detections,
        start=1
    ):
        x1, y1, x2, y2 = detection["box"]

        crop = image[y1:y2, x1:x2].copy()

        if crop.size == 0:
            continue

        features, mask, contour = extract_all_features(
            crop,
            feature_names
        )

        if features is None:
            print(
                f"Detection {detection_index}: "
                f"feature extraction failed."
            )
            continue

        (
            feature_class,
            feature_confidence,
            feature_probabilities
        ) = predict_feature_class(
            feature_model,
            feature_names,
            features
        )

        (
            final_class,
            final_confidence,
            decision_method,
            rejected
        ) = fuse_predictions(
            detection["yolo_class"],
            detection["yolo_confidence"],
            feature_class,
            feature_confidence,
            feature_probabilities
        )

        if rejected:
            print()
            print(
                f"Rejected detection {detection_index}: "
                f"YOLO={detection['yolo_class']} "
                f"{detection['yolo_confidence'] * 100:.2f}% | "
                f"Feature={feature_class} "
                f"{feature_confidence * 100:.2f}% | "
                f"Reason={decision_method}"
            )
            continue

        geometry = extract_size_geometry(
            contour,
            image.shape
        )

        if final_class == UNKNOWN_CLASS_NAME:
            threshold_size = "Unknown"
            size_feature = None
            size_value = None
        else:
            (
                threshold_size,
                size_feature,
                size_value
            ) = classify_dataset_size(
                final_class,
                geometry,
                size_thresholds
            )

        contour_global = contour.copy()
        contour_global[:, 0, 0] += x1
        contour_global[:, 0, 1] += y1

        objects.append({
            "box": detection["box"],

            "yolo_class": detection["yolo_class"],
            "yolo_confidence": detection["yolo_confidence"],

            "feature_class": feature_class,
            "feature_confidence": feature_confidence,
            "feature_probabilities": feature_probabilities,

            "final_class": final_class,
            "final_confidence": final_confidence,
            "decision_method": decision_method,

            "features": features,
            "geometry": geometry,

            "threshold_size": threshold_size,
            "size": threshold_size,
            "size_feature": size_feature,
            "size_value": size_value,
            "size_method": "dataset_threshold",
            "relative_ratio": None,

            "contour_global": contour_global,
        })

    apply_group_relative_size(objects)

    print_section("FINAL INDIVIDUAL FRUIT RESULTS")

    if not objects:
        print("No valid fruits detected.")

    for index, obj in enumerate(objects, start=1):
        print()
        print("-" * 60)
        print(f"Fruit {index}")

        print()
        print("--- YOLO ---")
        print(f"YOLO class: {obj['yolo_class']}")
        print(
            f"YOLO confidence: "
            f"{obj['yolo_confidence'] * 100:.2f}%"
        )

        print()
        print("--- GEOMETRICAL + TEXTURE CLASSIFIER ---")
        print(f"Feature class: {obj['feature_class']}")
        print(
            f"Feature confidence: "
            f"{obj['feature_confidence'] * 100:.2f}%"
        )

        print()
        print("--- FINAL CLASS ---")
        print(f"FINAL class: {obj['final_class']}")
        print(
            f"FINAL confidence: "
            f"{obj['final_confidence'] * 100:.2f}%"
        )
        print(f"Decision method: {obj['decision_method']}")

        print()
        print("--- GEOMETRICAL FEATURES ---")
        print(
            f"Aspect Ratio: "
            f"{obj['features']['geo_aspect_ratio']:.6f}"
        )
        print(
            f"Circularity: "
            f"{obj['features']['geo_circularity']:.6f}"
        )
        print(
            f"Extent: "
            f"{obj['features']['geo_extent']:.6f}"
        )

        print()
        print("--- TEXTURE FEATURES ---")
        print(
            f"Contrast: "
            f"{obj['features']['tex_contrast']:.6f}"
        )
        print(
            f"Energy: "
            f"{obj['features']['tex_energy']:.6f}"
        )
        print(
            f"Homogeneity: "
            f"{obj['features']['tex_homogeneity']:.6f}"
        )
        print(
            f"Entropy: "
            f"{obj['features']['tex_entropy']:.6f}"
        )
        print(
            f"Mean Intensity: "
            f"{obj['features']['tex_mean_intensity']:.6f}"
        )
        print(
            f"Std Intensity: "
            f"{obj['features']['tex_std_intensity']:.6f}"
        )

        print()
        print("--- SIZE ---")

        if obj["final_class"] == UNKNOWN_CLASS_NAME:
            print("Size skipped: unsupported / unknown fruit")
        else:
            print(f"Threshold Size: {obj['threshold_size']}")
            print(f"FINAL Size: {obj['size']}")
            print(f"Size Feature: {obj['size_feature']}")

            if obj["size_value"] is not None:
                print(
                    f"Size Value: "
                    f"{obj['size_value']:.6f}"
                )

            print(f"Size Method: {obj['size_method']}")

            if obj["relative_ratio"] is not None:
                print(
                    f"Relative Ratio: "
                    f"{obj['relative_ratio']:.3f}"
                )

    fruit_counts = Counter(
        obj["final_class"]
        for obj in objects
    )

    size_counts = Counter(
        obj["size"]
        for obj in objects
        if obj["final_class"] != UNKNOWN_CLASS_NAME
    )

    print_section("FINAL FRUIT COUNT")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name.capitalize():12s}: "
            f"{fruit_counts[class_name]}"
        )

    print(
        f"{'Unknown':12s}: "
        f"{fruit_counts[UNKNOWN_CLASS_NAME]}"
    )

    print_section("FINAL SIZE COUNT")

    for size_name in ["S", "M", "L", "XL", "Unknown"]:
        print(
            f"{size_name:8s}: "
            f"{size_counts[size_name]}"
        )

    annotated = draw_result(
        image,
        objects
    )

    cv2.imwrite(
        str(OUTPUT_PATH),
        annotated
    )

    print_section("TEST COMPLETED")
    print("Result saved as:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
