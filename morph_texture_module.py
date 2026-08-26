"""
Morphological + Texture Feature Extraction module for the Streamlit app.

This module is intentionally separated from app.py so the existing UI does not
need to be redesigned.  It uses the CURRENT working V10 detector/model while
V11 is training.  Later, model paths can be replaced in one place only.

Current pipeline:
    YOLO localisation
    -> clipped fruit segmentation
    -> geometrical/morphological features
    -> GLCM texture features
    -> existing feature classifier
    -> contour/geometry measurements
    -> Streamlit-compatible result dictionary

The primary report technique remains:
    Morphological and Texture Feature Extraction

Color features are only loaded when the existing combined model requires them;
they are not presented as the primary technique.
"""

from pathlib import Path
from collections import Counter, defaultdict

import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

from segmentation import segmentation_mask_and_contour


PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# CURRENT WORKING MODELS
# ============================================================
# Keep V10 for now.  When V11 is approved, replace only this path.
YOLO_MODEL_CANDIDATES = [
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "fruit_individual_v10"
    / "yolo11n_fruit_individual_v10"
    / "weights"
    / "best.pt",

    # Future V11 path - automatically preferred once it exists.
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "fruit_individual_v11"
    / "yolo11n_fruit_individual_v11"
    / "weights"
    / "best.pt",
]

# Prefer the CURRENT stable model first. V2/V3 are intentionally not used
# because the Orange/Mango validation result was not good enough.
FEATURE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "combined_model_5class_with_color.pkl",
]

FEATURE_NAMES_CANDIDATES = [
    PROJECT_ROOT / "combined_feature_names_5class_with_color.pkl",
]

SIZE_THRESHOLD_PATH = PROJECT_ROOT / "fruit_size_thresholds.csv"


CLASS_NAMES = [
    "apple",
    "banana",
    "orange",
    "mango",
    "strawberry",
]

# Detection settings - conservative enough for the app.
DEFAULT_CONF = 0.05
YOLO_IOU = 0.65
YOLO_IMGSZ = 640
MAX_DETECTIONS = 100

# Keep small fruits, but reject tiny noise.
MIN_BOX_AREA_RATIO = 0.003
MAX_ASPECT_RATIO = 6.0

# Duplicate filtering.
DUPLICATE_IOU = 0.70
DUPLICATE_CONTAINMENT = 0.92

# Crop / segmentation.
CROP_PADDING = 0.03
YOLO_MASK_INNER_MARGIN_RATIO = 0.015
MIN_FOREGROUND_PIXELS = 100

# Relative size labels for multiple fruits of the same class.
GROUP_S_MAX = 0.82
GROUP_M_MAX = 0.95
GROUP_L_MAX = 1.15


# ============================================================
# MODEL CACHE
# ============================================================

_MODEL_CACHE = {
    "yolo": None,
    "feature_model": None,
    "feature_names": None,
    "size_thresholds": None,
    "paths": {},
}


def _first_existing(paths):
    """Return first existing path, or None."""
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def _load_models():
    """Lazy-load models once per Python process."""
    if _MODEL_CACHE["yolo"] is None:
        # Prefer V11 only AFTER it exists; otherwise use V10.
        v11 = YOLO_MODEL_CANDIDATES[1]
        v10 = YOLO_MODEL_CANDIDATES[0]
        yolo_path = v11 if v11.exists() else v10

        if not yolo_path.exists():
            raise FileNotFoundError(
                "No fruit YOLO model found. Expected one of:\n"
                + "\n".join(str(p) for p in YOLO_MODEL_CANDIDATES)
            )

        _MODEL_CACHE["yolo"] = YOLO(str(yolo_path))
        _MODEL_CACHE["paths"]["yolo"] = str(yolo_path)

    if _MODEL_CACHE["feature_model"] is None:
        feature_model_path = _first_existing(FEATURE_MODEL_CANDIDATES)
        feature_names_path = _first_existing(FEATURE_NAMES_CANDIDATES)

        if feature_model_path is None or feature_names_path is None:
            raise FileNotFoundError(
                "Current Morphological/Texture feature model files are missing."
            )

        _MODEL_CACHE["feature_model"] = joblib.load(feature_model_path)
        _MODEL_CACHE["feature_names"] = list(joblib.load(feature_names_path))
        _MODEL_CACHE["paths"]["feature_model"] = str(feature_model_path)
        _MODEL_CACHE["paths"]["feature_names"] = str(feature_names_path)

    if _MODEL_CACHE["size_thresholds"] is None:
        _MODEL_CACHE["size_thresholds"] = _load_size_thresholds()

    return (
        _MODEL_CACHE["yolo"],
        _MODEL_CACHE["feature_model"],
        _MODEL_CACHE["feature_names"],
        _MODEL_CACHE["size_thresholds"],
    )


def get_active_model_info():
    """Useful for Streamlit captions/debugging."""
    _load_models()
    return dict(_MODEL_CACHE["paths"])


# ============================================================
# BOX HELPERS
# ============================================================

def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _box_iou(a, b):
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b):
    inter = _intersection_area(a, b)
    smaller = min(_box_area(a), _box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def _valid_box(box, image_w, image_h):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    area_ratio = (width * height) / max(1, image_w * image_h)
    if area_ratio < MIN_BOX_AREA_RATIO:
        return False

    aspect = max(width / height, height / width)
    if aspect > MAX_ASPECT_RATIO:
        return False

    return True


def _remove_duplicates(detections):
    """Keep the highest-confidence localisation for obvious duplicates."""
    detections = sorted(
        detections,
        key=lambda item: item["yolo_confidence"],
        reverse=True,
    )

    kept = []

    for candidate in detections:
        duplicate = False

        for existing in kept:
            iou = _box_iou(candidate["box"], existing["box"])
            contain = _containment(candidate["box"], existing["box"])

            if iou >= DUPLICATE_IOU or contain >= DUPLICATE_CONTAINMENT:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ============================================================
# CROP + SEGMENTATION
# ============================================================

def _get_crop(image, box):
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = box

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_x = int(width * CROP_PADDING)
    pad_y = int(height * CROP_PADDING)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(image_w, x2 + pad_x)
    cy2 = min(image_h, y2 + pad_y)

    return (
        image[cy1:cy2, cx1:cx2].copy(),
        (cx1, cy1, cx2, cy2),
    )


def _raw_mask_contour(crop):
    try:
        mask, contour = segmentation_mask_and_contour(crop)
    except Exception:
        return None, None

    if mask is None or contour is None:
        return None, None

    mask = (mask > 0).astype(np.uint8) * 255

    if cv2.contourArea(contour) <= 0:
        return None, None

    return mask, contour


def _clipped_mask_contour(crop, original_box, crop_box):
    """
    Segment the expanded crop, then clip the mask to the original YOLO box.

    This is the same idea used in the tested geometry/texture pipeline to
    reduce neighbouring-fruit leakage in overlapping scenes.
    """
    raw_mask, raw_contour = _raw_mask_contour(crop)

    if raw_mask is None:
        return None, None

    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = original_box

    lx1 = x1 - crop_x1
    ly1 = y1 - crop_y1
    lx2 = x2 - crop_x1
    ly2 = y2 - crop_y1

    width = max(1, lx2 - lx1)
    height = max(1, ly2 - ly1)

    margin_x = int(width * YOLO_MASK_INNER_MARGIN_RATIO)
    margin_y = int(height * YOLO_MASK_INNER_MARGIN_RATIO)

    lx1 = max(0, lx1 + margin_x)
    ly1 = max(0, ly1 + margin_y)
    lx2 = min(crop.shape[1], lx2 - margin_x)
    ly2 = min(crop.shape[0], ly2 - margin_y)

    if lx2 <= lx1 or ly2 <= ly1:
        return raw_mask, raw_contour

    allowed = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.rectangle(allowed, (lx1, ly1), (lx2, ly2), 255, -1)

    clipped = cv2.bitwise_and(raw_mask, allowed)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, kernel, iterations=1)
    clipped = cv2.morphologyEx(clipped, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        clipped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return raw_mask, raw_contour

    # Largest contour is normally the current fruit inside its YOLO box.
    best = max(contours, key=cv2.contourArea)

    if cv2.contourArea(best) <= 0:
        return raw_mask, raw_contour

    final_mask = np.zeros_like(clipped)
    cv2.drawContours(final_mask, [best], -1, 255, -1)

    return final_mask, best


# ============================================================
# GEOMETRICAL / MORPHOLOGICAL FEATURES
# ============================================================

def _extract_morphology(crop, box, crop_box):
    mask, contour = _clipped_mask_contour(crop, box, crop_box)

    if contour is None:
        return None, None, None

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area <= 0 or perimeter <= 0:
        return None, None, None

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return None, None, None

    aspect_ratio = w / h
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    extent = area / (w * h)

    features = {
        "geo_aspect_ratio": float(aspect_ratio),
        "geo_circularity": float(circularity),
        "geo_extent": float(extent),
    }

    return features, mask, contour


# ============================================================
# GLCM TEXTURE FEATURES
# ============================================================

def _extract_texture(crop):
    """
    Keep the same texture feature definition as the CURRENT feature model,
    so the existing .pkl remains compatible while V11/V4 training is pending.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    quantized = (gray // 8).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True,
    )

    probabilities = glcm[:, :, 0, 0]
    nonzero = probabilities[probabilities > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    return {
        "tex_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "tex_energy": float(graycoprops(glcm, "energy")[0, 0]),
        "tex_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "tex_entropy": float(entropy),
        "tex_mean_intensity": float(np.mean(gray)),
        "tex_std_intensity": float(np.std(gray)),
    }


# ============================================================
# COMPATIBILITY COLOR FEATURES
# ============================================================

def _extract_color_compat(crop, mask):
    """
    Compatibility only.

    The current combined classifier was trained with color columns too.
    We therefore compute them if its feature_names require them.  The app
    still presents this module as Morphological + Texture Feature Extraction.
    """
    foreground = mask > 0

    if int(foreground.sum()) < MIN_FOREGROUND_PIXELS:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    H = hsv[:, :, 0][foreground]
    S = hsv[:, :, 1][foreground]
    V = hsv[:, :, 2][foreground]

    L = lab[:, :, 0][foreground]
    A = lab[:, :, 1][foreground]
    B = lab[:, :, 2][foreground]

    return {
        "color_mean_h": float(np.mean(H)),
        "color_mean_s": float(np.mean(S)),
        "color_mean_v": float(np.mean(V)),
        "color_std_h": float(np.std(H)),
        "color_std_s": float(np.std(S)),
        "color_std_v": float(np.std(V)),
        "color_mean_l": float(np.mean(L)),
        "color_mean_a": float(np.mean(A)),
        "color_mean_b": float(np.mean(B)),
        "color_std_l": float(np.std(L)),
        "color_std_a": float(np.std(A)),
        "color_std_b": float(np.std(B)),
    }


def _extract_all_features(crop, box, crop_box, feature_names):
    morph, mask, contour = _extract_morphology(crop, box, crop_box)

    if morph is None or mask is None or contour is None:
        return None, None, None

    texture = _extract_texture(crop)

    features = {}
    features.update(morph)
    features.update(texture)

    # Only calculate compatibility color features when the current model
    # explicitly requests them.
    if any(str(name).startswith("color_") for name in feature_names):
        color = _extract_color_compat(crop, mask)
        if color is None:
            return None, None, None
        features.update(color)

    missing = [name for name in feature_names if name not in features]

    if missing:
        raise ValueError(
            "Feature model expects unsupported columns: "
            + ", ".join(str(x) for x in missing)
        )

    ordered = {
        name: float(features[name])
        for name in feature_names
    }

    return ordered, mask, contour


# ============================================================
# FEATURE CLASSIFIER
# ============================================================

def _predict_feature_class(feature_model, feature_names, features):
    vector = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = str(feature_model.predict(vector)[0]).strip().lower()
    confidence = 1.0
    probabilities = {name: 0.0 for name in CLASS_NAMES}

    if hasattr(feature_model, "predict_proba"):
        probs = feature_model.predict_proba(vector)[0]
        classes = [str(c).strip().lower() for c in feature_model.classes_]

        for class_name, probability in zip(classes, probs):
            if class_name in probabilities:
                probabilities[class_name] = float(probability)

        confidence = float(probabilities.get(prediction, max(probs)))

    return prediction, confidence, probabilities


def _choose_final_class(
    yolo_class,
    yolo_confidence,
    feature_class,
    feature_confidence,
):
    """
    CURRENT safe fusion while V11 is still training.

    Strong YOLO results are preserved.
    Very weak YOLO results can be rescued by Morphological/Texture features.
    This avoids allowing the old feature model to override good YOLO apple
    detections while still letting it help on weak detections.
    """
    if yolo_confidence >= 0.50:
        return yolo_class, yolo_confidence, "YOLO (high confidence)"

    if yolo_class == feature_class:
        combined = min(
            1.0,
            0.55 * yolo_confidence
            + 0.45 * feature_confidence
            + 0.15,
        )
        return yolo_class, combined, "YOLO + Morphology/Texture agreement"

    if yolo_confidence < 0.20 and feature_confidence >= 0.45:
        return (
            feature_class,
            feature_confidence,
            "Morphology/Texture correction",
        )

    # Medium disagreement: use the stronger evidence conservatively.
    yolo_score = 0.65 * yolo_confidence
    feature_score = 0.35 * feature_confidence

    if yolo_score >= feature_score:
        return yolo_class, yolo_confidence, "YOLO (medium confidence)"

    return feature_class, feature_confidence, "Morphology/Texture (medium confidence)"


# ============================================================
# SIZE / GEOMETRY
# ============================================================

def _load_size_thresholds():
    if not SIZE_THRESHOLD_PATH.exists():
        return {}

    df = pd.read_csv(SIZE_THRESHOLD_PATH)
    thresholds = {}

    for _, row in df.iterrows():
        fruit = str(row["fruit_class"]).strip().lower()
        thresholds[fruit] = {
            "feature": str(row["feature"]).strip(),
            "S": float(row["S_max"]),
            "M": float(row["M_max"]),
            "L": float(row["L_max"]),
        }

    return thresholds


def _geometry_measurements(contour, image_shape):
    image_h, image_w = image_shape[:2]

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    x, y, w, h = cv2.boundingRect(contour)

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect

    major_axis = float(max(rw, rh, 1.0))
    minor_axis = float(max(min(rw, rh), 1.0))

    equivalent_diameter = float(
        np.sqrt(4.0 * area / np.pi)
    ) if area > 0 else 0.0

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


def _classify_threshold_size(fruit_class, geometry, thresholds):
    rule = thresholds.get(fruit_class)

    if not rule:
        return "Unknown", None, None

    feature_name = rule["feature"]
    value = geometry.get(feature_name)

    if value is None:
        return "Unknown", feature_name, None

    if value <= rule["S"]:
        size = "S"
    elif value <= rule["M"]:
        size = "M"
    elif value <= rule["L"]:
        size = "L"
    else:
        size = "XL"

    return size, feature_name, float(value)


def _apply_relative_sizes(objects):
    groups = defaultdict(list)

    for obj in objects:
        groups[obj["fruit_type"]].append(obj)

    for fruit_type, group in groups.items():
        valid = [
            obj
            for obj in group
            if obj.get("size_value") is not None
        ]

        if len(valid) < 2:
            continue

        values = np.array(
            [obj["size_value"] for obj in valid],
            dtype=float,
        )

        median = float(np.median(values))

        if median <= 0:
            continue

        for obj in valid:
            ratio = obj["size_value"] / median

            if ratio < GROUP_S_MAX:
                final_size = "S"
            elif ratio < GROUP_M_MAX:
                final_size = "M"
            elif ratio <= GROUP_L_MAX:
                final_size = "L"
            else:
                final_size = "XL"

            obj["size_class"] = final_size
            obj["size_method"] = "group_relative"
            obj["relative_size_ratio"] = float(ratio)


# ============================================================
# ANNOTATION
# ============================================================

def _make_isolated_crop(crop, mask):
    isolated = crop.copy()

    # White background makes the fruit itself easier to inspect in Streamlit.
    white = np.full_like(isolated, 255)
    white[mask > 0] = isolated[mask > 0]

    return white


def _draw_annotation(image, objects):
    """
    Keep the image readable:
    - contour
    - bounding box
    - only short '#n class size' label
    """
    output = image.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        contour_global = obj["contour_global"]

        cv2.drawContours(
            output,
            [contour_global],
            -1,
            (255, 0, 255),
            2,
        )

        short_label = (
            f"#{obj['index'] + 1} "
            f"{obj['fruit_type']} "
            f"{obj.get('size_class', '?')}"
        )

        cv2.putText(
            output,
            short_label,
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# PUBLIC APP ENTRYPOINT
# ============================================================

def inspect_image_morph_texture(
    image,
    calibration=None,
    yolo_confidence=None,
):
    """
    Streamlit-compatible entrypoint.

    Returns the same main keys that app.py already expects:
        original
        annotated
        objects
        count
        summary
        calibration_method
        calibration_confidence
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    yolo_model, feature_model, feature_names, thresholds = _load_models()

    image_h, image_w = image.shape[:2]

    conf = (
        float(yolo_confidence)
        if yolo_confidence is not None
        else DEFAULT_CONF
    )

    # The app's old slider starts at 0.05. Do not force a higher threshold
    # here because the current V10 model has some valid low-confidence fruits.
    conf = max(0.01, min(conf, 0.90))

    prediction = yolo_model.predict(
        source=image,
        conf=conf,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        agnostic_nms=False,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )[0]

    detections = []

    for raw_box in prediction.boxes:
        yolo_confidence_value = float(raw_box.conf[0].item())
        class_id = int(raw_box.cls[0].item())
        yolo_class = str(
            yolo_model.names[class_id]
        ).strip().lower()

        x1, y1, x2, y2 = map(
            int,
            raw_box.xyxy[0].tolist(),
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_w, x2)
        y2 = min(image_h, y2)

        box = (x1, y1, x2, y2)

        if not _valid_box(
            box,
            image_w,
            image_h,
        ):
            continue

        detections.append({
            "box": box,
            "yolo_class": yolo_class,
            "yolo_confidence": yolo_confidence_value,
        })

    detections = _remove_duplicates(detections)

    objects = []

    for detection in detections:
        crop, crop_box = _get_crop(
            image,
            detection["box"],
        )

        if crop.size == 0:
            continue

        features, mask, contour = _extract_all_features(
            crop,
            detection["box"],
            crop_box,
            feature_names,
        )

        if features is None:
            continue

        (
            feature_class,
            feature_confidence,
            feature_probabilities,
        ) = _predict_feature_class(
            feature_model,
            feature_names,
            features,
        )

        (
            final_class,
            final_confidence,
            classification_method,
        ) = _choose_final_class(
            detection["yolo_class"],
            detection["yolo_confidence"],
            feature_class,
            feature_confidence,
        )

        geometry = _geometry_measurements(
            contour,
            image.shape,
        )

        (
            threshold_size,
            size_feature,
            size_value,
        ) = _classify_threshold_size(
            final_class,
            geometry,
            thresholds,
        )

        crop_x1, crop_y1, _, _ = crop_box

        contour_global = contour.copy()
        contour_global[:, 0, 0] += crop_x1
        contour_global[:, 0, 1] += crop_y1

        isolated = _make_isolated_crop(
            crop,
            mask,
        )

        # Optional physical calibration while retaining app compatibility.
        width_cm = None
        height_cm = None
        area_cm2 = None

        if calibration is not None and getattr(
            calibration,
            "is_calibrated",
            False,
        ):
            try:
                width_cm = float(
                    calibration.px_to_cm(
                        geometry["width_px"]
                    )
                )
                height_cm = float(
                    calibration.px_to_cm(
                        geometry["height_px"]
                    )
                )
                area_cm2 = float(
                    geometry["area_px"]
                    * (
                        calibration.px_to_cm(1.0)
                        ** 2
                    )
                )
            except Exception:
                width_cm = None
                height_cm = None
                area_cm2 = None

        obj = {
            "index": len(objects),

            "box": detection["box"],
            "crop_box": crop_box,
            "crop_isolated": isolated,

            # App-compatible fruit classification.
            "fruit_type": final_class,
            "fruit_type_confidence": float(final_confidence),

            # Keep quality fields empty; this module is type/shape/texture,
            # not Fresh/Rotten quality classification.
            "label": None,
            "confidence": 0.0,
            "defect_fraction": 0.0,

            # Current source predictions.
            "yolo_class": detection["yolo_class"],
            "yolo_confidence": detection["yolo_confidence"],
            "feature_class": feature_class,
            "feature_confidence": feature_confidence,
            "feature_probabilities": feature_probabilities,
            "classification_method": classification_method,

            # Main Morphological features.
            "geo_aspect_ratio": features.get("geo_aspect_ratio"),
            "geo_circularity": features.get("geo_circularity"),
            "geo_extent": features.get("geo_extent"),

            # Main Texture features.
            "tex_contrast": features.get("tex_contrast"),
            "tex_energy": features.get("tex_energy"),
            "tex_homogeneity": features.get("tex_homogeneity"),
            "tex_entropy": features.get("tex_entropy"),
            "tex_mean_intensity": features.get("tex_mean_intensity"),
            "tex_std_intensity": features.get("tex_std_intensity"),

            # Geometry / size.
            **geometry,
            "threshold_size": threshold_size,
            "size_class": threshold_size,
            "size_feature": size_feature,
            "size_value": size_value,
            "size_method": "dataset_threshold",
            "relative_size_ratio": None,

            "width_cm": width_cm,
            "height_cm": height_cm,
            "area_cm2": area_cm2,

            # Drawing helpers.
            "contour_global": contour_global,

            # app.py checks cls.error. Use None so existing code is safe.
            "classification": None,
        }

        objects.append(obj)

    # Relative size only compares fruits of the same final class.
    _apply_relative_sizes(objects)

    annotated = _draw_annotation(
        image,
        objects,
    )

    # Existing app summary_to_text expects nested dicts.
    summary = {}

    counts = Counter(
        obj["fruit_type"]
        for obj in objects
    )

    for fruit_type, count in counts.items():
        summary[fruit_type] = {
            "Detected": int(count)
        }

    calibration_method = "None"
    calibration_confidence = "N/A"

    if calibration is not None:
        calibration_method = getattr(
            calibration,
            "method",
            "None",
        )
        calibration_confidence = getattr(
            calibration,
            "confidence",
            "N/A",
        )

    return {
        "original": image.copy(),
        "annotated": annotated,
        "objects": objects,
        "count": len(objects),
        "summary": summary,
        "calibration_method": calibration_method,
        "calibration_confidence": calibration_confidence,
        "technique": "Morphological and Texture Feature Extraction",
        "model_info": get_active_model_info(),
    }
