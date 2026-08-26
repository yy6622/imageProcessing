import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from segmentation import (
    segmentation_mask_and_contour,
    segment_all_objects,
    detect_defect_fraction,
    contour_shape_metrics,
    compute_texture_roughness,
    DEFAULT_DEFECT_AREA_FRACTION,
    DEFAULT_DEFECT_LOW_FRACTION,
)

import preprocessing as prep
from calibration import CalibrationResult, uncalibrated

DEFAULT_IMAGE_SIZE = (512, 512)

FRUIT_DEBUG = os.environ.get("FRUIT_DEBUG", "0").lower() not in ("0", "", "false", "no", "off")


def _dbg(*args):
    if FRUIT_DEBUG:
        print("[FRUIT_DEBUG]", *args)


# ======================================================
# Detection result + display/crop helpers
# ======================================================
@dataclass
class DetectionResult:
    found: bool
    bbox: Optional[tuple] = None          # (x, y, w, h) in pixels
    contour: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    area_px: float = 0.0
    perimeter_px: float = 0.0


DEDUPE_IOU_THRESHOLD = 0.5  # mask-overlap fraction above which two detections are treated as the same physical fruit


def _dedupe_overlapping_detections(objects, iou_threshold=DEDUPE_IOU_THRESHOLD):
    ranked = sorted(objects, key=lambda o: o["fruit_type_confidence"], reverse=True)
    kept = []
    for obj in ranked:
        mask = obj["detection"].mask
        is_dup = False
        for kept_obj in kept:
            kept_mask = kept_obj["detection"].mask
            if mask is None or kept_mask is None:
                continue
            inter = int(np.logical_and(mask > 0, kept_mask > 0).sum())
            union = int(np.logical_or(mask > 0, kept_mask > 0).sum())
            iou = (inter / union) if union > 0 else 0.0
            if iou > iou_threshold:
                is_dup = True
                _dbg(f"dedupe: DROP {obj.get('fruit_type')} (conf={obj['fruit_type_confidence']:.3f}) "
                     f"-- overlaps {kept_obj.get('fruit_type')} (conf={kept_obj['fruit_type_confidence']:.3f}) "
                     f"at IoU={iou:.2f}")
                break
        if not is_dup:
            kept.append(obj)
    kept.sort(key=lambda o: o["index"])
    return kept


def draw_detections(image, objects, box_color=(0, 255, 0), contour_color=(0, 165, 255)):
    annotated = image.copy()
    for obj in objects:
        detection = obj["detection"]
        if not detection.found:
            continue
        x, y, w, h = detection.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.drawContours(annotated, [detection.contour], -1, contour_color, 2)

        fruit_type = obj.get("fruit_type") or "?"
        quality = obj.get("label") or "?"
        text = f"#{obj['index'] + 1} {fruit_type} {quality}"
        text_y = max(y - 8, 15)
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


CROP_EXTRA_ERODE_PIXELS = 6  # applied only to isolate=True crops, on top of


def crop_object(image, detection: DetectionResult, pad_frac=0.06, isolate=False):

    if not detection.found:
        return image.copy()

    x, y, w, h = detection.bbox
    pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)

    src = image
    if isolate and detection.mask is not None:
        mask = detection.mask
        if CROP_EXTRA_ERODE_PIXELS > 0:
            kernel = np.ones((CROP_EXTRA_ERODE_PIXELS, CROP_EXTRA_ERODE_PIXELS), np.uint8)
            mask = cv2.erode(mask, kernel)
        src = cv2.bitwise_and(image, image, mask=mask)

    return src[y0:y1, x0:x1].copy()


# ======================================================
# Classification result
# ======================================================
@dataclass
class ClassificationResult:
    fruit_type: Optional[str] = None        # e.g. "Apple" / "Banana" / "Orange" — from YOLO's own label
    fruit_type_confidence: float = 0.0      # YOLO's box confidence
    type_source: str = "YOLO"               # "YOLO" or "CNN override" -- which model actually produced fruit_type
    label: Optional[str] = None             # quality: Fresh / Unripe / Rotten — PRIMARY result, from the
                                             # Colour Feature Extraction KNN (train_color_knn.py)
    confidence: float = 0.0                 # KNN's confidence for the predicted quality label
    quality_backend: str = "color_knn"      # "color_knn" (primary) -- kept as a field for clarity
    error: Optional[str] = None             # e.g. "No colour-feature model found for fruit type 'Apple'"
    defect_fraction: float = 0.0            # fraction of the fruit's OWN surface flagged as a dark
    defect_override: bool = False           # True if defect_fraction crossed the threshold and


COLOR_KNN_MODELS_DIR = "color_knn_models"
_color_knn_model_cache = {}


def _load_color_knn_model(fruit_type, models_dir=COLOR_KNN_MODELS_DIR):
    key = (os.path.abspath(models_dir) if os.path.isdir(models_dir) else models_dir, fruit_type)
    if key in _color_knn_model_cache:
        return _color_knn_model_cache[key]

    path = os.path.join(models_dir, f"{fruit_type}.joblib")
    if not os.path.isfile(path):
        _color_knn_model_cache[key] = (None, None, None)
        return None, None, None

    try:
        import train_color_knn as _color_knn_module
        scaler, model, classes = _color_knn_module.load_color_knn_model(path)
    except Exception as e:  # missing joblib/sklearn, corrupt file, etc.
        print(f"[colorDetection] Colour-feature KNN model at {path} unavailable ({e}); "
              f"colour-based ripeness will be reported unavailable for '{fruit_type}'.")
        _color_knn_model_cache[key] = (None, None, None)
        return None, None, None

    _color_knn_model_cache[key] = (scaler, model, classes)
    return scaler, model, classes


def classify_quality_color_knn(crop_bgr, fruit_type, models_dir=COLOR_KNN_MODELS_DIR):
    """Colour Feature Extraction ripeness classifier: LAB+HSV colour moments -> KNN.
    Uses only explicit colour statistics, no raw pixels, shape, or texture."""
    scaler, model, classes = _load_color_knn_model(fruit_type, models_dir)
    if scaler is None or crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, {}
    import train_color_knn as _color_knn_module
    return _color_knn_module.predict_ripeness_from_color(crop_bgr, scaler, model, classes)



# ======================================================
# Fruit-specific quality safety rules
# ======================================================
# These conservative rules supplement KNN. Every threshold may be tuned through
# an environment variable after reviewing FRUIT_DEBUG logs on your own dataset.
FRUIT_QUALITY_RULES = {
    "Apple": {
        "fresh_min": float(os.environ.get("APPLE_FRESH_COLOUR_MIN", "0.35")),
        "fresh_max_unripe": float(os.environ.get("APPLE_FRESH_MAX_GREEN", "0.35")),
        "unripe_min": float(os.environ.get("APPLE_UNRIPE_GREEN_MIN", "0.50")),
        "unripe_max_ripe": float(os.environ.get("APPLE_UNRIPE_MAX_RIPE", "0.25")),
        "rotten_defect": float(os.environ.get("APPLE_ROTTEN_DEFECT_MIN", "0.060")),
        "brown_min": float(os.environ.get("APPLE_ROTTEN_BROWN_MIN", "0.025")),
        "dark_min": float(os.environ.get("APPLE_ROTTEN_DARK_MIN", "0.012")),
    },
    "Orange": {
        "fresh_min": float(os.environ.get("ORANGE_FRESH_ORANGE_MIN", "0.45")),
        "fresh_max_unripe": float(os.environ.get("ORANGE_FRESH_MAX_GREEN", "0.25")),
        "unripe_min": float(os.environ.get("ORANGE_UNRIPE_GREEN_MIN", "0.35")),
        "unripe_max_ripe": float(os.environ.get("ORANGE_UNRIPE_MAX_ORANGE", "0.35")),
        "rotten_defect": float(os.environ.get("ORANGE_ROTTEN_DEFECT_MIN", "0.070")),
        "brown_min": float(os.environ.get("ORANGE_ROTTEN_BROWN_MIN", "0.050")),
        "dark_min": float(os.environ.get("ORANGE_ROTTEN_DARK_MIN", "0.020")),
    },
    "Banana": {
        "fresh_min": float(os.environ.get("BANANA_FRESH_YELLOW_MIN", "0.45")),
        "fresh_max_unripe": float(os.environ.get("BANANA_FRESH_MAX_GREEN", "0.25")),
        "unripe_min": float(os.environ.get("BANANA_UNRIPE_GREEN_MIN", "0.35")),
        "unripe_max_ripe": float(os.environ.get("BANANA_UNRIPE_MAX_YELLOW", "0.30")),
        "rotten_defect": float(os.environ.get("BANANA_ROTTEN_DEFECT_MIN", "0.055")),
        "brown_min": float(os.environ.get("BANANA_ROTTEN_BROWN_MIN", "0.080")),
        "dark_min": float(os.environ.get("BANANA_ROTTEN_DARK_MIN", "0.025")),
    },
    "Strawberry": {
        "fresh_min": float(os.environ.get("STRAWBERRY_FRESH_RED_MIN", "0.45")),
        "fresh_max_unripe": float(os.environ.get("STRAWBERRY_FRESH_MAX_PALE", "0.40")),
        "unripe_min": float(os.environ.get("STRAWBERRY_UNRIPE_PALE_MIN", "0.45")),
        "unripe_max_ripe": float(os.environ.get("STRAWBERRY_UNRIPE_MAX_RED", "0.28")),
        "rotten_defect": float(os.environ.get("STRAWBERRY_ROTTEN_DEFECT_MIN", "0.070")),
        "brown_min": float(os.environ.get("STRAWBERRY_ROTTEN_BROWN_MIN", "0.060")),
        "dark_min": float(os.environ.get("STRAWBERRY_ROTTEN_DARK_MIN", "0.035")),
    },
    "Mango": {
        # Direct Mango rule requested by the project:
        # black -> Rotten, green -> Unripe, yellow/orange/red -> Fresh.
        "fresh_min": float(os.environ.get("MANGO_FRESH_RIPE_COLOUR_MIN", "0.30")),
        "fresh_max_unripe": float(os.environ.get("MANGO_FRESH_MAX_GREEN", "0.50")),
        "unripe_min": float(os.environ.get("MANGO_UNRIPE_GREEN_MIN", "0.30")),
        "unripe_max_ripe": float(os.environ.get("MANGO_UNRIPE_MAX_RIPE", "0.50")),
        "rotten_defect": float(os.environ.get("MANGO_ROTTEN_DEFECT_MIN", "1.0")),
        "brown_min": float(os.environ.get("MANGO_ROTTEN_BROWN_MIN", "1.0")),
        "dark_min": float(os.environ.get("MANGO_ROTTEN_BLACK_MIN", "0.015")),
    },
}


def _fraction(condition, denominator):
    return float(np.count_nonzero(condition)) / max(1, int(denominator))


def _fruit_surface_evidence(bgr, mask, fruit_type):
    """Measure ripe, unripe, brown and dark surface colour for one fruit."""
    foreground = mask > 0
    foreground_count = int(foreground.sum())
    empty = {
        "ripe_fraction": 0.0, "unripe_fraction": 0.0,
        "brown_dark_fraction": 0.0, "very_dark_fraction": 0.0,
        "red_fraction": 0.0, "yellow_fraction": 0.0,
        "green_fraction": 0.0, "pale_fraction": 0.0,
    }
    if foreground_count == 0:
        return empty

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colourful = saturation >= 45

    red = foreground & colourful & ((hue < 12) | (hue >= 170)) & (value >= 55)
    orange = (
        foreground & (saturation >= 55)
        & (hue >= 6) & (hue < 22) & (value >= 75)
    )
    yellow = (
        foreground & (saturation >= 50)
        & (hue >= 18) & (hue < 38) & (value >= 80)
    )
    green = (
        foreground & (saturation >= 45)
        & (hue >= 35) & (hue < 90) & (value >= 40)
    )
    pale = foreground & (
        ((saturation < 85) & (value >= 115))
        | ((hue >= 15) & (hue < 40) & (saturation < 155) & (value >= 105))
    )

    if fruit_type == "Banana":
        brown_dark = (
            foreground & (saturation >= 45)
            & (hue >= 5) & (hue < 25) & (value < 135)
        )
    elif fruit_type == "Orange":
        brown_dark = (
            foreground & (saturation >= 45)
            & (hue >= 5) & (hue < 20) & (value < 100)
        )
    elif fruit_type == "Strawberry":
        # Do not count every naturally dark strawberry seed as decay.
        brown_dark = (
            foreground & (saturation >= 40)
            & (hue >= 8) & (hue < 25) & (value < 105)
        )
    elif fruit_type == "Mango":
        brown_dark = (
            foreground & (saturation >= 40)
            & (hue >= 7) & (hue < 24) & (value < 110)
        )
    else:
        brown_dark = (
            foreground & (saturation >= 55)
            & (hue >= 8) & (hue < 23) & (value < 125)
        )

    very_dark = foreground & (value < 42)

    reported_green = green
    if fruit_type == "Apple":
        ripe, unripe = red | yellow, green
    elif fruit_type == "Orange":
        ripe, unripe = orange | yellow, green
    elif fruit_type == "Banana":
        ripe, unripe = yellow, green
    elif fruit_type == "Strawberry":
        # A small green calyx is far below the conservative unripe threshold.
        ripe, unripe = red, pale | green
    elif fruit_type == "Mango":
        # Green mango skin is often yellow-green (Hue 27..34), which the
        # generic green range misses. Keep that range out of ripe yellow.
        mango_green = (
            foreground & (saturation >= 35)
            & (hue >= 27) & (hue < 90) & (value >= 35)
        )
        mango_yellow_or_orange = (
            foreground & (saturation >= 45)
            & (hue >= 7) & (hue < 27) & (value >= 70)
        )
        ripe, unripe = red | mango_yellow_or_orange, mango_green
        reported_green = mango_green
    else:
        ripe = unripe = np.zeros_like(foreground)

    return {
        "ripe_fraction": _fraction(ripe, foreground_count),
        "unripe_fraction": _fraction(unripe, foreground_count),
        "brown_dark_fraction": _fraction(brown_dark, foreground_count),
        "very_dark_fraction": _fraction(very_dark, foreground_count),
        "red_fraction": _fraction(red, foreground_count),
        "yellow_fraction": _fraction(yellow, foreground_count),
        "green_fraction": _fraction(reported_green, foreground_count),
        "pale_fraction": _fraction(pale, foreground_count),
    }


def _conditional_probability(color_probs, label):
    candidates = {
        key: float(value) for key, value in color_probs.items()
        if key != "Rotten"
    }
    total = sum(candidates.values())
    return float(candidates.get(label, 0.0) / total) if total > 1e-9 else 0.0


def _rule_confidence(label, evidence, rules, color_probs):
    strength = (
        evidence["ripe_fraction"] if label == "Fresh"
        else evidence["unripe_fraction"]
    )
    threshold = rules["fresh_min"] if label == "Fresh" else rules["unripe_min"]
    colour_confidence = np.clip(
        0.55 + 0.35 * (strength - threshold) / max(1e-6, 1.0 - threshold),
        0.55, 0.90,
    )
    return float(max(colour_confidence, _conditional_probability(color_probs, label)))


def _best_non_rotten_quality(color_probs, evidence, rules):
    strong_fresh = (
        evidence["ripe_fraction"] >= rules["fresh_min"]
        and evidence["unripe_fraction"] <= rules["fresh_max_unripe"]
    )
    strong_unripe = (
        evidence["unripe_fraction"] >= rules["unripe_min"]
        and evidence["ripe_fraction"] <= rules["unripe_max_ripe"]
    )
    if strong_fresh and not strong_unripe:
        return "Fresh", _rule_confidence("Fresh", evidence, rules, color_probs)
    if strong_unripe and not strong_fresh:
        return "Unripe", _rule_confidence("Unripe", evidence, rules, color_probs)

    candidates = {
        label: float(probability) for label, probability in color_probs.items()
        if label != "Rotten"
    }
    if not candidates:
        return None, 0.0
    label = max(candidates, key=candidates.get)
    total = sum(candidates.values())
    confidence = candidates[label] / total if total > 1e-9 else 0.0
    return label, float(confidence)


def _orange_mold_evidence(bgr, mask):
    """Detect a connected white/gray mold patch with an optional green center."""
    foreground_u8 = (mask > 0).astype(np.uint8)
    foreground_count = int(foreground_u8.sum())
    if foreground_count == 0:
        return {
            "white_gray_fraction": 0.0,
            "mold_green_fraction": 0.0,
            "largest_mold_fraction": 0.0,
            "visible_mold": False,
        }

    # Ignore the contour boundary where white background commonly leaks in.
    ys, xs = np.where(foreground_u8 > 0)
    box_size = max(
        int(ys.max()) - int(ys.min()) + 1,
        int(xs.max()) - int(xs.min()) + 1,
        1,
    )
    margin = max(2, int(round(box_size * 0.025)))
    margin = min(margin, 11)
    kernel_size = margin * 2 + 1
    interior = cv2.erode(
        foreground_u8 * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ),
    ) > 0
    if int(interior.sum()) < foreground_count * 0.35:
        interior = foreground_u8 > 0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    # Exclude very bright specular highlights (V > 230). Mold in the supplied
    # images is white/gray with moderate value and often has a dull green core.
    white_gray = (
        interior
        & (saturation < 70)
        & (value >= 80)
        & (value <= 230)
    )
    mold_green = (
        interior
        & (hue >= 30)
        & (hue <= 95)
        & (saturation >= 18)
        & (saturation <= 190)
        & (value >= 45)
        & (value <= 215)
    )

    candidate = ((white_gray | mold_green) * 255).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate, 8
    )
    largest_area = 0
    for component in range(1, component_count):
        largest_area = max(largest_area, int(stats[component, cv2.CC_STAT_AREA]))

    white_fraction = float(white_gray.sum()) / foreground_count
    green_fraction = float(mold_green.sum()) / foreground_count
    largest_fraction = float(largest_area) / foreground_count
    visible_mold = (
        largest_fraction >= 0.080
        and white_fraction >= 0.040
        and (green_fraction >= 0.006 or white_fraction >= 0.140)
    )
    return {
        "white_gray_fraction": white_fraction,
        "mold_green_fraction": green_fraction,
        "largest_mold_fraction": largest_fraction,
        "visible_mold": bool(visible_mold),
    }


def _apply_quality_safety_rules(classification, color_probs, bgr, mask):
    """Apply separate Apple/Orange/Banana/Strawberry/Mango quality rules."""
    defect_fraction = detect_defect_fraction(bgr, mask)
    classification.defect_fraction = defect_fraction
    if classification.label is None:
        return

    fruit_type = classification.fruit_type
    rules = FRUIT_QUALITY_RULES.get(fruit_type)
    if rules is None:
        if defect_fraction >= DEFAULT_DEFECT_AREA_FRACTION:
            classification.label = "Rotten"
            classification.defect_override = True
        return

    original_label = classification.label
    evidence = _fruit_surface_evidence(bgr, mask, fruit_type)
    roughness = compute_texture_roughness(bgr, mask)

    if fruit_type == "Apple":
        # Apple stems, calyxes and normal red/yellow patches often look like
        # dark local defects. Rules must therefore never turn a KNN Fresh or
        # Unripe apple into Rotten. They only validate/reject a KNN Rotten call.
        apple_rotten_evidence = (
            (
                defect_fraction >= 0.100
                and evidence["brown_dark_fraction"] >= 0.080
            )
            or (
                defect_fraction >= 0.070
                and evidence["brown_dark_fraction"] >= 0.050
                and roughness >= 35.0
            )
            or (
                defect_fraction >= 0.180
                and evidence["brown_dark_fraction"] >= 0.030
            )
        )
        # Pure yellow/yellow-green without meaningful red is treated as
        # Unripe for this project's Apple labels. Red, or a true red+yellow
        # bicolour surface, is Fresh.
        apple_fresh_colour = (
            evidence["red_fraction"] >= 0.30
            or (
                evidence["red_fraction"] >= 0.08
                and evidence["yellow_fraction"] >= 0.08
                and (
                    evidence["red_fraction"] + evidence["yellow_fraction"]
                    >= 0.30
                )
            )
        )
        apple_unripe_colour = (
            evidence["red_fraction"] < 0.18
            and (
                evidence["green_fraction"] + evidence["yellow_fraction"]
                >= 0.42
            )
        )
        _dbg(
            "Apple conservative quality rule:",
            f"knn={original_label}",
            f"ripe={evidence['ripe_fraction']:.4f}",
            f"red={evidence['red_fraction']:.4f}",
            f"yellow={evidence['yellow_fraction']:.4f}",
            f"green={evidence['green_fraction']:.4f}",
            f"brown={evidence['brown_dark_fraction']:.4f}",
            f"defect={defect_fraction:.4f}",
            f"roughness={roughness:.2f}",
            f"rotten_evidence={apple_rotten_evidence}",
            f"fresh_colour={apple_fresh_colour}",
            f"unripe_colour={apple_unripe_colour}",
        )

        if original_label == "Rotten":
            if not apple_rotten_evidence:
                if apple_unripe_colour:
                    replacement = "Unripe"
                    confidence = _rule_confidence(
                        "Unripe", evidence, rules, color_probs
                    )
                elif apple_fresh_colour:
                    replacement = "Fresh"
                    confidence = _rule_confidence(
                        "Fresh", evidence, rules, color_probs
                    )
                else:
                    replacement, confidence = _best_non_rotten_quality(
                        color_probs, evidence, rules
                    )
                if replacement is not None:
                    classification.label = replacement
                    classification.confidence = confidence
                    classification.defect_override = True
            return

        # Fresh <-> Unripe may still be corrected by a very clear surface
        # colour, but Apple Rotten is never introduced by this rule.
        if apple_unripe_colour and not apple_fresh_colour and original_label != "Unripe":
            classification.label = "Unripe"
            classification.confidence = _rule_confidence(
                "Unripe", evidence, rules, color_probs
            )
            classification.defect_override = True
        elif apple_fresh_colour and not apple_unripe_colour and original_label != "Fresh":
            classification.label = "Fresh"
            classification.confidence = _rule_confidence(
                "Fresh", evidence, rules, color_probs
            )
            classification.defect_override = True
        return

    if fruit_type == "Orange":
        # Dark orange caused by basket shadows is not decay. Like Apple, the
        # Orange rule never changes a KNN Fresh/Unripe prediction to Rotten; it
        # only decides whether an existing KNN Rotten call has enough evidence.
        orange_fresh_colour = (
            evidence["ripe_fraction"] >= 0.38
            and evidence["green_fraction"] < 0.30
        )
        orange_unripe_colour = (
            evidence["green_fraction"] >= 0.35
            and evidence["ripe_fraction"] < 0.35
        )
        mold = _orange_mold_evidence(bgr, mask)
        orange_rotten_evidence = mold["visible_mold"] or (
            (
                defect_fraction >= 0.120
                and evidence["brown_dark_fraction"] >= 0.120
                and roughness >= 32.0
            )
            or (
                defect_fraction >= 0.200
                and evidence["brown_dark_fraction"] >= 0.080
            )
            or (
                defect_fraction >= 0.100
                and evidence["brown_dark_fraction"] >= 0.180
                and evidence["very_dark_fraction"] >= 0.025
            )
        )

        _dbg(
            "Orange conservative quality rule:",
            f"knn={original_label}",
            f"orange_yellow={evidence['ripe_fraction']:.4f}",
            f"green={evidence['green_fraction']:.4f}",
            f"brown={evidence['brown_dark_fraction']:.4f}",
            f"dark={evidence['very_dark_fraction']:.4f}",
            f"defect={defect_fraction:.4f}",
            f"roughness={roughness:.2f}",
            f"mold_white={mold['white_gray_fraction']:.4f}",
            f"mold_green={mold['mold_green_fraction']:.4f}",
            f"mold_largest={mold['largest_mold_fraction']:.4f}",
            f"visible_mold={mold['visible_mold']}",
            f"rotten_evidence={orange_rotten_evidence}",
            f"fresh_colour={orange_fresh_colour}",
            f"unripe_colour={orange_unripe_colour}",
        )

        if mold["visible_mold"]:
            classification.label = "Rotten"
            classification.confidence = float(max(
                color_probs.get("Rotten", 0.0),
                min(0.97, 0.70 + mold["largest_mold_fraction"]),
            ))
            classification.defect_override = original_label != "Rotten"
            return

        if original_label == "Rotten":
            if not orange_rotten_evidence:
                if orange_fresh_colour:
                    replacement = "Fresh"
                    confidence = _rule_confidence(
                        "Fresh", evidence, rules, color_probs
                    )
                elif orange_unripe_colour:
                    replacement = "Unripe"
                    confidence = _rule_confidence(
                        "Unripe", evidence, rules, color_probs
                    )
                else:
                    replacement, confidence = _best_non_rotten_quality(
                        color_probs, evidence, rules
                    )
                if replacement is not None:
                    classification.label = replacement
                    classification.confidence = confidence
                    classification.defect_override = True
            return

        if orange_unripe_colour and not orange_fresh_colour and original_label != "Unripe":
            classification.label = "Unripe"
            classification.confidence = _rule_confidence(
                "Unripe", evidence, rules, color_probs
            )
            classification.defect_override = True
        elif orange_fresh_colour and not orange_unripe_colour and original_label != "Fresh":
            classification.label = "Fresh"
            classification.confidence = _rule_confidence(
                "Fresh", evidence, rules, color_probs
            )
            classification.defect_override = True
        return

    if fruit_type == "Mango":
        black_mango = evidence["very_dark_fraction"] >= rules["dark_min"]
        green_mango = (
            evidence["green_fraction"] >= rules["unripe_min"]
            and evidence["green_fraction"] > evidence["ripe_fraction"]
        )
        yellow_orange_red_mango = (
            evidence["ripe_fraction"] >= rules["fresh_min"]
            and evidence["ripe_fraction"] >= evidence["green_fraction"]
        )
        _dbg(
            "Mango direct colour rule:",
            f"knn={original_label}",
            f"black={evidence['very_dark_fraction']:.4f}",
            f"green={evidence['green_fraction']:.4f}",
            f"yellow_orange_red={evidence['ripe_fraction']:.4f}",
            f"black_mango={black_mango}",
            f"green_mango={green_mango}",
            f"ripe_mango={yellow_orange_red_mango}",
        )

        if black_mango:
            classification.label = "Rotten"
            classification.confidence = float(max(
                color_probs.get("Rotten", 0.0),
                min(0.95, 0.60 + evidence["very_dark_fraction"] * 3.0),
            ))
            classification.defect_override = original_label != "Rotten"
            return
        if green_mango:
            classification.label = "Unripe"
            classification.confidence = _rule_confidence(
                "Unripe", evidence, rules, color_probs
            )
            classification.defect_override = original_label != "Unripe"
            return
        if yellow_orange_red_mango:
            classification.label = "Fresh"
            classification.confidence = _rule_confidence(
                "Fresh", evidence, rules, color_probs
            )
            classification.defect_override = original_label != "Fresh"
            return

        # Ambiguous colour may use KNN, but Rotten is never accepted without
        # the black-surface condition above.
        if original_label == "Rotten":
            replacement, confidence = _best_non_rotten_quality(
                color_probs, evidence, rules
            )
            if replacement is not None:
                classification.label = replacement
                classification.confidence = confidence
                classification.defect_override = True
        return

    brown_lesion = (
        evidence["brown_dark_fraction"] >= rules["brown_min"]
        and defect_fraction >= 0.015
    )
    dark_lesion = (
        evidence["very_dark_fraction"] >= rules["dark_min"]
        and defect_fraction >= 0.010
    )
    large_brown_region = evidence["brown_dark_fraction"] >= max(
        0.12, rules["brown_min"] * 2.5
    )
    texture_lesion = (
        roughness >= 30.0
        and defect_fraction >= 0.025
        and evidence["brown_dark_fraction"] >= rules["brown_min"] * 0.35
    )

    # Protect normal red/yellow apple skin from false L-channel defect evidence.
    normal_red_yellow_apple = (
        fruit_type == "Apple"
        and evidence["red_fraction"] >= 0.06
        and evidence["yellow_fraction"] >= 0.06
        and not (brown_lesion or dark_lesion or texture_lesion)
    )
    strong_defect = (
        defect_fraction >= rules["rotten_defect"]
        and not normal_red_yellow_apple
    )
    has_rotten_evidence = (
        strong_defect or brown_lesion or dark_lesion
        or large_brown_region or texture_lesion
    )

    strong_fresh = (
        evidence["ripe_fraction"] >= rules["fresh_min"]
        and evidence["unripe_fraction"] <= rules["fresh_max_unripe"]
    )
    strong_unripe = (
        evidence["unripe_fraction"] >= rules["unripe_min"]
        and evidence["ripe_fraction"] <= rules["unripe_max_ripe"]
    )

    _dbg(
        f"{fruit_type} quality guard:",
        f"knn={original_label}",
        f"ripe={evidence['ripe_fraction']:.4f}",
        f"unripe={evidence['unripe_fraction']:.4f}",
        f"brown={evidence['brown_dark_fraction']:.4f}",
        f"dark={evidence['very_dark_fraction']:.4f}",
        f"defect={defect_fraction:.4f}",
        f"roughness={roughness:.2f}",
        f"rotten_evidence={has_rotten_evidence}",
        f"strong_fresh={strong_fresh}",
        f"strong_unripe={strong_unripe}",
    )

    if has_rotten_evidence:
        classification.label = "Rotten"
        classification.confidence = float(max(
            color_probs.get("Rotten", 0.0),
            min(0.95, 0.60 + defect_fraction * 2.0),
        ))
        classification.defect_override = original_label != "Rotten"
        return

    # A KNN Rotten label without local decay evidence is not accepted.
    if original_label == "Rotten":
        replacement, confidence = _best_non_rotten_quality(
            color_probs, evidence, rules
        )
        if replacement is not None:
            classification.label = replacement
            classification.confidence = confidence
            classification.defect_override = True
        return

    if strong_unripe and not strong_fresh and original_label != "Unripe":
        classification.label = "Unripe"
        classification.confidence = _rule_confidence(
            "Unripe", evidence, rules, color_probs
        )
        classification.defect_override = True
    elif strong_fresh and not strong_unripe and original_label != "Fresh":
        classification.label = "Fresh"
        classification.confidence = _rule_confidence(
            "Fresh", evidence, rules, color_probs
        )
        classification.defect_override = True


CNN_TYPE_MODELS_DIR = "cnn_type_models"
_cnn_type_model_cache = {}


def _load_cnn_type_model(models_dir=CNN_TYPE_MODELS_DIR):
    key = os.path.abspath(models_dir) if os.path.isdir(models_dir) else models_dir
    if key in _cnn_type_model_cache:
        return _cnn_type_model_cache[key]

    path = os.path.join(models_dir, "fruit_type.pt")
    if not os.path.isfile(path):
        _cnn_type_model_cache[key] = (None, None)
        return None, None

    try:
        import train_fruit_type as _type_module
        model, classes = _type_module.load_type_model(path)
    except Exception as e:  # missing torch, corrupt checkpoint, etc.
        print(f"[colorDetection] CNN fruit-type model at {path} unavailable ({e}); "
              f"fallback (non-YOLO) fruit detection will be skipped.")
        _cnn_type_model_cache[key] = (None, None)
        return None, None

    _cnn_type_model_cache[key] = (model, classes)
    return model, classes


def classify_fruit_type_cnn(crop_bgr, models_dir=CNN_TYPE_MODELS_DIR):
    model, classes = _load_cnn_type_model(models_dir)
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, {}
    import train_fruit_type as _type_module
    # train_fruit_type owns its preprocessing. Passing the raw isolated crop
    # avoids the old double-resize that destroyed Mango/Orange aspect ratio.
    return _type_module.predict_fruit_type_with_probs(model, classes, crop_bgr)


# ======================================================
# YOLO detection + fruit type
# ======================================================
FALLBACK_ONLY_SPECIES = {"Mango", "Strawberry"}

STRAWBERRY_OVERRIDE_MIN_CONF = 0.9   # every real strawberry test photo measured 1.000; no observed false positive at any confidence
MANGO_OVERRIDE_MIN_CONF = 0.6        # lower than strawberry's bar on purpose -- this alone is NOT trusted; see MANGO_OVERRIDE_MIN_ASPECT, both must pass
MANGO_OVERRIDE_MIN_ASPECT = 1.5      # local contour's long/short side ratio. Raised from 1.3 after a live
MANGO_OVERRIDE_MAX_ROUGHNESS = 19.5  # compute_texture_roughness() (segmentation.py) -- mean Sobel gradient
MANGO_OVERRIDE_MIN_CONF_DOUBLE = 0.5  # lower floor used ONLY when BOTH aspect_ok and roughness_ok pass --

YOLO_FRUIT_CLASS_NAMES = {"apple", "banana", "orange", "mango", "strawberry"}
YOLO_CONFIDENCE_THRESHOLD = 0.5

DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"
FINE_TUNED_YOLO_WEIGHTS = os.path.join("yolo_fruit_models", "best.pt")


def _resolve_yolo_weights(requested):
    if requested == DEFAULT_YOLO_WEIGHTS and os.path.isfile(FINE_TUNED_YOLO_WEIGHTS):
        return FINE_TUNED_YOLO_WEIGHTS
    return requested


FINE_TUNED_YOLO_IMGSZ = 416


_yolo_model_cache = {}


def _load_yolo_model(weights):
    if weights not in _yolo_model_cache:
        from ultralytics import YOLO
        _yolo_model_cache[weights] = YOLO(weights)
    return _yolo_model_cache[weights]


def inspect_image_yolo(
    image_or_path,
    calibration: Optional[CalibrationResult] = None,
    image_size=DEFAULT_IMAGE_SIZE,
    denoise_method="median",
    enhance_method="clahe",
    erode_pixels=10,
    yolo_weights=DEFAULT_YOLO_WEIGHTS,
    yolo_confidence=YOLO_CONFIDENCE_THRESHOLD,
):
    if isinstance(image_or_path, str):
        original = cv2.imread(image_or_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_or_path}")
    else:
        original = image_or_path

    original = cv2.resize(original, image_size)
    preprocessed = prep.preprocess_image(original, denoise_method=denoise_method, enhance_method=enhance_method)

    if calibration is None:
        calibration = uncalibrated()

    resolved_weights = _resolve_yolo_weights(yolo_weights)
    yolo_model = _load_yolo_model(resolved_weights)  # lazy-imports ultralytics; see that function
    predict_imgsz = FINE_TUNED_YOLO_IMGSZ if resolved_weights == FINE_TUNED_YOLO_WEIGHTS else 640
    yolo_results = yolo_model.predict(original, conf=yolo_confidence, imgsz=predict_imgsz, verbose=False)

    from collections import defaultdict

    summary = defaultdict(lambda: defaultdict(int))
    objects = []
    i = 0
    for r in yolo_results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            yolo_label = names.get(cls_id, str(cls_id))
            if yolo_label not in YOLO_FRUIT_CLASS_NAMES:
                continue
            yolo_conf = float(box.conf[0])
            x1f, y1f, x2f, y2f = box.xyxy[0].tolist()
            _dbg(f"main loop: YOLO box label={yolo_label} conf={yolo_conf:.3f} "
                 f"xyxy=({x1f:.0f},{y1f:.0f},{x2f:.0f},{y2f:.0f})")

            h_img, w_img = original.shape[:2]
            pad = 0.08
            bw, bh = x2f - x1f, y2f - y1f
            x0 = max(0, int(x1f - pad * bw))
            y0 = max(0, int(y1f - pad * bh))
            x1 = min(w_img, int(x2f + pad * bw))
            y1 = min(h_img, int(y2f + pad * bh))
            if x1 <= x0 or y1 <= y0:
                continue

            crop_for_seg = prep.preprocess_image(original[y0:y1, x0:x1], denoise_method=denoise_method, enhance_method="none")
            local_mask, local_contour = segmentation_mask_and_contour(crop_for_seg)
            used_fallback_rect = local_contour is None
            if used_fallback_rect:
                local_contour = np.array([[[0, 0]], [[x1 - x0 - 1, 0]], [[x1 - x0 - 1, y1 - y0 - 1]], [[0, y1 - y0 - 1]]])
                local_mask = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
                local_aspect = None  # a synthetic rectangle's aspect isn't informative about the real object
                local_circularity = None
            else:
                _local_solidity, local_aspect, local_circularity = contour_shape_metrics(local_contour)

            local_roughness = compute_texture_roughness(crop_for_seg, local_mask)
            _dbg(f"main loop: shape metrics aspect="
                 f"{'n/a' if local_aspect is None else f'{local_aspect:.3f}'} "
                 f"circularity={'n/a' if local_circularity is None else f'{local_circularity:.3f}'} "
                 f"roughness={local_roughness:.1f}")

            contour = local_contour + [x0, y0]  # shift into full-image coordinates
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            if erode_pixels > 0:
                kernel = np.ones((erode_pixels, erode_pixels), np.uint8)
                mask = cv2.erode(mask, kernel)

            bx, by, bw2, bh2 = cv2.boundingRect(contour)
            det = DetectionResult(
                found=True, bbox=(bx, by, bw2, bh2), contour=contour, mask=mask,
                area_px=float(cv2.contourArea(contour)),
                perimeter_px=float(cv2.arcLength(contour, closed=True)),
            )

            raw_isolated_crop = crop_object(original, det, isolate=True)

            yolo_species = yolo_label.capitalize()  # "apple" -> "Apple", matches train_fruit_type.py's convention
            type_label, type_conf, type_probs = classify_fruit_type_cnn(raw_isolated_crop)
            if FRUIT_DEBUG and type_probs:
                probs_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(type_probs.items(), key=lambda kv: -kv[1]))
                _dbg(f"main loop: type-CNN full breakdown: {probs_str}")
            fruit_type, fruit_type_confidence = yolo_species, yolo_conf
            type_source = "YOLO"

            if type_label == "Strawberry" and type_conf >= STRAWBERRY_OVERRIDE_MIN_CONF:
                fruit_type, fruit_type_confidence = type_label, type_conf
                type_source = "CNN override"
                _dbg(f"main loop: OVERRIDE {yolo_species}(conf={yolo_conf:.3f}) -> "
                     f"Strawberry(conf={type_conf:.3f})")
            elif type_label == "Mango" and type_conf >= MANGO_OVERRIDE_MIN_CONF_DOUBLE:
                aspect_ok = local_aspect is not None and local_aspect >= MANGO_OVERRIDE_MIN_ASPECT
                roughness_ok = local_roughness <= MANGO_OVERRIDE_MAX_ROUGHNESS
                both_ok = aspect_ok and roughness_ok
                required_conf = MANGO_OVERRIDE_MIN_CONF_DOUBLE if both_ok else MANGO_OVERRIDE_MIN_CONF
                if type_conf >= required_conf and (aspect_ok or roughness_ok):
                    fruit_type, fruit_type_confidence = type_label, type_conf
                    type_source = "CNN override"
                    passed_via = "+".join(g for g, ok in (("aspect", aspect_ok), ("roughness", roughness_ok)) if ok)
                    _dbg(f"main loop: OVERRIDE {yolo_species}(conf={yolo_conf:.3f}) -> "
                         f"Mango(conf={type_conf:.3f}, aspect="
                         f"{'n/a' if local_aspect is None else f'{local_aspect:.3f}'}, "
                         f"roughness={local_roughness:.1f}, passed_via={passed_via}, "
                         f"required_conf={required_conf})")
                else:
                    _dbg(f"main loop: Mango candidate REJECTED (shape+texture+conf gate) {yolo_species}(conf={yolo_conf:.3f}) "
                         f"type-CNN=Mango(conf={type_conf:.3f}, needs >= {required_conf}) aspect="
                         f"{'n/a' if local_aspect is None else f'{local_aspect:.3f}'} "
                         f"(needs >= {MANGO_OVERRIDE_MIN_ASPECT}) roughness={local_roughness:.1f} "
                         f"(needs <= {MANGO_OVERRIDE_MAX_ROUGHNESS})")

            _already_logged_disagreement = type_label == "Mango" and type_conf >= MANGO_OVERRIDE_MIN_CONF_DOUBLE
            if (FRUIT_DEBUG and type_label is not None and type_label != fruit_type
                    and fruit_type == yolo_species and not _already_logged_disagreement):
                _dbg(f"main loop: FINAL label={fruit_type} (YOLO conf={yolo_conf:.3f}) | "
                     f"type-CNN says {type_label} (conf={type_conf:.3f}) -- DISAGREES, not overridden")

            classification = ClassificationResult(fruit_type=fruit_type, fruit_type_confidence=fruit_type_confidence,
                                                   type_source=type_source)
            color_label, color_conf, color_probs = classify_quality_color_knn(raw_isolated_crop, fruit_type)
            if color_label is not None:
                classification.label = color_label
                classification.confidence = color_conf
            else:
                classification.error = (
                    f"No colour-feature model found for fruit type '{fruit_type}' "
                    f"(train one with train_color_knn.py, saved as {COLOR_KNN_MODELS_DIR}/{fruit_type}.joblib)"
                )

            if FRUIT_DEBUG and color_label is not None:
                _dbg(f"main loop: colour-feature KNN says {color_label} (conf={color_conf:.3f})")

            denoised_only = prep.denoise(original, method=denoise_method)
            _apply_quality_safety_rules(
                classification, color_probs, denoised_only, det.mask
            )

            width_px, height_px = float(bw2), float(bh2)
            area_px = det.area_px
            width_cm = height_cm = area_cm2 = None
            if calibration.confidence != "uncalibrated":
                width_cm = calibration.px_to_cm(width_px)
                height_cm = calibration.px_to_cm(height_px)
                area_cm2 = calibration.px_area_to_cm2(area_px)

            objects.append({
                "index": i,
                "detection": det,
                "bbox": det.bbox,
                "area_px": area_px,
                "width_px": width_px,
                "height_px": height_px,
                "area_cm2": area_cm2,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "classification": classification,
                "fruit_type": classification.fruit_type,
                "fruit_type_confidence": classification.fruit_type_confidence,
                "type_source": classification.type_source,
                "label": classification.label,
                "confidence": classification.confidence,
                "defect_fraction": classification.defect_fraction,
                "defect_override": classification.defect_override,
                "crop": crop_object(original, det, isolate=False),
                "crop_isolated": crop_object(original, det, isolate=True),
            })

            fruit_key = classification.fruit_type or "Unknown"
            quality_key = classification.label or "Unclassified"
            summary[fruit_key][quality_key] += 1
            i += 1

    fallback_type_model, _fallback_type_classes = _load_cnn_type_model()
    if fallback_type_model is not None:
        for blob in segment_all_objects(original):
            bx, by, bw3, bh3 = blob["bbox"]
            cx, cy = bx + bw3 // 2, by + bh3 // 2

            already_claimed = any(
                obj["detection"].mask is not None
                and 0 <= cy < obj["detection"].mask.shape[0]
                and 0 <= cx < obj["detection"].mask.shape[1]
                and obj["detection"].mask[cy, cx] > 0
                for obj in objects
            )
            if already_claimed:
                _dbg(f"fallback: SKIP (already_claimed by YOLO) bbox={blob['bbox']}")
                continue

            det = DetectionResult(
                found=True, bbox=blob["bbox"], contour=blob["contour"], mask=blob["mask"],
                area_px=blob["area_px"],
                perimeter_px=float(cv2.arcLength(blob["contour"], closed=True)),
            )

            raw_isolated_crop = crop_object(original, det, isolate=True)
            type_label, type_conf, type_probs = classify_fruit_type_cnn(raw_isolated_crop)
            if FRUIT_DEBUG and type_probs:
                probs_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(type_probs.items(), key=lambda kv: -kv[1]))
                _dbg(f"fallback: type-CNN full breakdown: {probs_str}")

            _blob_solidity, blob_aspect, blob_circularity = contour_shape_metrics(blob["contour"])
            blob_roughness = compute_texture_roughness(prep.denoise(original, method=denoise_method), blob["mask"])
            _dbg(f"fallback: bbox={blob['bbox']} type-CNN says {type_label} (conf={type_conf:.3f}) "
                 f"aspect={blob_aspect:.3f} circularity={blob_circularity:.3f} roughness={blob_roughness:.1f}")
            if type_label is None:
                continue
            if type_label not in FALLBACK_ONLY_SPECIES:
                continue

            required_conf = STRAWBERRY_OVERRIDE_MIN_CONF if type_label == "Strawberry" else MANGO_OVERRIDE_MIN_CONF
            if type_conf < required_conf:
                _dbg(f"fallback: REJECT (confidence floor) bbox={blob['bbox']} "
                     f"type-CNN={type_label}(conf={type_conf:.3f}) needs >= {required_conf}")
                continue

            classification = ClassificationResult(fruit_type=type_label, fruit_type_confidence=type_conf,
                                                   type_source="CNN (fallback pass)")
            color_label, color_conf, color_probs = classify_quality_color_knn(raw_isolated_crop, type_label)
            if color_label is not None:
                classification.label = color_label
                classification.confidence = color_conf
            else:
                classification.error = (
                    f"No colour-feature model found for fruit type '{type_label}' "
                    f"(train one with train_color_knn.py, saved as {COLOR_KNN_MODELS_DIR}/{type_label}.joblib)"
                )

            if FRUIT_DEBUG and color_label is not None:
                _dbg(f"fallback: colour-feature KNN says {color_label} (conf={color_conf:.3f})")

            denoised_only = prep.denoise(original, method=denoise_method)
            _apply_quality_safety_rules(
                classification, color_probs, denoised_only, det.mask
            )

            width_px, height_px = float(bw3), float(bh3)
            area_px = det.area_px
            width_cm = height_cm = area_cm2 = None
            if calibration.confidence != "uncalibrated":
                width_cm = calibration.px_to_cm(width_px)
                height_cm = calibration.px_to_cm(height_px)
                area_cm2 = calibration.px_area_to_cm2(area_px)

            objects.append({
                "index": i,
                "detection": det,
                "bbox": det.bbox,
                "area_px": area_px,
                "width_px": width_px,
                "height_px": height_px,
                "area_cm2": area_cm2,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "classification": classification,
                "fruit_type": classification.fruit_type,
                "fruit_type_confidence": classification.fruit_type_confidence,
                "type_source": classification.type_source,
                "label": classification.label,
                "confidence": classification.confidence,
                "defect_fraction": classification.defect_fraction,
                "defect_override": classification.defect_override,
                "crop": crop_object(original, det, isolate=False),
                "crop_isolated": raw_isolated_crop,
            })

            fruit_key = classification.fruit_type or "Unknown"
            quality_key = classification.label or "Unclassified"
            summary[fruit_key][quality_key] += 1
            i += 1

    objects = _dedupe_overlapping_detections(objects)
    summary = defaultdict(lambda: defaultdict(int))
    for new_i, obj in enumerate(objects):
        obj["index"] = new_i
        fruit_key = obj["fruit_type"] or "Unknown"
        quality_key = obj["label"] or "Unclassified"
        summary[fruit_key][quality_key] += 1

    annotated = draw_detections(preprocessed, objects)

    return {
        "original": original,
        "preprocessed": preprocessed,
        "annotated": annotated,
        "objects": objects,
        "summary": {k: dict(v) for k, v in summary.items()},
        "count": len(objects),
        "calibration_method": calibration.method,
        "calibration_confidence": calibration.confidence,
    }
