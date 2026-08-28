
"""
Overall analysis orchestrator -- ties the independently-built modules
(colour, fruit type, defect, stem, morphological/texture)
together into one combined per-fruit result.

This file is NEW integration code written for the "overall" system. It does
not modify any of the copied module files in common/, colour/, fruit_type/,
defect/, or stem/ -- it only imports and calls their existing public
functions.
"""

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

FRUIT_DEBUG = os.environ.get("FRUIT_DEBUG", "0") not in ("0", "", "false", "False")


def _dbg(*args):
    if FRUIT_DEBUG:
        print("[FRUIT_DEBUG]", *args)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("common", "colour", "fruit_type", "defect", "morphological"):
    _p = os.path.join(BASE_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocessing as prep                                   # common/
from segmentation import (                                     # common/
    segmentation_mask_and_contour,
    segment_all_objects,
    contour_shape_metrics,
    compute_texture_roughness,
)
from colour import train_color_knn as color_knn_module
from fruit_type import train_fruit_type as fruit_type_module
from defect.defect_detection import detect_defect
from defect.ripeness_detection import classify_ripeness                                      # defect/ (defect's own ripeness rule)
from stem.detector import StemDetector                         # stem/ (package)

try:
    # V12 (fruit_v12_hybrid_final_agreement_guard.py, wrapped by
    # morph_v12_bridge.py) replaces the old v10/v11 morph_texture_module.py
    # entirely, matching ASS's current app.py (which no longer imports
    # morph_texture_module at all). Aliased to the old name so every
    # downstream reference below keeps working unchanged.
    from morphological import morph_v12_bridge as morph_texture_module
    _MORPH_IMPORT_ERROR = None
except Exception as _morph_import_exc:                          # pragma: no cover
    morph_texture_module = None
    _MORPH_IMPORT_ERROR = str(_morph_import_exc)

# ---------------------------------------------------------------
# Species detection: ported from ASS's CURRENT app.py (run_exact_latest_
# defect_pipeline / choose_cnn_override), which has evolved well past
# colorDetection.py. Key change: the primary detector is now a YOLO model
# fine-tuned on the project's own photos (fruit_yolo_v4, apple/banana/orange
# only) instead of generic COCO yolov8n.pt -- it localises touching/occluded
# fruit far better, which is why ASS separates 3 touching strawberries into
# 3 boxes while yolov8n.pt + classical segmentation could not.
# ---------------------------------------------------------------
YOLO_FRUIT_CLASS_NAMES = {"apple", "banana", "orange", "mango", "strawberry"}
FRUIT_YOLO_V4_WEIGHTS = os.path.join(BASE_DIR, "defect", "weights", "fruit_yolo_v4_best.pt")
YOLO_CONF = 0.25
YOLO_IOU = 0.45
# colorDetection.py's analyze_image() resizes every input to this fixed size
# BEFORE running YOLO or segmentation -- every pixel-based constant in this
# file (10px/6px erosion, roughness/aspect thresholds, segment_all_objects'
# own area/peak-window fractions) was tuned against images at this size.
DEFAULT_IMAGE_SIZE = (512, 512)

# Applied AFTER choose_cnn_override, only if species is still Apple/Banana
# (i.e. the CNN did not override it to Mango/Strawberry). No extra floor for
# Orange -- a badly rotten orange may legitimately score lower confidence.
APPLE_MIN_CONF = 0.65
BANANA_MIN_CONF = 0.55

# choose_cnn_override: four independent Strawberry acceptance rules (normal /
# strong-red / green-unripe / damaged) plus one Mango rule, each using CNN
# confidence + red-pixel-fraction + shape (aspect, circularity) as a safety
# gate -- because this YOLO has no Mango/Strawberry class at all, so it can
# only ever guess Apple/Banana/Orange on those, sometimes confidently wrong.
STRAWBERRY_MIN_CONF = 0.95
STRAWBERRY_MAX_CIRCULARITY = 0.82
STRAWBERRY_MIN_RED_RATIO = 0.70

STRAWBERRY_STRONG_RED_RATIO = 0.80
STRAWBERRY_STRONG_RED_MIN_CONF = 0.90

STRAWBERRY_GREEN_MIN_CONF = 0.98
STRAWBERRY_GREEN_MAX_RED_RATIO = 0.15
STRAWBERRY_GREEN_MAX_CIRCULARITY = 0.65
STRAWBERRY_GREEN_MIN_ASPECT = 1.08

STRAWBERRY_DAMAGED_MIN_CONF = 0.97
STRAWBERRY_DAMAGED_MIN_RED_RATIO = 0.15
STRAWBERRY_DAMAGED_MAX_RED_RATIO = 0.70
STRAWBERRY_DAMAGED_MAX_CIRCULARITY = 0.72
STRAWBERRY_DAMAGED_MIN_ASPECT = 1.02

MANGO_MIN_CONF = 0.75  # lowered from 0.85 -- a real unripe/green mango scored 0.797 (aspect/circularity both passed) and was wrongly rejected
MANGO_MIN_ASPECT = 1.18
MANGO_MAX_CIRCULARITY = 0.90

# Mango rescue thresholds.
# The current YOLO weights were trained only on Apple/Banana/Orange, so a
# Mango can receive a very confident Banana/Apple YOLO label. These thresholds
# allow the five-class CNN to correct ONLY that specific non-native case,
# without weakening the Orange protection below.
MANGO_RAW_RESCUE_MIN_CONF = 0.97
MANGO_RAW_RESCUE_MIN_ASPECT = 1.12
MANGO_RAW_RESCUE_MAX_CIRCULARITY = 0.92
MANGO_TWO_VIEW_MIN_CONF = 0.80
MANGO_MORPH_SUPPORT_MIN_CONF = 0.70

# One-vote Mango rescue for the exact case where YOLO (which cannot output
# Mango) says Apple/Banana and the five-class CNN+rules says Mango, while the
# morphology module has no usable vote. The CNN+rules Mango candidate has
# already passed Mango confidence + shape/circularity safety gates, so allow it
# to win only when the two confidences are close. This prevents a 90% Banana
# guess from beating an 84% Mango solely because there are only two voters.
MANGO_SINGLE_RULE_MIN_CONF = 0.82
MANGO_SINGLE_RULE_MAX_YOLO_CONF = 0.92
MANGO_SINGLE_RULE_MAX_YOLO_GAP = 0.10

# Ripe Mango appearance rescue:
# Some red/yellow ripe mangoes are confidently called Apple by both the
# Apple/Banana/Orange-only YOLO and even the five-class CNN. A very specific
# shape + vertical colour-gradient pattern can rescue those without weakening
# Orange protection or ordinary round apples.
RIPE_MANGO_MIN_ASPECT = 1.28
RIPE_MANGO_MAX_CIRCULARITY = 0.88
RIPE_MANGO_MIN_PATTERN_SCORE = 0.88

# Extra rescue for the specific failure mode where BOTH YOLO and the CNN/rules
# confidently call a ripe red/yellow Mango Apple/Banana, while Morph V12
# independently identifies Mango. The appearance gate is evaluated directly
# on the raw rectangular crop using its own bbox aspect ratio, so it is not
# weakened by a poor segmentation contour. Orange is deliberately excluded.
RIPE_MANGO_MORPH_RESCUE_MIN_CONF = 0.80
RIPE_MANGO_MORPH_RESCUE_MIN_PATTERN_SCORE = 0.88

# NOT part of the ASS port -- added per explicit instruction. choose_cnn_override
# only ever corrects YOLO into Mango/Strawberry (classes it was never trained
# on); it has no rule for YOLO confusing two classes it DOES know (e.g. a real
# green Apple boxed as Orange). This lets the CNN win in THAT situation too,
# but only when it is both very confident on its own AND clearly more
# confident than YOLO -- a real photo scored CNN=Apple(0.99) vs YOLO=Orange
# (0.74), gap=0.25; these numbers are a starting point from that one photo.
KNOWN_CLASS_OVERRIDE_MIN_CNN_CONF = 0.90
KNOWN_CLASS_OVERRIDE_MIN_GAP = 0.15

# Duplicate/fragment YOLO box removal: drop a new box if its centre falls
# inside an already-accepted box AND its area is under this fraction of it.
DUPLICATE_FRAGMENT_MAX_AREA_RATIO = 0.35
# Drop boxes smaller than this fraction of the whole image as noise.
MIN_BOX_AREA_RATIO = 0.002

# Segmentation fallback can recover ANY supported fruit if whole-image YOLO
# misses it. Native YOLO species use a stricter CNN/morph confidence gate below.
FALLBACK_ONLY_SPECIES = {"Apple", "Banana", "Orange", "Mango", "Strawberry"}
FALLBACK_NATIVE_MIN_CNN_CONF = 0.80
FALLBACK_NATIVE_MIN_MORPH_CONF = 0.60

# Species that can participate in the independent 3-way species vote.
# YOLO itself only knows Apple/Banana/Orange, while the CNN/rule candidate
# and Morph V12 may also identify Mango/Strawberry.
VALID_SPECIES = {"Apple", "Banana", "Orange", "Mango", "Strawberry"}
YOLO_NATIVE_SPECIES = {"Apple", "Banana", "Orange"}

# High-confidence native YOLO protection.
#
# fruit_yolo_v4 was specifically trained for Apple/Banana/Orange. When it is
# extremely confident about one of those native classes, a Mango/Strawberry
# majority is allowed to overturn it only when BOTH alternative voters are
# also extremely confident. This prevents a green/unripe Orange from becoming
# Mango simply because CNN + morphology both react to its green colour/shape.
NATIVE_YOLO_LOCK_CONF = 0.95
NON_NATIVE_OVERRIDE_MIN_CONF = 0.95

DEFECT_SUPPORTED_SPECIES = {"Apple", "Banana", "Orange", "Mango", "Strawberry"}  # defect_detection.py's dispatcher now covers all 5

# ---------------------------------------------------------------
# Final quality fusion: two independent ripeness detectors are run on
# every fruit --
#   1. The user's own colour-KNN model (colour/train_color_knn.py)
#   2. The defect module's own rule-based ripeness classifier
#      (defect/ripeness_detection.py), which already folds each species'
#      own defect-percentage thresholds into its verdict.
# Whichever one reports the higher confidence for THIS specific photo wins.
# defect/ripeness_detection.py has no native probability, so its confidence
# is derived from numbers it already computes: for Ripe/Unripe, the
# dominant colour-percentage it based its own decision on; for Overripe,
# the defect percentage itself (normalised so 20% defect == full confidence,
# a starting point to tune against real photos, not a measured constant).
# ---------------------------------------------------------------
_RIPENESS_TO_QUALITY = {"Ripe": "Fresh", "Unripe": "Unripe", "Overripe": "Rotten"}
RIPENESS_RULE_SPECIES = {"Apple", "Banana", "Orange", "Strawberry", "Mango"}  # ripeness_detection.py now covers Mango too (classify_mango)
OVERRIPE_FULL_CONFIDENCE_DEFECT_PCT = 20.0

# Agreed rule: colour alone is NOT allowed to call something Rotten -- the
# defect side must confirm it (either its ripeness rule says Overripe, or
# raw defect percentage clears this floor, which covers Mango since it has
# no ripeness rule). This is a starting point to tune against real photos.
DEFECT_CONFIRMS_DEFECT_PCT = 3.0

# When colour KNN and the defect ripeness rule are compared head-to-head and
# BOTH end up under this confidence, neither number is meaningful (e.g. "1%
# vs 0%") -- flag the result as "Uncertain" instead of picking whichever is
# barely higher.
LOW_CONFIDENCE_THRESHOLD = 0.10

STEM_EXPECTED_SPECIES = {"Mango", "Apple", "Strawberry"}  # stem is only meaningful evidence for these

_yolo_model_cache = {}
_type_model_cache = {}
_color_model_cache = {}
_stem_detector_cache = {}


def _load_yolo(weights=FRUIT_YOLO_V4_WEIGHTS):
    if weights not in _yolo_model_cache:
        from ultralytics import YOLO
        _yolo_model_cache[weights] = YOLO(weights)
    return _yolo_model_cache[weights]


def calculate_red_ratio(roi, mask=None):
    """
    Estimate how much of the fruit surface is strawberry-red. Used only as a
    safety gate so a green/orange citrus fruit can't become Strawberry just
    because the CNN is confident. Ported from ASS's app.py, unchanged.
    """
    if roi is None or roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 90, 45]), np.array([8, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 90, 45]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_low, red_high)
    if mask is not None and mask.shape[:2] == roi.shape[:2] and cv2.countNonZero(mask) > 0:
        red_mask = cv2.bitwise_and(red_mask, mask)
        total = cv2.countNonZero(mask)
    else:
        total = roi.shape[0] * roi.shape[1]
    if total <= 0:
        return 0.0
    return cv2.countNonZero(red_mask) / total



def calculate_ripe_mango_pattern(roi, mask, aspect, circularity):
    """
    Detect the very specific appearance of a common ripe mango that transitions
    from red in the upper half to yellow/orange in the lower half.

    This is deliberately narrow. It is NOT a generic "red + yellow = mango"
    rule. It also requires an elongated mango-like shape and a strong vertical
    colour gradient, so ordinary round red/yellow apples remain protected.

    Returns:
        (matched: bool, score: float, metrics: dict)
    """
    metrics = {
        "red_ratio": 0.0,
        "yellow_ratio": 0.0,
        "top_red_ratio": 0.0,
        "bottom_red_ratio": 0.0,
        "top_yellow_ratio": 0.0,
        "bottom_yellow_ratio": 0.0,
        "red_gradient": 0.0,
        "yellow_gradient": 0.0,
    }

    if roi is None or roi.size == 0:
        return False, 0.0, metrics

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Use the pipeline's own segmentation mask when available. If it is not
    # usable, fall back only to chromatic pixels; white/black background then
    # contributes almost nothing.
    if (
        isinstance(mask, np.ndarray)
        and mask.shape[:2] == roi.shape[:2]
        and cv2.countNonZero(mask) > 0
    ):
        base = mask > 0
    else:
        base = np.ones(roi.shape[:2], dtype=bool)

    chromatic = (
        base
        & (sat >= 45)
        & (val >= 40)
    )

    ys, xs = np.where(chromatic)
    if len(xs) < 250:
        return False, 0.0, metrics

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    if y1 <= y0 or x1 <= x0:
        return False, 0.0, metrics

    yy = np.indices(chromatic.shape)[0]
    top_limit = y0 + int((y1 - y0) * 0.45)
    bottom_start = y0 + int((y1 - y0) * 0.55)

    top = chromatic & (yy <= top_limit)
    bottom = chromatic & (yy >= bottom_start)

    if np.count_nonzero(top) < 80 or np.count_nonzero(bottom) < 80:
        return False, 0.0, metrics

    red = (
        chromatic
        & (((hue <= 8) | (hue >= 170)))
        & (sat >= 75)
        & (val >= 45)
    )

    yellow_orange = (
        chromatic
        & (hue >= 8)
        & (hue <= 38)
        & (sat >= 55)
        & (val >= 55)
    )

    def _ratio(candidate, region):
        denom = int(np.count_nonzero(region))
        if denom <= 0:
            return 0.0
        return float(np.count_nonzero(candidate & region) / denom)

    metrics["red_ratio"] = _ratio(red, chromatic)
    metrics["yellow_ratio"] = _ratio(yellow_orange, chromatic)
    metrics["top_red_ratio"] = _ratio(red, top)
    metrics["bottom_red_ratio"] = _ratio(red, bottom)
    metrics["top_yellow_ratio"] = _ratio(yellow_orange, top)
    metrics["bottom_yellow_ratio"] = _ratio(yellow_orange, bottom)
    metrics["red_gradient"] = (
        metrics["top_red_ratio"] - metrics["bottom_red_ratio"]
    )
    metrics["yellow_gradient"] = (
        metrics["bottom_yellow_ratio"] - metrics["top_yellow_ratio"]
    )

    shape_ok = (
        aspect >= RIPE_MANGO_MIN_ASPECT
        and (
            circularity is None
            or circularity <= RIPE_MANGO_MAX_CIRCULARITY
        )
    )

    colour_ok = (
        0.15 <= metrics["red_ratio"] <= 0.75
        and metrics["yellow_ratio"] >= 0.20
        and metrics["top_red_ratio"] >= 0.55
        and metrics["bottom_yellow_ratio"] >= 0.45
        and metrics["red_gradient"] >= 0.25
        and metrics["yellow_gradient"] >= 0.25
    )

    if not (shape_ok and colour_ok):
        return False, 0.0, metrics

    def _clamp01(v):
        return max(0.0, min(1.0, float(v)))

    evidence = [
        _clamp01((aspect - 1.25) / 0.25),
        _clamp01((metrics["top_red_ratio"] - 0.55) / 0.35),
        _clamp01((metrics["bottom_yellow_ratio"] - 0.45) / 0.40),
        _clamp01((metrics["red_gradient"] - 0.25) / 0.45),
        _clamp01((metrics["yellow_gradient"] - 0.25) / 0.45),
        _clamp01(
            (metrics["red_ratio"] + metrics["yellow_ratio"]) / 0.75
        ),
    ]

    evidence_score = float(sum(evidence) / len(evidence))

    # This is a deterministic rule score, not a neural-network probability.
    # Keep it below 0.96 so the UI does not imply impossible certainty.
    score = min(
        0.95,
        0.80 + 0.16 * evidence_score
    )

    return True, float(score), metrics


def get_shape_values(contour, roi_width, roi_height):
    """Orientation-independent aspect ratio and circularity. Ported from ASS's app.py, unchanged."""
    aspect = max(roi_width / max(1, roi_height), roi_height / max(1, roi_width))
    circularity = None
    if contour is not None:
        try:
            _sol, contour_aspect, contour_circularity = contour_shape_metrics(contour)
            if contour_aspect is not None:
                contour_aspect = float(contour_aspect)
                aspect = max(contour_aspect, 1.0 / max(contour_aspect, 1e-6))
            circularity = contour_circularity
        except Exception:
            pass
    return aspect, circularity


def choose_cnn_override(yolo_confidence, cnn_type, cnn_conf, aspect, circularity, red_ratio):
    """
    Safely allow Mango/Strawberry to replace a weak YOLO guess.

    fruit_yolo_v4 was trained only on Apple/Banana/Orange, so a real Mango
    or Strawberry can score high confidence as one of those three -- YOLO
    confidence alone must not be trusted to block the CNN here. Instead each
    species has its own confidence + colour + shape safety gate. Ported
    verbatim from ASS's app.py (choose_cnn_override), unchanged.
    """
    strawberry_normal_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_MIN_CONF
        and red_ratio >= STRAWBERRY_MIN_RED_RATIO
        and (circularity is None or circularity < 0.78)
    )
    strawberry_strong_red_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_STRONG_RED_MIN_CONF
        and red_ratio >= STRAWBERRY_STRONG_RED_RATIO
        and (circularity is None or circularity < 0.78)
    )
    strawberry_green_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_GREEN_MIN_CONF
        and red_ratio <= STRAWBERRY_GREEN_MAX_RED_RATIO
        and aspect >= STRAWBERRY_GREEN_MIN_ASPECT
        and circularity is not None and circularity < STRAWBERRY_GREEN_MAX_CIRCULARITY
    )
    strawberry_damaged_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_DAMAGED_MIN_CONF
        and STRAWBERRY_DAMAGED_MIN_RED_RATIO <= red_ratio < STRAWBERRY_DAMAGED_MAX_RED_RATIO
        and aspect >= STRAWBERRY_DAMAGED_MIN_ASPECT
        and circularity is not None and circularity < STRAWBERRY_DAMAGED_MAX_CIRCULARITY
    )
    if strawberry_normal_ok or strawberry_strong_red_ok or strawberry_green_ok or strawberry_damaged_ok:
        return "strawberry"

    if (
        cnn_type == "Mango" and cnn_conf >= MANGO_MIN_CONF
        and aspect >= MANGO_MIN_ASPECT
        and (circularity is None or circularity < MANGO_MAX_CIRCULARITY)
    ):
        return "mango"

    return None


def _load_type_model():
    path = os.path.join(BASE_DIR, "fruit_type", "cnn_type_models", "fruit_type.pt")
    if path not in _type_model_cache:
        _type_model_cache[path] = fruit_type_module.load_type_model(path)
    return _type_model_cache[path]


def _load_color_model(fruit_type):
    path = os.path.join(BASE_DIR, "colour", "color_knn_models", f"{fruit_type}.joblib")
    if path not in _color_model_cache:
        if os.path.isfile(path):
            _color_model_cache[path] = color_knn_module.load_color_knn_model(path)
        else:
            _color_model_cache[path] = (None, None, None)
    return _color_model_cache[path]


def _load_stem_detector():
    if "default" not in _stem_detector_cache:
        weights = os.path.join(BASE_DIR, "stem", "weights", "best.pt")
        _stem_detector_cache["default"] = StemDetector(model_path=weights)
    return _stem_detector_cache["default"]


@dataclass
class FruitResult:
    bbox: tuple
    crop: np.ndarray
    species: Optional[str] = None
    species_confidence: float = 0.0
    species_source: Optional[str] = None
    yolo_species: Optional[str] = None
    yolo_confidence: float = 0.0
    cnn_species: Optional[str] = None
    cnn_confidence: float = 0.0
    raw_cnn_species: Optional[str] = None
    raw_cnn_confidence: float = 0.0
    cnn_rule_species: Optional[str] = None
    cnn_rule_confidence: float = 0.0
    cnn_rule_reason: str = ""
    # Legacy names kept because app.py/report.py may already display them.
    # They now represent the FULL CNN/rule vote (not YOLO+CNN fused together).
    own_species: Optional[str] = None
    own_confidence: float = 0.0
    morph_fruit_type: Optional[str] = None
    morph_fruit_type_confidence: float = 0.0
    colour_quality: Optional[str] = None
    colour_confidence: float = 0.0
    defect_percentage: Optional[float] = None
    defect_note: str = ""
    defect_marked_crop: Optional[np.ndarray] = None  # crop with the defect region(s) outlined, from defect_detection.py
    defect_ripeness: Optional[str] = None
    defect_ripeness_confidence: float = 0.0
    stem_detected: bool = False
    stem_confidence: float = 0.0
    stem_crop: Optional[np.ndarray] = None  # the crop stem detection ran on, with its own box/contour drawn on it
    morphological_note: str = "Not matched yet"
    morph_aspect_ratio: Optional[float] = None
    morph_circularity: Optional[float] = None
    morph_extent: Optional[float] = None
    morph_size_class: Optional[str] = None
    tex_contrast: Optional[float] = None
    tex_energy: Optional[float] = None
    tex_homogeneity: Optional[float] = None
    tex_entropy: Optional[float] = None
    tex_mean_intensity: Optional[float] = None
    tex_std_intensity: Optional[float] = None
    final_quality: Optional[str] = None
    final_quality_confidence: float = 0.0
    quality_note: str = ""


def _predict_type_candidate(image_bgr):
    """Run the current five-class fruit-type CNN and return label/conf/probs."""
    model, classes = _load_type_model()
    return fruit_type_module.predict_fruit_type_with_probs(model, classes, image_bgr)


def _contained_strawberry_evidence(bbox_xyxy, verified_strawberry_blobs):
    """Return strong Strawberry evidence contained in this YOLO box.

    Ported from test_defect.py's large-Orange/Strawberry-fragment check. The
    evidence is used only by the CNN/rule species candidate; it is not counted
    as an extra vote of its own.
    """
    if not bbox_xyxy or not verified_strawberry_blobs:
        return []

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    evidence_inside = []
    for evidence in verified_strawberry_blobs:
        ex1, ey1, ex2, ey2 = evidence["bbox"]
        evidence_area = max(1, (ex2 - ex1) * (ey2 - ey1))
        ix1, iy1 = max(x1, ex1), max(y1, ey1)
        ix2, iy2 = min(x2, ex2), min(y2, ey2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        contained_ratio = (iw * ih) / evidence_area
        if contained_ratio >= 0.80:
            evidence_inside.append(evidence)
    return evidence_inside


def _build_cnn_rule_candidate(
    crop_bgr,
    raw_crop_bgr,
    yolo_species,
    yolo_confidence,
    local_contour,
    raw_mask=None,
    bbox_xyxy=None,
    verified_strawberry_blobs=None,
):
    """Build ONE independent CNN/rule vote using the complete test_defect logic.

    Important: this is deliberately separate from the raw YOLO vote. The
    candidate starts from the five-class CNN, then applies the tested
    Strawberry/Mango correction rules (processed CNN + raw-ROI CNN + colour +
    shape + fragment evidence). That prevents YOLO from being counted twice in
    the final majority vote.
    """
    type_label, type_conf, type_probs = _predict_type_candidate(crop_bgr)

    raw_cnn_type, raw_cnn_conf, raw_probs = None, 0.0, {}
    if raw_crop_bgr is not None and raw_crop_bgr.size > 0:
        try:
            raw_cnn_type, raw_cnn_conf, raw_probs = _predict_type_candidate(raw_crop_bgr)
        except Exception as exc:  # keep processed CNN usable if raw check fails
            _dbg(f"_build_cnn_rule_candidate: raw-ROI CNN error: {exc}")

    measure_roi = raw_crop_bgr if raw_crop_bgr is not None and raw_crop_bgr.size > 0 else crop_bgr
    roi_h, roi_w = measure_roi.shape[:2]
    aspect, circularity = get_shape_values(local_contour, roi_w, roi_h)
    red_ratio = calculate_red_ratio(measure_roi, raw_mask)

    (
        ripe_mango_pattern,
        ripe_mango_pattern_score,
        ripe_mango_metrics,
    ) = calculate_ripe_mango_pattern(
        measure_roi,
        raw_mask,
        aspect,
        circularity,
    )

    # Native Orange consensus guard:
    # if YOLO + processed CNN + raw-ROI CNN all agree that this is Orange,
    # do not let shape-only/low-red Mango fallback rules overturn that
    # three-source Orange evidence. This is especially important for
    # green/unripe oranges, which naturally have very low red_ratio and can
    # become elongated/non-circular when a leaf enters the crop.
    native_orange_consensus = (
        yolo_species == "Orange"
        and type_label == "Orange"
        and raw_cnn_type == "Orange"
    )

    rule_species = None
    rule_confidence = 0.0
    rule_reason = "no_safe_cnn_candidate"

    # ------------------------------------------------------------------
    # Base independent CNN candidate.
    # Apple/Banana/Orange are native, direct five-class CNN outputs.
    # Mango/Strawberry must pass the same safety gates used in test_defect.py.
    # ------------------------------------------------------------------
    basic_override = choose_cnn_override(
        yolo_confidence, type_label, type_conf, aspect, circularity, red_ratio
    )

    if type_label in YOLO_NATIVE_SPECIES:
        rule_species = type_label
        rule_confidence = float(type_conf)
        rule_reason = "processed_cnn"
    elif basic_override == "strawberry":
        rule_species = "Strawberry"
        rule_confidence = float(type_conf)
        rule_reason = "processed_cnn_strawberry_safety_gate"
    elif basic_override == "mango":
        rule_species = "Mango"
        rule_confidence = float(type_conf)
        rule_reason = "processed_cnn_mango_safety_gate"

    # When both CNN views agree, keep the stronger confidence for this one
    # CNN/rule vote. This is still ONE vote, not two.
    if raw_cnn_type == rule_species and raw_cnn_conf > rule_confidence:
        rule_confidence = float(raw_cnn_conf)
        rule_reason += "+raw_roi_agreement"

    # Known-class disagreement: let a very strong raw-ROI CNN replace the
    # processed-CNN known class only when it is clearly stronger.
    if (
        raw_cnn_type in YOLO_NATIVE_SPECIES
        and rule_species in YOLO_NATIVE_SPECIES
        and raw_cnn_type != rule_species
        and raw_cnn_conf >= KNOWN_CLASS_OVERRIDE_MIN_CNN_CONF
        and (raw_cnn_conf - rule_confidence) >= KNOWN_CLASS_OVERRIDE_MIN_GAP
    ):
        rule_species = raw_cnn_type
        rule_confidence = float(raw_cnn_conf)
        rule_reason = "raw_roi_known_class_override"

    # ------------------------------------------------------------------
    # Complete special-case rules ported from test_defect.py.
    # These modify the ONE CNN/rule vote; they never create extra votes.
    # ------------------------------------------------------------------

    # GREEN STRAWBERRY / WEAK ORANGE SPECIAL CASE
    if (
        yolo_species == "Orange"
        and yolo_confidence < 0.70
        and type_label == "Strawberry"
        and type_conf >= 0.975
        and raw_cnn_type == "Strawberry"
        and raw_cnn_conf >= 0.995
        and red_ratio <= 0.10
        and aspect >= 1.15
        and circularity is not None
        and circularity < 0.76
    ):
        rule_species = "Strawberry"
        rule_confidence = float(max(type_conf, raw_cnn_conf))
        rule_reason = "green_strawberry_weak_orange"

    # DAMAGED STRAWBERRY / APPLE OVERRIDE
    if (
        yolo_species == "Apple"
        and type_label == "Strawberry"
        and type_conf >= 0.90
        and raw_cnn_type == "Strawberry"
        and raw_cnn_conf >= 0.99
        and 0.15 <= red_ratio < 0.70
        and aspect >= 1.15
        and circularity is not None
        and circularity < 0.72
    ):
        rule_species = "Strawberry"
        rule_confidence = float(max(type_conf, raw_cnn_conf))
        rule_reason = "damaged_strawberry_apple_override"

    # RAW-ROI CNN MANGO RECHECK
    if (
        rule_species not in {"Mango", "Strawberry"}
        and yolo_species in {"Orange", "Banana"}
        and raw_cnn_type == "Mango"
        and raw_cnn_conf >= 0.95
        and type_label == "Mango"
        and type_conf >= 0.80
        and aspect >= 1.15
        and circularity is not None
        and circularity < 0.90
    ):
        rule_species = "Mango"
        rule_confidence = float(max(raw_cnn_conf, type_conf))
        rule_reason = "raw_roi_mango_recheck"

    # STRONG RAW-ROI MANGO RESCUE (Apple/Banana YOLO only)
    #
    # A badly damaged/rotten Mango may look very different after isolation,
    # so the processed CNN can be less reliable while the raw rectangular ROI
    # still gives a very strong Mango prediction. Because the current YOLO has
    # NO Mango class, a high-confidence Banana/Apple YOLO score is not evidence
    # that the fruit cannot be Mango.
    #
    # Keep this deliberately narrow:
    #   * only rescue YOLO Apple/Banana (NOT Orange -- preserves the existing
    #     green/unripe-Orange protection),
    #   * require an extremely strong raw Mango CNN score,
    #   * require Mango-like shape,
    #   * do not override a strongly contradictory processed CNN.
    if (
        rule_species != "Strawberry"
        and yolo_species in {"Apple", "Banana"}
        and raw_cnn_type == "Mango"
        and raw_cnn_conf >= MANGO_RAW_RESCUE_MIN_CONF
        and aspect >= MANGO_RAW_RESCUE_MIN_ASPECT
        and (circularity is None or circularity < MANGO_RAW_RESCUE_MAX_CIRCULARITY)
        and (
            type_label == "Mango"
            or type_conf < 0.90
        )
    ):
        rule_species = "Mango"
        rule_confidence = float(max(raw_cnn_conf, type_conf if type_label == "Mango" else 0.0))
        rule_reason = "raw_roi_strong_mango_rescue"

    # BANANA -> MANGO DISAGREEMENT FALLBACK
    if (
        yolo_species == "Banana"
        and yolo_confidence <= 0.85
        and type_label == "Orange"
        and type_conf <= 0.85
        and 0.10 <= red_ratio <= 0.35
        and aspect <= 1.25
        and circularity is not None
        and 0.70 <= circularity < 0.85
    ):
        rule_species = "Mango"
        rule_confidence = float(max(yolo_confidence, type_conf))
        rule_reason = "banana_mango_disagreement_fallback"

    # LOW-RED WEAK-ORANGE -> MANGO FALLBACK
    #
    # IMPORTANT:
    # A healthy GREEN/UNRIPE ORANGE also has very low red_ratio. The old
    # version changed an Orange to Mango even when BOTH CNN views themselves
    # said Orange. That is too aggressive.
    #
    # Mango fallback now requires actual Mango evidence from the RAW-ROI CNN.
    # If YOLO + processed CNN + raw CNN agree on Orange, Orange is preserved.
    if (
        yolo_species == "Orange"
        and type_label == "Orange"
        and not native_orange_consensus
        and yolo_confidence < 0.90
        and type_conf < 0.85
        and raw_cnn_type == "Mango"
        and raw_cnn_conf >= 0.75
        and red_ratio <= 0.08
        and circularity is not None
        and circularity < 0.77
    ):
        rule_species = "Mango"
        rule_confidence = float(max(yolo_confidence, type_conf, raw_cnn_conf))
        rule_reason = "low_red_weak_orange_mango_fallback_with_mango_evidence"

    # ORANGE -> MANGO SHAPE FALLBACK
    #
    # Shape alone is not allowed to turn an Orange into Mango anymore.
    # Leaves/occlusion can make a true orange crop look elongated and reduce
    # circularity. Require supporting Mango evidence from the raw-ROI CNN.
    if (
        basic_override is None
        and yolo_species == "Orange"
        and type_label == "Orange"
        and not native_orange_consensus
        and raw_cnn_type == "Mango"
        and raw_cnn_conf >= 0.75
        and circularity is not None
        and (
            (
                type_conf <= 0.93
                and red_ratio <= 0.15
                and aspect >= 1.23
                and circularity < 0.79
            )
            or (aspect >= 1.24 and circularity < 0.76)
        )
    ):
        rule_species = "Mango"
        rule_confidence = float(max(type_conf, raw_cnn_conf))
        rule_reason = "orange_mango_shape_fallback_with_mango_evidence"

    # LARGE/WEAK YOLO ORANGE -> STRAWBERRY USING VERIFIED SEGMENTED FRAGMENTS
    strawberry_evidence = _contained_strawberry_evidence(
        bbox_xyxy, verified_strawberry_blobs
    )
    if yolo_species == "Orange" and strawberry_evidence:
        x1, y1, x2, y2 = bbox_xyxy
        current_area = max(1, (x2 - x1) * (y2 - y1))
        strong_fragment_count = len(strawberry_evidence)
        large_fragment = any(
            max(1, (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]))
            >= current_area * 0.28
            for e in strawberry_evidence
        )
        if (
            (yolo_confidence < 0.55 and strong_fragment_count >= 1)
            or strong_fragment_count >= 2
            or large_fragment
        ):
            rule_species = "Strawberry"
            rule_confidence = float(max(e["confidence"] for e in strawberry_evidence))
            rule_reason = "contained_strawberry_fragment_evidence"

    # RIPE RED/YELLOW MANGO -> APPLE/BANANA RESCUE
    #
    # YOLO cannot output Mango at all, and some smooth red/yellow ripe mangoes
    # can also fool the five-class CNN into Apple. Only apply this rule to
    # Apple/Banana YOLO guesses, never Orange, and only when the mango-specific
    # elongated shape + top-red/bottom-yellow gradient is very strong.
    if (
        yolo_species in {"Apple", "Banana"}
        and rule_species not in {"Mango", "Strawberry"}
        and ripe_mango_pattern
        and ripe_mango_pattern_score >= RIPE_MANGO_MIN_PATTERN_SCORE
    ):
        rule_species = "Mango"
        rule_confidence = float(ripe_mango_pattern_score)
        rule_reason = "ripe_mango_colour_shape_pattern"

    _dbg(
        "_build_cnn_rule_candidate: "
        f"yolo={yolo_species}({yolo_confidence:.3f}) "
        f"cnn={type_label}({type_conf:.3f}) "
        f"raw_cnn={raw_cnn_type}({raw_cnn_conf:.3f}) "
        f"aspect={aspect:.3f} "
        f"circularity={'n/a' if circularity is None else f'{circularity:.3f}'} "
        f"red={red_ratio:.3f} ripe_mango_pattern={ripe_mango_pattern_score:.3f} "
        f"-> rule={rule_species}({rule_confidence:.3f}) "
        f"reason={rule_reason}"
    )

    return {
        "species": rule_species,
        "confidence": rule_confidence,
        "reason": rule_reason,
        "cnn_species": type_label,
        "cnn_confidence": float(type_conf),
        "cnn_probs": type_probs,
        "raw_cnn_species": raw_cnn_type,
        "raw_cnn_confidence": float(raw_cnn_conf),
        "raw_cnn_probs": raw_probs,
        "aspect": aspect,
        "circularity": circularity,
        "red_ratio": red_ratio,
        "ripe_mango_pattern": bool(ripe_mango_pattern),
        "ripe_mango_pattern_score": float(ripe_mango_pattern_score),
        "ripe_mango_metrics": ripe_mango_metrics,
    }


def _prescan_verified_strawberry_blobs(original_bgr, segmented_blobs):
    """Pre-scan segmented objects for strong Strawberry evidence.

    This is the same idea as test_defect.py's pre-scan. It is used only to
    strengthen the CNN/rule candidate when a mouldy Strawberry is swallowed by
    one large YOLO Orange box.
    """
    verified = []
    if not segmented_blobs:
        return verified

    h_img, w_img = original_bgr.shape[:2]
    for blob in segmented_blobs:
        bx, by, bw, bh = blob.get("bbox", (0, 0, 0, 0))
        x1, y1 = max(0, int(bx)), max(0, int(by))
        x2, y2 = min(w_img, int(bx + bw)), min(h_img, int(by + bh))
        if x2 <= x1 or y2 <= y1:
            continue

        raw_roi = original_bgr[y1:y2, x1:x2].copy()
        if raw_roi.size == 0:
            continue

        local_mask = None
        cnn_roi = raw_roi
        full_mask = blob.get("mask")
        if isinstance(full_mask, np.ndarray):
            candidate_mask = full_mask[y1:y2, x1:x2]
            if candidate_mask.shape[:2] == raw_roi.shape[:2]:
                local_mask = candidate_mask
                cnn_roi = cv2.bitwise_and(raw_roi, raw_roi, mask=local_mask)

        try:
            cnn_type, cnn_conf, _ = _predict_type_candidate(cnn_roi)
        except Exception as exc:
            _dbg(f"_prescan_verified_strawberry_blobs: CNN error: {exc}")
            continue

        aspect, circularity = get_shape_values(
            blob.get("contour"), raw_roi.shape[1], raw_roi.shape[0]
        )
        red_ratio = calculate_red_ratio(raw_roi, local_mask)
        override = choose_cnn_override(
            0.0, cnn_type, cnn_conf, aspect, circularity, red_ratio
        )
        if override == "strawberry":
            verified.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(cnn_conf),
                    "red_ratio": float(red_ratio),
                    "aspect": float(aspect),
                    "circularity": circularity,
                }
            )
    return verified


def _classify_species(
    crop_bgr,
    yolo_label,
    yolo_conf,
    local_contour,
    raw_crop_bgr=None,
    raw_mask=None,
    bbox_xyxy=None,
    verified_strawberry_blobs=None,
    morph_species=None,
    morph_confidence=0.0,
):
    """Choose fruit species by an independent majority vote.

    Vote 1: raw fruit_yolo_v4 result (Apple/Banana/Orange only)
    Vote 2: full five-class CNN/rule candidate, including BOTH CNN views and
            every Strawberry/Mango correction/fallback from test_defect.py
    Vote 3: teammate Morph V12 candidate

    The old implementation counted YOLO twice (raw YOLO + an 'own' candidate
    that started from YOLO). This version removes that double-counting.
    """
    if raw_crop_bgr is None:
        raw_crop_bgr = crop_bgr

    yolo_species = yolo_label.capitalize() if yolo_label else None
    yolo_confidence = float(yolo_conf or 0.0)
    if yolo_species not in YOLO_NATIVE_SPECIES:
        yolo_species = None
        yolo_confidence = 0.0

    rule = _build_cnn_rule_candidate(
        crop_bgr=crop_bgr,
        raw_crop_bgr=raw_crop_bgr,
        yolo_species=yolo_species,
        yolo_confidence=yolo_confidence,
        local_contour=local_contour,
        raw_mask=raw_mask,
        bbox_xyxy=bbox_xyxy,
        verified_strawberry_blobs=verified_strawberry_blobs,
    )
    rule_species = rule["species"]
    rule_confidence = float(rule["confidence"])

    if morph_species is not None:
        morph_species = str(morph_species).capitalize()
        if morph_species not in VALID_SPECIES:
            morph_species = None
            morph_confidence = 0.0

    candidates = []
    if yolo_species is not None:
        candidates.append(("yolo", yolo_species, yolo_confidence))
    if rule_species is not None:
        candidates.append(("cnn_rules", rule_species, rule_confidence))
    if morph_species is not None:
        candidates.append(("morph", morph_species, float(morph_confidence)))

    # --------------------------------------------------------------
    # RIPE MANGO: YOLO + CNN BOTH WRONG, MORPH + APPEARANCE RESCUE
    # --------------------------------------------------------------
    # Real failure example:
    #   YOLO        -> Apple 0.96
    #   CNN + Rules -> Apple 0.99
    #   Morph       -> Mango 0.82
    #
    # A normal 2-of-3 majority would force Apple, and the >=0.95 native-YOLO
    # lock would reinforce that result. But this YOLO has no Mango class, and
    # a smooth ripe Mango can be Apple-like to the CNN too. We therefore allow
    # one very narrow exception BEFORE the majority/native lock:
    #   1) YOLO and CNN/rules agree on Apple or Banana,
    #   2) Morph independently says Mango >=80%,
    #   3) the RAW crop itself has a strong elongated red-top/yellow-bottom
    #      ripe-Mango appearance.
    #
    # The appearance check uses the raw crop's own bbox aspect ratio rather
    # than the segmentation contour, because a poor contour can make the
    # measured shape too round and prevent an otherwise obvious Mango rescue.
    # Orange is intentionally excluded to preserve all existing green-orange
    # protection rules.
    ripe_mango_morph_rescue = False
    ripe_mango_morph_score = 0.0
    ripe_mango_morph_metrics = {}

    if (
        yolo_species in {"Apple", "Banana"}
        and rule_species == yolo_species
        and morph_species == "Mango"
        and float(morph_confidence) >= RIPE_MANGO_MORPH_RESCUE_MIN_CONF
        and raw_crop_bgr is not None
        and raw_crop_bgr.size > 0
    ):
        raw_h, raw_w = raw_crop_bgr.shape[:2]
        raw_bbox_aspect = max(
            raw_w / max(1, raw_h),
            raw_h / max(1, raw_w),
        )

        (
            ripe_mango_morph_rescue,
            ripe_mango_morph_score,
            ripe_mango_morph_metrics,
        ) = calculate_ripe_mango_pattern(
            raw_crop_bgr,
            raw_mask,
            raw_bbox_aspect,
            None,  # do not trust a possibly distorted contour circularity here
        )

        ripe_mango_morph_rescue = (
            ripe_mango_morph_rescue
            and ripe_mango_morph_score
                >= RIPE_MANGO_MORPH_RESCUE_MIN_PATTERN_SCORE
        )

        _dbg(
            "ripe_mango_morph_rescue: "
            f"yolo={yolo_species}({yolo_confidence:.3f}) "
            f"rule={rule_species}({rule_confidence:.3f}) "
            f"morph={morph_species}({float(morph_confidence):.3f}) "
            f"raw_bbox_aspect={raw_bbox_aspect:.3f} "
            f"pattern={ripe_mango_morph_score:.3f} "
            f"metrics={ripe_mango_morph_metrics} "
            f"accepted={ripe_mango_morph_rescue}"
        )

    if ripe_mango_morph_rescue:
        species = "Mango"
        confidence = max(
            float(morph_confidence),
            float(ripe_mango_morph_score),
        )
        source = "ripe_mango_morph_appearance_rescue"

    elif not candidates:
        species, confidence, source = None, 0.0, "no_species_candidate"
    else:
        # ----------------------------------------------------------
        # HIGH-CONFIDENCE NATIVE YOLO PROTECTION
        # ----------------------------------------------------------
        #
        # Example that motivated this guard:
        #   YOLO      -> Orange 0.98
        #   CNN rules -> Mango  0.99
        #   Morph     -> Mango  0.93
        #
        # A green/unripe Orange can look Mango-like to both CNN and
        # morphology. Because the project YOLO was specifically trained on
        # Apple/Banana/Orange, a >=95% native YOLO result is treated as strong
        # class-specific evidence. A non-native majority may still overturn
        # it, but only when BOTH alternative voters agree AND EACH is >=95%.
        native_lock_applied = False

        # ----------------------------------------------------------
        # MANGO RESCUE BEFORE NATIVE-YOLO LOCK
        # ----------------------------------------------------------
        # The current YOLO cannot output Mango at all. Therefore a high
        # Banana/Apple YOLO confidence must not automatically veto strong Mango
        # evidence from the five-class CNN. This rescue is intentionally NOT
        # applied to YOLO Orange because the existing dataset contains true
        # green/unripe oranges that can look Mango-like.
        #
        # Accept Mango when either:
        #   A) processed + raw CNN views BOTH say Mango with >=80%, or
        #   B) the raw-ROI Mango rescue above fired at >=97%, optionally
        #      strengthened by Morph V12 agreeing with Mango.
        mango_two_view_agreement = (
            yolo_species in {"Apple", "Banana"}
            and rule_species == "Mango"
            and rule["cnn_species"] == "Mango"
            and rule["cnn_confidence"] >= MANGO_TWO_VIEW_MIN_CONF
            and rule["raw_cnn_species"] == "Mango"
            and rule["raw_cnn_confidence"] >= MANGO_TWO_VIEW_MIN_CONF
        )

        mango_strong_raw_rescue = (
            yolo_species in {"Apple", "Banana"}
            and rule_species == "Mango"
            and rule["reason"] == "raw_roi_strong_mango_rescue"
            and rule["raw_cnn_species"] == "Mango"
            and rule["raw_cnn_confidence"] >= MANGO_RAW_RESCUE_MIN_CONF
        )

        mango_morph_support = (
            morph_species == "Mango"
            and float(morph_confidence) >= MANGO_MORPH_SUPPORT_MIN_CONF
        )

        mango_ripe_pattern_rescue = (
            yolo_species in {"Apple", "Banana"}
            and rule_species == "Mango"
            and rule["reason"] == "ripe_mango_colour_shape_pattern"
            and rule.get("ripe_mango_pattern_score", 0.0)
                >= RIPE_MANGO_MIN_PATTERN_SCORE
        )

        if (
            mango_two_view_agreement
            or mango_strong_raw_rescue
            or mango_ripe_pattern_rescue
        ):
            species = "Mango"
            confidence = max(
                rule_confidence,
                float(rule["cnn_confidence"]),
                float(rule["raw_cnn_confidence"]),
                float(morph_confidence) if mango_morph_support else 0.0,
            )
            if mango_two_view_agreement:
                source = "mango_rescue:two_cnn_views"
            elif mango_strong_raw_rescue:
                source = "mango_rescue:strong_raw_roi"
            else:
                source = "mango_rescue:ripe_colour_shape_pattern"
            if mango_morph_support:
                source += "+morph"
            native_lock_applied = True

        if (
            not native_lock_applied
            and yolo_species in YOLO_NATIVE_SPECIES
            and yolo_confidence >= NATIVE_YOLO_LOCK_CONF
        ):
            opposing = [
                (name, sp, conf)
                for name, sp, conf in candidates
                if name != "yolo" and sp != yolo_species
            ]

            # Only a same-species pair of extremely strong alternative votes
            # is allowed to beat the locked native YOLO result.
            strong_opposing_override = (
                len(opposing) >= 2
                and opposing[0][1] == opposing[1][1]
                and all(
                    conf >= NON_NATIVE_OVERRIDE_MIN_CONF
                    for _, _, conf in opposing[:2]
                )
            )

            if not strong_opposing_override:
                species = yolo_species
                confidence = yolo_confidence
                source = "high_conf_native_yolo_lock"
                native_lock_applied = True

        if not native_lock_applied:
            vote_counts = Counter(sp for _, sp, _ in candidates)
            top_species, top_votes = vote_counts.most_common(1)[0]

            if top_votes >= 2:
                agreeing = [
                    (s, sp, c)
                    for s, sp, c in candidates
                    if sp == top_species
                ]
                species = top_species
                confidence = max(c for _, _, c in agreeing)
                source = "majority(" + "+".join(
                    s for s, _, _ in agreeing
                ) + ")"
            else:
                # ------------------------------------------------------
                # CLOSE APPLE/BANANA vs MANGO DISAGREEMENT
                # ------------------------------------------------------
                # fruit_yolo_v4 has no Mango class, so in a two-voter tie
                # (e.g. YOLO Banana 0.90 vs CNN+rules Mango 0.84, Morph no
                # match) blindly selecting the numerically highest confidence
                # forces a real Mango to become Banana. CNN+rules can only
                # produce Mango after passing its Mango-specific confidence +
                # shape safety gates. Therefore let that Mango candidate win
                # when YOLO is Apple/Banana, YOLO is not extremely confident,
                # and the confidence gap is small. Orange is deliberately
                # excluded to preserve the green/unripe-orange safeguards.
                mango_close_disagreement = (
                    morph_species is None
                    and yolo_species in {"Apple", "Banana"}
                    and rule_species == "Mango"
                    and rule_confidence >= MANGO_SINGLE_RULE_MIN_CONF
                    and yolo_confidence <= MANGO_SINGLE_RULE_MAX_YOLO_CONF
                    and (yolo_confidence - rule_confidence) <= MANGO_SINGLE_RULE_MAX_YOLO_GAP
                )

                if mango_close_disagreement:
                    species = "Mango"
                    confidence = rule_confidence
                    source = "mango_rescue:close_cnn_rule_disagreement"
                else:
                    source_name, species, confidence = max(
                        candidates,
                        key=lambda c: c[2]
                    )
                    source = f"no_majority_highest_conf:{source_name}"

    _dbg(
        f"_classify_species: candidates=[{', '.join(f'{s}={sp}({c:.3f})' for s, sp, c in candidates)}] "
        f"-> winner={source} final={species}({confidence:.3f})"
    )

    return {
        "species": species,
        "species_confidence": float(confidence),
        "species_source": source,
        "yolo_species": yolo_species,
        "yolo_confidence": yolo_confidence,
        "cnn_species": rule["cnn_species"],
        "cnn_confidence": rule["cnn_confidence"],
        "raw_cnn_species": rule["raw_cnn_species"],
        "raw_cnn_confidence": rule["raw_cnn_confidence"],
        "cnn_rule_species": rule_species,
        "cnn_rule_confidence": rule_confidence,
        "cnn_rule_reason": rule["reason"],
        # legacy fields now mirror the independent CNN/rule vote
        "own_species": rule_species,
        "own_confidence": rule_confidence,
        "morph_species": morph_species,
        "morph_confidence": float(morph_confidence),
    }

def _defect_ripeness_confidence(ripeness, colour1, colour2, defect_pct):
    """
    Derive a 0-1 confidence for defect/ripeness_detection.py's rule-based
    verdict, using numbers it already computed (not invented):
    - Overripe verdicts are driven by defect percentage crossing a
      per-species threshold inside classify_ripeness itself, so defect
      percentage is the evidence -- normalised against a 20% reference.
    - Ripe/Unripe verdicts are driven by whichever colour percentage
      (colour1 or colour2) dominated the decision.
    """
    if ripeness == "Overripe":
        return min(1.0, (defect_pct or 0.0) / OVERRIPE_FULL_CONFIDENCE_DEFECT_PCT)
    if ripeness not in _RIPENESS_TO_QUALITY:
        return 0.0
    return min(1.0, max(colour1, colour2) / 100.0)


def _best_non_rotten(colour_probs, colour_label, colour_conf):
    """
    Fallback when colour alone says Rotten but the defect side found no
    corroborating defect: pick colour's own best NON-Rotten class instead of
    trusting the Rotten call. Mirrors the original apple-rotten-guard idea
    in colorDetection.py (_best_non_rotten_quality) using the probabilities
    the colour KNN already computed -- nothing invented.
    """
    candidates = {k: v for k, v in (colour_probs or {}).items() if k != "Rotten"}
    if not candidates:
        return colour_label, colour_conf
    label = max(candidates, key=candidates.get)
    total = sum(candidates.values())
    conf = candidates[label] / total if total > 1e-9 else 0.0
    return label, float(conf)


def _fuse_quality(species, colour_label, colour_conf, colour_probs, defect_label, defect_conf, defect_pct, stem_detected):
    """
    Run TWO independent ripeness detectors -- the user's own colour-KNN and
    the defect module's own rule-based ripeness classifier -- and trust
    whichever produced the higher confidence for this specific photo.

    Agreed rule: colour is NOT allowed to call something Rotten on its own.
    The defect side must confirm it (its ripeness rule says Overripe, or the
    raw defect percentage clears DEFECT_CONFIRMS_DEFECT_PCT -- the latter
    covers Mango, which has no ripeness rule). If colour says Rotten but
    defect finds nothing, colour's own best non-Rotten class is used instead.

    Stem presence/absence is only meaningful for Mango, Apple and
    Strawberry; for other species it is not used as quality evidence.
    Having a stem is a small positive signal, not a hard override.
    """
    notes = []

    # When the defect module HAS a ripeness verdict for this species (Apple/
    # Banana/Orange/Strawberry), that verdict is authoritative -- it already
    # applied its own per-species defect% threshold (e.g. strawberry needs
    # 8-15%, not a flat number) and concluded, so a generic floor here must
    # not override an explicit "Fresh"/"Unripe" call. The flat percentage
    # floor is only a fallback for species with no ripeness rule (Mango).
    if defect_label is not None:
        defect_confirms_defect = defect_label == "Rotten"
    else:
        defect_confirms_defect = defect_pct is not None and defect_pct >= DEFECT_CONFIRMS_DEFECT_PCT

    # Checked against the real 300-image evaluate_overall.py run: this veto
    # never once saved a genuinely-Fresh fruit in that run -- the few Fresh
    # images where colour cried Rotten all had colour_conf well under 0.9
    # (~0.66), so the confidence arbitration below already resolves those
    # correctly on its own. But the veto was unconditionally discarding
    # colour whenever colour was maximally confident (~1.0, an n_neighbors=1
    # artifact) and defect simply found no defect *region* on a truly rotten
    # Orange/Apple -- that alone accounted for most of the dataset's
    # Rotten-mistaken-for-Fresh errors (Orange 15/20, Apple 9/20). Gating the
    # veto on colour_conf keeps it protecting genuinely uncertain colour
    # calls while letting confidence-based arbitration settle the rest.
    if colour_label == "Rotten" and not defect_confirms_defect and colour_conf < 0.9:
        fallback_label, fallback_conf = _best_non_rotten(colour_probs, colour_label, colour_conf)
        pct_str = f"{defect_pct:.1f}%" if defect_pct is not None else "n/a"
        notes.append(f"colour KNN said Rotten but defect found nothing (defect%={pct_str}) -- downgraded to {fallback_label}")
        colour_label, colour_conf = fallback_label, fallback_conf

    if colour_label is None and defect_label is None:
        final_label, final_conf = None, 0.0
    elif defect_label is None:
        final_label, final_conf = colour_label, colour_conf
    elif colour_label is None:
        final_label, final_conf = defect_label, defect_conf
        notes.append(f"defect ripeness rule only ({defect_conf * 100:.0f}%)")
    else:
        if defect_conf > colour_conf:
            winner_label, winner_conf, winner_name = defect_label, defect_conf, "defect ripeness rule"
            loser_conf, loser_name = colour_conf, "colour KNN"
        else:
            winner_label, winner_conf, winner_name = colour_label, colour_conf, "colour KNN"
            loser_conf, loser_name = defect_conf, "defect ripeness rule"

        # Both sides can end up with near-zero confidence at once -- e.g.
        # colour KNN's Rotten call got downgraded (leaving almost no
        # probability for any other class) while the defect ripeness
        # rule's own colour-percentage confidence also happened to be
        # tiny. Picking whichever is barely higher (1% vs 0%) is not a
        # real decision -- it's noise. Flag it instead of pretending
        # either number means something.
        if max(colour_conf, defect_conf) < LOW_CONFIDENCE_THRESHOLD:
            final_label, final_conf = "Uncertain", max(colour_conf, defect_conf)
            notes.append(
                f"LOW CONFIDENCE on both sides ({winner_name} {winner_conf * 100:.0f}% vs {loser_name} "
                f"{loser_conf * 100:.0f}%, neither clears {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%) -- "
                f"not enough evidence to trust either signal, flagged for manual review instead of "
                f"picking the barely-higher one"
            )
        else:
            final_label, final_conf = winner_label, winner_conf
            notes.append(f"{winner_name} won ({winner_conf * 100:.0f}% vs {loser_name} {loser_conf * 100:.0f}%)")

    if colour_label is not None and defect_label is not None and colour_label != defect_label:
        notes.append(f"disagreement: colour KNN={colour_label}, defect rule={defect_label}")

    if species in STEM_EXPECTED_SPECIES:
        notes.append("Stem present" if stem_detected else "Stem not detected")

    return final_label, final_conf, "; ".join(notes)


def analyse_fruit(crop_bgr, yolo_label, yolo_conf, mask=None, contour=None, roughness=None, raw_crop_bgr=None,
                   type_raw_crop_bgr=None, type_mask=None, bbox_xyxy=None, verified_strawberry_blobs=None,
                   morph_species=None, morph_confidence=0.0, morph_obj=None, morph_iou=0.0, morph_status=None):
    """Run every available module on one already-cropped fruit and return a
    combined FruitResult.

    crop_bgr is the ISOLATED crop (background blackened, cropped tight to the
    contour) -- used for species CNN and colour KNN, matching ASS's
    colorDetection.py (classify_fruit_type_cnn / classify_quality_color_knn
    both take raw_isolated_crop).

    raw_crop_bgr is the RAW, un-isolated rectangular crop (falls back to
    crop_bgr if not given) -- used for defect_detection/ripeness_detection,
    matching ASS's CURRENT app.py, which explicitly warns that a calibrated/
    black-background crop "can make healthy peel look defective" for those
    two modules specifically.

    morph_species/morph_confidence: this fruit's matched result (if any)
    from the teammate's independent morphological/texture module, already
    IoU-matched by the caller against that module's own whole-image
    detection pass -- fed into _classify_species as the third candidate.
    morph_obj/morph_iou are used purely to attach the geometry/texture
    numbers (aspect ratio, circularity, GLCM texture) for display; they do
    NOT affect final_quality.
    """
    if raw_crop_bgr is None:
        raw_crop_bgr = crop_bgr
    if type_raw_crop_bgr is None:
        type_raw_crop_bgr = raw_crop_bgr
    result = FruitResult(bbox=(0, 0, crop_bgr.shape[1], crop_bgr.shape[0]), crop=crop_bgr)

    species_out = _classify_species(
        crop_bgr, yolo_label, yolo_conf, contour,
        raw_crop_bgr=type_raw_crop_bgr,
        raw_mask=type_mask,
        bbox_xyxy=bbox_xyxy,
        verified_strawberry_blobs=verified_strawberry_blobs,
        morph_species=morph_species,
        morph_confidence=morph_confidence,
    )
    species = species_out["species"]
    result.species = species
    result.species_confidence = species_out["species_confidence"]
    result.species_source = species_out["species_source"]
    result.yolo_species = species_out["yolo_species"]
    result.yolo_confidence = species_out["yolo_confidence"]
    result.cnn_species = species_out["cnn_species"]
    result.cnn_confidence = species_out["cnn_confidence"]
    result.raw_cnn_species = species_out["raw_cnn_species"]
    result.raw_cnn_confidence = species_out["raw_cnn_confidence"]
    result.cnn_rule_species = species_out["cnn_rule_species"]
    result.cnn_rule_confidence = species_out["cnn_rule_confidence"]
    result.cnn_rule_reason = species_out["cnn_rule_reason"]
    result.own_species = species_out["own_species"]
    result.own_confidence = species_out["own_confidence"]
    result.morph_fruit_type = species_out["morph_species"]
    result.morph_fruit_type_confidence = species_out["morph_confidence"]

    if morph_status is not None:
        result.morphological_note = morph_status
    elif morph_obj is not None:
        result.morph_aspect_ratio = morph_obj.get("geo_aspect_ratio")
        result.morph_circularity = morph_obj.get("geo_circularity")
        result.morph_extent = morph_obj.get("geo_extent")
        result.morph_size_class = morph_obj.get("size_class")
        result.tex_contrast = morph_obj.get("tex_contrast")
        result.tex_energy = morph_obj.get("tex_energy")
        result.tex_homogeneity = morph_obj.get("tex_homogeneity")
        result.tex_entropy = morph_obj.get("tex_entropy")
        result.tex_mean_intensity = morph_obj.get("tex_mean_intensity")
        result.tex_std_intensity = morph_obj.get("tex_std_intensity")
        result.morphological_note = f"Matched morphological/texture detection (IoU={morph_iou:.2f})"
    else:
        result.morphological_note = "No matching morphological/texture detection for this box"

    scaler, knn_model, classes = _load_color_model(species)
    colour_probs = {}
    if scaler is not None:
        # Match the same downscale the colour KNN's training pipeline uses
        # (color_knn_module.MAX_IMAGE_SIDE) before extracting colour
        # features. crop_bgr here is the isolated crop at the uploaded
        # photo's ORIGINAL resolution (this pipeline no longer resizes the
        # whole photo down first). The colour module's texture/gradient
        # features are absolute-scale, so without capping to the same
        # working size training used, a large phone photo produces gradient
        # values far outside anything the model was trained on and biases
        # predictions toward Rotten. Scoped to this colour call only --
        # does not affect raw_crop_bgr, which defect/stem detection below
        # still receive unchanged.
        colour_input = crop_bgr
        _colour_h, _colour_w = colour_input.shape[:2]
        _colour_max_side = getattr(color_knn_module, "MAX_IMAGE_SIDE", 500)
        if max(_colour_h, _colour_w) > _colour_max_side:
            _colour_scale = _colour_max_side / float(max(_colour_h, _colour_w))
            colour_input = cv2.resize(
                colour_input,
                (max(1, int(_colour_w * _colour_scale)), max(1, int(_colour_h * _colour_scale))),
                interpolation=cv2.INTER_AREA,
            )
        label, conf, colour_probs = color_knn_module.predict_ripeness_from_color(colour_input, scaler, knn_model, classes)
        result.colour_quality = label
        result.colour_confidence = conf
    else:
        result.colour_quality = None
        result.colour_confidence = 0.0

    if species in DEFECT_SUPPORTED_SPECIES:
        try:
            # Mango defect detection can reuse the segmentation boundary that
            # the overall pipeline already computed for this ROI. This avoids
            # re-admitting leaves/background/crop corners into the Mango mask.
            # Other fruit types keep their exact existing call path.
            if species == "Mango":
                defect_out = detect_defect(
                    raw_crop_bgr,
                    species,
                    fruit_mask_hint=type_mask,
                )
            else:
                defect_out = detect_defect(
                    raw_crop_bgr,
                    species,
                )
            # defect_detection.py's helpers mostly return (annotated, mask, percentage);
            # be defensive since this module was built independently and not touched here.
            if isinstance(defect_out, tuple) and len(defect_out) >= 1:
                numeric = [v for v in defect_out if isinstance(v, (int, float))]
                result.defect_percentage = float(numeric[-1]) if numeric else None
                # The first element is the crop with defect region(s) outlined in
                # red -- same shape/dtype as raw_crop_bgr -- keep it for display.
                images = [
                    v for v in defect_out
                    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3
                ]
                if images:
                    result.defect_marked_crop = images[0]
            elif isinstance(defect_out, (int, float)):
                result.defect_percentage = float(defect_out)
            else:
                result.defect_note = "Unrecognised return shape from defect_detection"
        except Exception as e:
            result.defect_note = f"defect_detection error: {e}"
        _dbg(f"analyse_fruit: species={species} defect_percentage={result.defect_percentage} "
             f"raw_crop_shape={raw_crop_bgr.shape[:2]} isolated_crop_shape={crop_bgr.shape[:2]}")
    else:
        result.defect_note = f"Not supported for {species} by the current defect module"

    defect_label, defect_conf = None, 0.0
    if species in RIPENESS_RULE_SPECIES:
        try:
            ripeness_out = classify_ripeness(
                raw_crop_bgr, species, result.defect_percentage or 0.0
            )
            raw_ripeness = ripeness_out.get("ripeness")
            colour1 = ripeness_out.get("colour1", 0.0)
            colour2 = ripeness_out.get("colour2", 0.0)
            defect_label = _RIPENESS_TO_QUALITY.get(raw_ripeness)
            defect_conf = _defect_ripeness_confidence(raw_ripeness, colour1, colour2, result.defect_percentage)
            result.defect_ripeness = defect_label
            result.defect_ripeness_confidence = defect_conf
            _dbg(f"analyse_fruit: species={species} raw_ripeness={raw_ripeness} "
                 f"colour1(green)={colour1:.2f} colour2(red)={colour2:.2f} "
                 f"defect_percentage={result.defect_percentage} -> defect_label={defect_label}")
        except Exception as e:
            _dbg(f"analyse_fruit: ripeness_detection error: {e}")

    try:
        # Stem detection has two real candidate algorithms (YOLO model and
        # classical HSV/contour heuristics). Run both and keep whichever one
        # produced the higher-confidence detection for this specific fruit,
        # instead of always preferring YOLO first (the old "Automatic" mode).
        #
        # Runs on raw_crop_bgr (NOT the isolated/background-blackened
        # crop_bgr) -- confirmed via ASS's own app.py: its standalone stem
        # tool runs on the plain un-segmented photo, never on an
        # isolated/masked crop. A stem/calyx sits right at the fruit's own
        # edge, so blackening everything outside the contour is exactly the
        # kind of edge artefact that would confuse stem detection, same
        # reasoning already applied to defect_detection/ripeness_detection.
        stem_detector = _load_stem_detector()
        yolo_detections, _elapsed_a, _m1 = stem_detector.detect(raw_crop_bgr, species, method="YOLO")
        traditional_detections, _elapsed_b, _m2 = stem_detector.detect(raw_crop_bgr, species, method="Traditional")

        yolo_best = max((d.confidence for d in yolo_detections), default=0.0)
        traditional_best = max((d.confidence for d in traditional_detections), default=0.0)

        detections = yolo_detections if yolo_best >= traditional_best else traditional_detections
        result.stem_detected = len(detections) > 0
        result.stem_confidence = max((d.confidence for d in detections), default=0.0)
        # Draw the winning method's box/contour onto the same crop it ran
        # on, so the UI can show it right next to the fruit photo -- same
        # idea as StemDetector.annotate() already used by the standalone
        # stem app, just applied per-fruit here. Only when a stem was
        # actually found -- otherwise leave stem_crop blank instead of
        # showing an unannotated duplicate of the fruit photo.
        result.stem_crop = stem_detector.annotate(raw_crop_bgr, detections) if result.stem_detected else None
    except Exception:
        result.stem_detected = False
        result.stem_confidence = 0.0
        result.stem_crop = None

    final_label, final_conf, note = _fuse_quality(
        species, result.colour_quality, result.colour_confidence, colour_probs,
        defect_label, defect_conf, result.defect_percentage, result.stem_detected,
    )
    result.final_quality = final_label
    result.final_quality_confidence = final_conf
    result.quality_note = note

    return result


def _isolate_and_crop(original_bgr, local_contour, shift_x, shift_y, mask_erode=10, extra_erode=6, pad_frac=0.06):
    """
    Reproduces colorDetection.py's crop_object(..., isolate=True): blank out
    every pixel outside this fruit's own contour (so a touching, neighbouring
    fruit doesn't bleed into the crop), then crop tightly to the contour's
    own bounding box with a small pad -- not the original padded YOLO box.
    """
    global_contour = local_contour + [shift_x, shift_y]
    h_img, w_img = original_bgr.shape[:2]
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(full_mask, [global_contour], -1, 255, thickness=cv2.FILLED)
    if mask_erode > 0:
        full_mask = cv2.erode(full_mask, np.ones((mask_erode, mask_erode), np.uint8))
    if extra_erode > 0:
        full_mask = cv2.erode(full_mask, np.ones((extra_erode, extra_erode), np.uint8))

    bx, by, bw, bh = cv2.boundingRect(global_contour)
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    ix0, iy0 = max(0, bx - pad_x), max(0, by - pad_y)
    ix1, iy1 = min(w_img, bx + bw + pad_x), min(h_img, by + bh + pad_y)

    isolated = cv2.bitwise_and(original_bgr, original_bgr, mask=full_mask)
    return isolated[iy0:iy1, ix0:ix1].copy()


DEFECT_NEUTRAL_FILL = (190, 190, 190)


def _neutral_isolate_and_crop(original_bgr, local_contour, shift_x, shift_y, pad_frac=0.06,
                               fill_value=DEFECT_NEUTRAL_FILL):
    """
    Same idea as _isolate_and_crop, but for defect_detection.py/
    ripeness_detection.py instead of the species CNN / colour KNN.

    Those two modules build their own "what counts as fruit" mask via HSV
    thresholds (get_fruit_mask: saturation>25 & value>20, OR a dark-pixel
    catch-all), then fill the LARGEST external contour of that mask solid --
    so if a neighbouring, touching fruit (or the shadow gap between two
    touching fruits) falls inside the same raw rectangular box, it can get
    silently welded into "this fruit"'s mask, and the dark gap/neighbour
    pixels then get flagged as rot/mould, inflating defect% (confirmed via
    FRUIT_DEBUG: a visibly clean strawberry in a 3-strawberry cluster showed
    defect_percentage=88%).

    Blacking out the background (like _isolate_and_crop does for the CNN)
    is not an option here -- that is the exact "isolated crop can make
    healthy peel look defective" problem this project already hit, since
    near-black is itself one of the rot signals those modules look for.

    Instead, fill outside-this-fruit pixels with a bright, LOW-SATURATION
    grey. That grey fails both halves of get_fruit_mask()'s test (needs
    saturation>25 for normal skin, needs value<~110-170 for the dark/rot
    catch-alls), so it is invisible to the fruit-mask step and cannot be
    mistaken for either "more fruit" or "rot" -- it just disappears,
    correctly excluding the neighbour/gap without darkening real edges.
    """
    global_contour = local_contour + [shift_x, shift_y]
    h_img, w_img = original_bgr.shape[:2]
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(full_mask, [global_contour], -1, 255, thickness=cv2.FILLED)

    bx, by, bw, bh = cv2.boundingRect(global_contour)
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    ix0, iy0 = max(0, bx - pad_x), max(0, by - pad_y)
    ix1, iy1 = min(w_img, bx + bw + pad_x), min(h_img, by + bh + pad_y)

    neutral = np.full_like(original_bgr, fill_value, dtype=np.uint8)
    composed = np.where(full_mask[:, :, None] > 0, original_bgr, neutral)
    return composed[iy0:iy1, ix0:ix1].copy()


def run_overall_pipeline(original_bgr, yolo_confidence=YOLO_CONF, yolo_iou=YOLO_IOU):
    """Detect every fruit in the photo (fruit_yolo_v4 + classical-segmentation
    fallback) and analyse each one.

    Returns (results, display_image). The pipeline keeps the uploaded image's
    original resolution so YOLO receives exactly the same pixels/aspect ratio
    as the standalone defect test.
    """
    original_bgr = original_bgr.copy()

    # ImageProcessing_ASS's colorDetection.py resizes the WHOLE uploaded
    # photo to a fixed 512x512 working resolution (its DEFAULT_IMAGE_SIZE,
    # `original = cv2.resize(original, image_size)`) BEFORE any detection,
    # segmentation or classification runs -- every crop_object/colour-KNN
    # call there operates on that resized copy. This pipeline stopped doing
    # that so YOLO/defect/measurement could work at the uploaded photo's
    # native resolution (deliberate, kept as-is below). But
    # segmentation_mask_and_contour's morphological kernels and the colour
    # KNN models were both tuned/trained at that same ~512px scale, so for a
    # photo landing near the KNN's decision boundary, segmenting at native
    # resolution instead of 512x512 can be enough to flip which training
    # sample ends up nearest (confirmed on a real case: the true feature
    # vector was only ~17% closer to its nearest Rotten neighbour than to
    # its nearest Fresh one -- a near-tie, not a clear-cut Rotten photo).
    # This reference copy is used ONLY below to find each fruit's local
    # contour and build the isolated crop fed to species CNN + colour KNN
    # (the main YOLO-detection loop). YOLO detection itself, defect/
    # ripeness, stem detection, physical-size measurement, and the
    # classical-segmentation fallback loop all keep using original_bgr at
    # its native resolution, unchanged.
    _seg_h, _seg_w = original_bgr.shape[:2]
    _seg_ref_bgr = cv2.resize(original_bgr, (512, 512))
    _seg_scale_x = 512.0 / _seg_w
    _seg_scale_y = 512.0 / _seg_h

    def _to_seg_ref_box(bx0, by0, bx1, by1):
        rx0 = max(0, min(511, int(round(bx0 * _seg_scale_x))))
        ry0 = max(0, min(511, int(round(by0 * _seg_scale_y))))
        rx1 = max(rx0 + 1, min(512, int(round(bx1 * _seg_scale_x))))
        ry1 = max(ry0 + 1, min(512, int(round(by1 * _seg_scale_y))))
        return rx0, ry0, rx1, ry1

    # Run the teammate's independent morphological/texture module ONCE per
    # photo (it does its own whole-image YOLO pass), so its per-object
    # species guess is available as the third species candidate below AND
    # its geometry/texture numbers can be attached in the same pass --
    # no need to run it a second time at the end.
    morph_objects = None  # None = module unavailable/errored; [] = ran fine, found nothing
    morph_status = None
    if morph_texture_module is None:
        morph_status = f"Morphological module unavailable: {_MORPH_IMPORT_ERROR}"
    else:
        try:
            morph_out = morph_texture_module.inspect_image_morph_texture(original_bgr)
            morph_objects = morph_out.get("objects", [])
            if not morph_objects:
                morph_status = "Morphological module found no matching object in this photo"
        except Exception as e:
            _dbg(f"run_overall_pipeline: morphological module error: {e}")
            morph_status = f"Morphological module error: {e}"

    def _match_morph(bbox_xywh):
        """Best-IoU match for this box against morph_objects, or (None, 0.0)."""
        if not morph_objects:
            return None, 0.0
        best_obj, best_iou = None, 0.0
        for obj in morph_objects:
            iou = _box_iou_xywh_xyxy(bbox_xywh, obj["box"])
            if iou > best_iou:
                best_iou, best_obj = iou, obj
        if best_obj is None or best_iou < MORPH_MATCH_MIN_IOU:
            return None, 0.0
        return best_obj, best_iou

    # Segment once and reuse for both Strawberry evidence pre-scan and the
    # Mango/Strawberry fallback pass later.
    try:
        segmented_blobs = segment_all_objects(original_bgr)
    except Exception as e:
        _dbg(f"run_overall_pipeline: segment_all_objects error: {e}")
        segmented_blobs = []

    verified_strawberry_blobs = _prescan_verified_strawberry_blobs(
        original_bgr, segmented_blobs
    )
    _dbg(
        f"run_overall_pipeline: verified Strawberry evidence blobs="
        f"{len(verified_strawberry_blobs)}"
    )

    yolo_model = _load_yolo()
    # agnostic_nms + low conf on purpose: fruit_yolo_v4 only knows Apple/
    # Banana/Orange, so a real Mango/Strawberry can only surface as a WEAK
    # guess in one of those classes. Rejecting low-confidence boxes here
    # would throw away the only localisation evidence choose_cnn_override
    # needs to correct the label later.
    yolo_results = yolo_model.predict(original_bgr, conf=yolo_confidence, iou=yolo_iou, agnostic_nms=True, verbose=False)

    results = []
    claimed_mask = np.zeros(original_bgr.shape[:2], dtype=np.uint8)
    image_area = original_bgr.shape[0] * original_bgr.shape[1]
    accepted_boxes = []  # (x0, y0, x1p, y1p, area) of every box that survived dedup, for the next box's dedup check

    for r in yolo_results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            yolo_label = names.get(cls_id, str(cls_id))
            if yolo_label not in YOLO_FRUIT_CLASS_NAMES:
                continue
            yolo_conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            h_img, w_img = original_bgr.shape[:2]
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(x1 + 1, min(x2, w_img))
            y2 = max(y1 + 1, min(y2, h_img))

            # Match test_defect.py exactly: duplicate/tiny-box checks use the
            # ORIGINAL YOLO rectangle, not an expanded/padded rectangle.
            box_area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            is_fragment = any(
                ox1 <= cx <= ox2 and oy1 <= cy <= oy2
                and box_area < oarea * DUPLICATE_FRAGMENT_MAX_AREA_RATIO
                for ox1, oy1, ox2, oy2, oarea in accepted_boxes
            )
            if is_fragment:
                _dbg(f"REJECT duplicate fragment: {yolo_label}({yolo_conf:.3f}) box={(x1,y1,x2,y2)}")
                continue
            if box_area / image_area < MIN_BOX_AREA_RATIO:
                _dbg(f"REJECT tiny box: {yolo_label}({yolo_conf:.3f}) box={(x1,y1,x2,y2)}")
                continue
            accepted_boxes.append((x1, y1, x2, y2, box_area))

            # Use the raw YOLO box for the integrated path too. This keeps
            # localisation identical to the standalone defect test.
            x0, y0, x1p, y1p = x1, y1, x2, y2

            # ASS's colorDetection.py (inspect_image_yolo) pads the YOLO box
            # by 8% before running segmentation on it -- that margin is what
            # lets estimate_background_chroma() (segmentation.py) sample real
            # background from the crop's own border. This pipeline builds
            # crop_for_seg from the raw, unpadded YOLO box instead (kept
            # deliberately just above for the duplicate/tiny-box checks and
            # to match test_defect.py's localisation) -- when the YOLO box is
            # tight around the fruit, the crop's border is fruit, not
            # background, and the chroma estimate it feeds into is wrong.
            # Add the same 8% padding ASS uses, scoped ONLY to this
            # segmentation crop; x0/y0/x1p/y1p themselves (and therefore
            # raw_crop/defect_raw_crop/bbox_xyxy below) stay unpadded.
            _seg_pad_x = int(round((x1p - x0) * 0.08))
            _seg_pad_y = int(round((y1p - y0) * 0.08))
            _seg_box_x0 = max(0, x0 - _seg_pad_x)
            _seg_box_y0 = max(0, y0 - _seg_pad_y)
            _seg_box_x1 = min(w_img, x1p + _seg_pad_x)
            _seg_box_y1 = min(h_img, y1p + _seg_pad_y)
            _rx0, _ry0, _rx1, _ry1 = _to_seg_ref_box(_seg_box_x0, _seg_box_y0, _seg_box_x1, _seg_box_y1)
            crop_for_seg = prep.preprocess_image(_seg_ref_bgr[_ry0:_ry1, _rx0:_rx1], denoise_method="median", enhance_method="none")
            local_mask, local_contour = segmentation_mask_and_contour(crop_for_seg)
            used_fallback_rect = local_contour is None
            if used_fallback_rect:
                ch, cw = crop_for_seg.shape[:2]
                local_contour = np.array([[[0, 0]], [[cw - 1, 0]], [[cw - 1, ch - 1]], [[0, ch - 1]]])
                local_mask = np.full((ch, cw), 255, dtype=np.uint8)
            # A synthetic full-box rectangle's shape says nothing about the
            # real object, so it must NOT be passed to the species shape gate.
            shape_contour = None if used_fallback_rect else local_contour

            # Isolate this fruit's own pixels before classification, exactly like
            # the original colorDetection.py's crop_object(isolate=True): without
            # this, two touching fruits in the same padded box bleed into each
            # other's crop and can flip the CNN's species guess.
            crop = _isolate_and_crop(_seg_ref_bgr, local_contour, _rx0, _ry0)
            raw_crop = original_bgr[y0:y1p, x0:x1p]
            if crop.size == 0:
                crop = raw_crop

            # IMPORTANT: keep the true rectangular YOLO ROI for
            # defect detection, ripeness and stem analysis.
            #
            # The isolated/neutral-grey crop is useful for CNN species
            # classification, but it must NOT be used for the defect display.
            # The standalone defect test also calls detect_defect(raw_roi, ...),
            # so using raw_crop here keeps the annotated result looking like the
            # original fruit crop with only the red defect outline added.
            defect_raw_crop = raw_crop

            morph_obj, morph_iou = _match_morph((x0, y0, x1p - x0, y1p - y0))
            if morph_obj is not None:
                morph_species = morph_obj["fruit_type"].capitalize() if morph_obj.get("fruit_type") else None
                morph_confidence = float(morph_obj["fruit_type_confidence"]) if morph_obj.get("fruit_type") else 0.0
            else:
                # Morph V12 is optional third-vote evidence. It must never
                # delete a valid YOLO fruit when it has no spatial match.
                morph_species = None
                morph_confidence = 0.0
                _dbg(
                    f"NO MORPH MATCH: keeping {yolo_label}({yolo_conf:.3f}) "
                    f"box=({x0},{y0},{x1p},{y1p}) and continuing with YOLO+CNN/rules"
                )

            result = analyse_fruit(
                crop, yolo_label, yolo_conf,
                mask=local_mask,
                contour=shape_contour,
                raw_crop_bgr=defect_raw_crop,
                type_raw_crop_bgr=raw_crop,
                type_mask=local_mask,
                bbox_xyxy=(x0, y0, x1p, y1p),
                verified_strawberry_blobs=verified_strawberry_blobs,
                morph_species=morph_species,
                morph_confidence=morph_confidence,
                morph_obj=morph_obj,
                morph_iou=morph_iou,
                morph_status=morph_status,
            )

            # Confidence floors applied AFTER species override, and only if
            # the CNN did NOT move it away from Apple/Banana -- a real Mango/
            # Strawberry showing up as a low-confidence Apple/Banana box must
            # not be thrown away before the CNN gets a chance to correct it.
            # No floor for Orange: a badly rotten orange may score lower.
            # Rejected here (not a confirmed CNN override candidate), so it
            # does NOT claim this region -- the fallback pass may still find
            # a real fruit here.
            #
            # BUG FIX (confirmed via FRUIT_DEBUG on a real 3-mango photo):
            # this must only fire when species is Apple/Banana BECAUSE that
            # is literally YOLO's own uncorrected guess. If a different
            # mechanism (choose_cnn_override, KNOWN_CLASS_OVERRIDE, or the
            # morph candidate winning the 3-way comparison) already moved
            # the species TO Apple/Banana -- e.g. YOLO said Banana(0.57) but
            # the CNN override corrected it to Apple(0.92) -- that route
            # already has its own, separate confidence gate, so re-applying
            # this floor against YOLO's original (pre-override) confidence
            # was silently discarding a correctly-identified fruit. This is
            # exactly how a real 3rd mango vanished: YOLO's own guess for it
            # was Banana(0.57), the CNN override correctly flagged it as
            # Apple(0.92) via known_class override, but the floor below then
            # rejected the whole box because 0.57 < APPLE_MIN_CONF.
            species_is_raw_yolo_guess = result.species == result.yolo_species
            if species_is_raw_yolo_guess and result.species == "Apple" and yolo_conf < APPLE_MIN_CONF:
                continue
            if species_is_raw_yolo_guess and result.species == "Banana" and yolo_conf < BANANA_MIN_CONF:
                continue

            cv2.rectangle(claimed_mask, (x0, y0), (x1p, y1p), 255, thickness=-1)
            result.bbox = (x0, y0, x1p - x0, y1p - y0)
            results.append(result)
            _dbg(f"run_overall_pipeline: main-loop box claimed region=({x0},{y0})-({x1p},{y1p}) "
                 f"yolo_label={yolo_label}({yolo_conf:.3f}) final_species={result.species}")

    # Fallback pass: classical segmentation for fruit YOLO's base weights can't
    # name at all (Mango, Strawberry), mirroring the original system's approach.
    denoised_full = prep.denoise(original_bgr, method="median")
    for blob in segmented_blobs:
        bx, by, bw, bh = blob["bbox"]
        blob_x1, blob_y1 = int(bx), int(by)
        blob_x2, blob_y2 = int(bx + bw), int(by + bh)
        blob_area = max(1, int(bw) * int(bh))

        # Match test_defect.py's confirmed-box idea: skip only when MOST of
        # this segmented blob is already covered by a fruit that actually
        # survived the main YOLO path. Do not use a single centre-point test.
        already_detected = False
        for confirmed in results:
            cx, cy, cw, ch = confirmed.bbox
            conf_x2, conf_y2 = cx + cw, cy + ch
            ix1, iy1 = max(blob_x1, cx), max(blob_y1, cy)
            ix2, iy2 = min(blob_x2, conf_x2), min(blob_y2, conf_y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap_ratio = inter / blob_area
            threshold = 0.75 if confirmed.species == "Strawberry" else 0.45
            if overlap_ratio >= threshold:
                already_detected = True
                break

        if already_detected:
            _dbg(
                f"fallback blob bbox=({bx},{by},{bw},{bh}) SKIPPED -- "
                f"mostly covered by a confirmed fruit"
            )
            continue
        _dbg(f"fallback blob bbox=({bx},{by},{bw},{bh}) accepted, analysing...")
        # blob["contour"]/["mask"] are already in full-image coordinates.
        # NOTE: unlike the main loop, colorDetection.py's fallback pass does
        # NOT apply the extra 10px mask erosion (that only happens when a
        # fresh mask is drawn from a YOLO-box contour) -- blob["mask"] goes
        # straight into crop_object's isolate step, which only adds its own
        # fixed 6px erosion. Using the main loop's 10+6=16px here over-erodes
        # smaller/concave blobs (e.g. strawberries with a calyx notch),
        # collapsing the mask to near-nothing and making the isolated crop
        # mostly black -- which defect_detection then misreads as rot.
        crop = _isolate_and_crop(original_bgr, blob["contour"], 0, 0, mask_erode=0, extra_erode=6)
        raw_crop = original_bgr[by:by + bh, bx:bx + bw]
        if crop.size == 0:
            crop = raw_crop
        if crop.size == 0:
            continue
        # Keep the original rectangular segmented-object ROI for
        # defect/ripeness/stem, matching the standalone defect pipeline.
        defect_raw_crop = raw_crop
        blob_full_mask = blob.get("mask")
        blob_local_mask = None
        if isinstance(blob_full_mask, np.ndarray):
            candidate_local_mask = blob_full_mask[by:by + bh, bx:bx + bw]
            if candidate_local_mask.shape[:2] == raw_crop.shape[:2]:
                blob_local_mask = candidate_local_mask
        blob_roughness = compute_texture_roughness(denoised_full, blob["mask"])
        morph_obj, morph_iou = _match_morph((bx, by, bw, bh))
        if morph_obj is not None:
            morph_species = morph_obj["fruit_type"].capitalize() if morph_obj.get("fruit_type") else None
            morph_confidence = float(morph_obj["fruit_type_confidence"]) if morph_obj.get("fruit_type") else 0.0
        else:
            morph_species = None
            morph_confidence = 0.0
            _dbg(f"NO MORPH MATCH on fallback blob ({bx},{by},{bw},{bh}) -- keeping candidate")
        result = analyse_fruit(
            crop, "unknown", 0.0,
            mask=blob["mask"],
            contour=blob["contour"],
            roughness=blob_roughness,
            raw_crop_bgr=defect_raw_crop,
            type_raw_crop_bgr=raw_crop,
            type_mask=blob_local_mask,
            bbox_xyxy=(bx, by, bx + bw, by + bh),
            verified_strawberry_blobs=verified_strawberry_blobs,
            morph_species=morph_species,
            morph_confidence=morph_confidence,
            morph_obj=morph_obj,
            morph_iou=morph_iou,
            morph_status=morph_status,
        )
        if result.species not in FALLBACK_ONLY_SPECIES:
            continue

        # Apple/Banana/Orange used to be discarded here because the fallback
        # was restricted to Mango/Strawberry only. That means a damaged Orange
        # missed by whole-image YOLO could be correctly segmented/classified,
        # then silently thrown away. Allow native species to recover too, but
        # require stronger independent evidence so background blobs do not turn
        # into false fruit detections.
        if result.species in YOLO_NATIVE_SPECIES:
            cnn_native_ok = (
                result.cnn_rule_species == result.species
                and result.cnn_rule_confidence >= FALLBACK_NATIVE_MIN_CNN_CONF
            )
            morph_native_ok = (
                result.morph_fruit_type == result.species
                and result.morph_fruit_type_confidence >= FALLBACK_NATIVE_MIN_MORPH_CONF
            )
            cnn_morph_agree = (
                result.cnn_rule_species == result.species
                and result.morph_fruit_type == result.species
                and result.cnn_rule_confidence >= 0.65
                and result.morph_fruit_type_confidence >= 0.50
            )

            if not (cnn_native_ok or morph_native_ok or cnn_morph_agree):
                _dbg(
                    f"REJECT fallback {result.species}: "
                    f"CNN={result.cnn_rule_species}({result.cnn_rule_confidence:.3f}) "
                    f"Morph={result.morph_fruit_type}({result.morph_fruit_type_confidence:.3f})"
                )
                continue

        _dbg(
            f"ACCEPT fallback fruit: species={result.species} "
            f"CNN={result.cnn_rule_species}({result.cnn_rule_confidence:.3f}) "
            f"Morph={result.morph_fruit_type}({result.morph_fruit_type_confidence:.3f}) "
            f"bbox=({bx},{by},{bw},{bh})"
        )
        result.bbox = (bx, by, bw, bh)
        results.append(result)

    return results, original_bgr


def _box_iou_xywh_xyxy(box_xywh, box_xyxy):
    """IoU between our own (x, y, w, h) bbox and morph_texture_module's
    (x1, y1, x2, y2) bbox -- both are already in the same 512x512 image
    coordinate frame since both run on the same resized `original_bgr`."""
    ax0, ay0, aw, ah = box_xywh
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bx1, by1 = box_xyxy
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


MORPH_MATCH_MIN_IOU = 0.3


# ---------------------------------------------------------------
# Shared display helpers -- used by both app.py (Streamlit UI) and
# report.py (PDF export), so the two never drift apart on box colours.
# ---------------------------------------------------------------
BOX_COLOURS = {"Fresh": (0, 200, 0), "Unripe": (0, 165, 255), "Rotten": (0, 0, 255), "Uncertain": (255, 255, 0)}


def draw_annotations(img_bgr, fruits):
    """Draw each fruit's bbox + 'species: quality' label, colour-coded by
    final_quality, onto a copy of img_bgr (BGR, same frame fr.bbox is in)."""
    annotated = img_bgr.copy()
    for fr in fruits:
        x, y, w, h = [int(round(v)) for v in fr.bbox]
        colour = BOX_COLOURS.get(fr.final_quality, (255, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        label = f"{fr.species or '?'}: {fr.final_quality or '?'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(0, y - th - 8)
        cv2.rectangle(annotated, (x, label_y), (x + tw + 6, label_y + th + 8), colour, -1)
        cv2.putText(annotated, label, (x + 3, label_y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
