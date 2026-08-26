import os
import tempfile
import io
import contextlib
import importlib.util
from pathlib import Path
import re
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

import calibration as calib
from colorDetection import inspect_image_yolo, classify_fruit_type_cnn
from segmentation import contour_shape_metrics, segment_all_objects
import report as report_mod


# ======================================================
# TIFANY — EXACT LATEST DEFECT DETECTION
# ======================================================
# Source logic follows the latest uploaded test_defect.py.
#
# Only the desktop test wrapper is changed:
#   - cv.imread(...) is replaced by Streamlit uploaded image
#   - cv.imshow()/waitKey() are removed
#
# Detection thresholds, preprocessing, segmentation,
# CNN correction/fallback, Strawberry rules, duplicate
# suppression, defect detection and ripeness logic are kept.

PROJECT_ROOT = Path(__file__).resolve().parent
DEFECT_DIR = PROJECT_ROOT / "defect_detection"

CUSTOM_MODULES_AVAILABLE = False
CUSTOM_MODULE_ERROR = None
detect_defect = None
classify_ripeness = None


def _load_local_function(file_path, module_name, function_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(
        module,
        function_name,
    )


try:
    detect_defect = _load_local_function(
        DEFECT_DIR / "defect_detection.py",
        "tifany_exact_defect_detection",
        "detect_defect",
    )

    classify_ripeness = _load_local_function(
        DEFECT_DIR / "ripeness_detection.py",
        "tifany_exact_ripeness_detection",
        "classify_ripeness",
    )

    CUSTOM_MODULES_AVAILABLE = True

except Exception as exc:
    CUSTOM_MODULE_ERROR = str(exc)


# EXACT model path from latest test_defect.py
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "fruit_yolo_v4"
    / "weights"
    / "best.pt"
)


# =================================================
# SETTINGS
# =================================================

# YOLO
YOLO_CONF = 0.25
YOLO_IOU = 0.45

# Class-specific confidence filtering
APPLE_MIN_CONF = 0.85
BANANA_MIN_CONF = 0.55

# Do NOT add another Orange filter.
# YOLO's global 0.50 threshold is enough because a badly
# rotten orange may have lower confidence.

# Remove only extremely tiny boxes
MIN_BOX_AREA_RATIO = 0.002

# CNN correction for Mango / Strawberry
#
# Apple / Banana / Orange remain YOLO-first.
# The friend's CNN is allowed to correct only a weaker YOLO result,
# with extra colour / shape checks so a real Orange is not changed
# into Strawberry or Mango.
CNN_RECHECK_MAX_YOLO_CONF = 0.85

# Strawberry safety checks
STRAWBERRY_MIN_CONF = 0.95
STRAWBERRY_MAX_CIRCULARITY = 0.82
STRAWBERRY_MIN_RED_RATIO = 0.70

# If the fruit surface is overwhelmingly red, allow a slightly
# lower CNN confidence. This helps damaged / irregular strawberries
# without making normal oranges easy to override.
STRAWBERRY_STRONG_RED_RATIO = 0.80
STRAWBERRY_STRONG_RED_MIN_CONF = 0.90

# Unripe / green Strawberry:
# - friend's CNN must be extremely confident
# - red surface is naturally very low
# - strawberry shape must be clearly less circular than an orange
# - slight elongation helps separate it from round citrus fruit
STRAWBERRY_GREEN_MIN_CONF = 0.98
STRAWBERRY_GREEN_MAX_RED_RATIO = 0.15
STRAWBERRY_GREEN_MAX_CIRCULARITY = 0.65
STRAWBERRY_GREEN_MIN_ASPECT = 1.08

# Damaged / rotten Strawberry:
# Rotten strawberries can lose a lot of normal red colour, so the
# normal "red >= 70%" rule may reject them. Use a very confident
# Strawberry CNN result + non-round strawberry shape instead.
#
# These values are deliberately stricter than the normal rule so the
# earlier mouldy Orange false positives remain Orange.
STRAWBERRY_DAMAGED_MIN_CONF = 0.97
STRAWBERRY_DAMAGED_MIN_RED_RATIO = 0.15
STRAWBERRY_DAMAGED_MAX_RED_RATIO = 0.70
STRAWBERRY_DAMAGED_MAX_CIRCULARITY = 0.72
STRAWBERRY_DAMAGED_MIN_ASPECT = 1.02

# Mango safety checks
MANGO_MIN_CONF = 0.85
MANGO_MIN_ASPECT = 1.18
MANGO_MAX_CIRCULARITY = 0.90

# Segmented fallback object is considered already handled when
# most of it is covered by an accepted YOLO box.
YOLO_BLOB_OVERLAP_THRESHOLD = 0.55


# =================================================
# 1. PREPROCESSING
# =================================================

def preprocess_roi(roi):
    """
    Preprocessing:
    1. Reduce small image noise
    2. Improve local brightness/contrast slightly

    Kept conservative because defect detection uses colour.
    """

    # Small Gaussian blur
    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    # CLAHE on brightness channel only
    lab = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv.split(
        lab
    )

    clahe = cv.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    enhanced_lab = cv.merge(
        (
            l_channel,
            a_channel,
            b_channel
        )
    )

    enhanced = cv.cvtColor(
        enhanced_lab,
        cv.COLOR_LAB2BGR
    )

    return enhanced


# =================================================
# 2. SEGMENTATION
# =================================================

def create_fruit_mask(roi):
    """
    Extract fruit from the YOLO crop using GrabCut + spatial priors.

    Why:
    HSV-only segmentation is unreliable when a green fruit is
    surrounded by green leaves. GrabCut uses colour, texture,
    edges and foreground/background seeds.

    Returns:
        fruit_mask
        contour
    """

    h, w = roi.shape[:2]

    if h < 10 or w < 10:
        return (
            np.zeros((h, w), dtype=np.uint8),
            None
        )

    # -------------------------------------------------
    # GrabCut initial mask
    # -------------------------------------------------

    gc_mask = np.full(
        (h, w),
        cv.GC_PR_BGD,
        dtype=np.uint8
    )

    # Definite background: thin outer border
    border = max(
        2,
        int(min(h, w) * 0.04)
    )

    gc_mask[:border, :] = cv.GC_BGD
    gc_mask[h - border:, :] = cv.GC_BGD
    gc_mask[:, :border] = cv.GC_BGD
    gc_mask[:, w - border:] = cv.GC_BGD

    # Probable foreground: central part of YOLO box
    x1 = int(w * 0.12)
    y1 = int(h * 0.12)
    x2 = int(w * 0.88)
    y2 = int(h * 0.88)

    gc_mask[
        y1:y2,
        x1:x2
    ] = cv.GC_PR_FGD

    # Strong foreground seed in the centre.
    # This helps when green fruit is surrounded by green leaves.
    center = (
        w // 2,
        h // 2
    )

    axes = (
        max(5, int(w * 0.22)),
        max(5, int(h * 0.22))
    )

    cv.ellipse(
        gc_mask,
        center,
        axes,
        0,
        0,
        360,
        cv.GC_FGD,
        cv.FILLED
    )

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv.grabCut(
            roi,
            gc_mask,
            None,
            bg_model,
            fg_model,
            6,
            cv.GC_INIT_WITH_MASK
        )

        fruit_mask = np.where(
            (gc_mask == cv.GC_FGD)
            | (gc_mask == cv.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

    except cv.error:

        # Safe fallback if GrabCut fails
        fruit_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv.ellipse(
            fruit_mask,
            center,
            (
                max(5, int(w * 0.38)),
                max(5, int(h * 0.38))
            ),
            0,
            0,
            360,
            255,
            cv.FILLED
        )

    # -------------------------------------------------
    # Clean mask
    # -------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    fruit_mask = cv.morphologyEx(
        fruit_mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )

    fruit_mask = cv.morphologyEx(
        fruit_mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # -------------------------------------------------
    # Keep the best central connected component
    # instead of blindly taking the largest region.
    # -------------------------------------------------

    contours, _ = cv.findContours(
        fruit_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return (
            np.zeros((h, w), dtype=np.uint8),
            None
        )

    image_center = np.array(
        [
            w / 2.0,
            h / 2.0
        ],
        dtype=np.float32
    )

    best_contour = None
    best_score = -1.0

    roi_area = float(
        w * h
    )

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area < roi_area * 0.03:
            continue

        moments = cv.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        distance = np.linalg.norm(
            np.array(
                [cx, cy],
                dtype=np.float32
            )
            - image_center
        )

        max_distance = max(
            1.0,
            np.hypot(
                w / 2.0,
                h / 2.0
            )
        )

        centrality = (
            1.0
            - min(
                distance / max_distance,
                1.0
            )
        )

        perimeter = cv.arcLength(
            contour,
            True
        )

        circularity = (
            (4.0 * np.pi * area)
            / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )

        area_ratio = (
            area
            / roi_area
        )

        # Area matters most, followed by central position.
        # Circularity helps reject thin leaf regions.
        score = (
            area_ratio * 0.55
            + centrality * 0.30
            + min(circularity, 1.0) * 0.15
        )

        if score > best_score:

            best_score = score
            best_contour = contour

    if best_contour is None:

        best_contour = max(
            contours,
            key=cv.contourArea
        )

    clean_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv.drawContours(
        clean_mask,
        [best_contour],
        -1,
        255,
        cv.FILLED
    )

    # -------------------------------------------------
    # Smooth the fruit boundary slightly
    # -------------------------------------------------

    clean_mask = cv.morphologyEx(
        clean_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (7, 7),
            np.uint8
        ),
        iterations=1
    )

    # Re-find contour after cleanup
    final_contours, _ = cv.findContours(
        clean_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    final_contour = (
        max(
            final_contours,
            key=cv.contourArea
        )
        if final_contours
        else best_contour
    )

    return (
        clean_mask,
        final_contour
    )


# =================================================
# STRAWBERRY-SPECIFIC SEGMENTATION
# =================================================

def create_strawberry_mask(roi):
    """
    Create a strawberry body mask inside the YOLO crop.

    Main purpose:
    - keep strawberry body
    - remove soil / leaves / outside background
    - still allow a brown/rotten area attached to the strawberry

    This does NOT change Apple/Banana/Orange/Mango logic.
    """

    h, w = roi.shape[:2]

    if h < 10 or w < 10:
        return np.zeros(
            (h, w),
            dtype=np.uint8
        )

    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    # ---------------------------------------------
    # GrabCut mask
    # ---------------------------------------------

    gc_mask = np.full(
        (h, w),
        cv.GC_PR_BGD,
        dtype=np.uint8
    )

    # Crop edge = definite background.
    border = max(
        2,
        int(min(h, w) * 0.02)
    )

    gc_mask[:border, :] = cv.GC_BGD
    gc_mask[h - border:, :] = cv.GC_BGD
    gc_mask[:, :border] = cv.GC_BGD
    gc_mask[:, w - border:] = cv.GC_BGD

    # Most strawberries are approximately centred in YOLO crop.
    probable_fg = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv.ellipse(
        probable_fg,
        (w // 2, h // 2),
        (
            max(5, int(w * 0.40)),
            max(5, int(h * 0.43))
        ),
        0,
        0,
        360,
        255,
        cv.FILLED
    )

    gc_mask[
        probable_fg > 0
    ] = cv.GC_PR_FGD

    # ---------------------------------------------
    # Strong ripe-strawberry RED foreground seed
    # ---------------------------------------------

    red1 = cv.inRange(
        hsv,
        np.array([0, 70, 35]),
        np.array([12, 255, 255])
    )

    red2 = cv.inRange(
        hsv,
        np.array([168, 70, 35]),
        np.array([180, 255, 255])
    )

    red_seed = cv.bitwise_or(
        red1,
        red2
    )

    safe_seed_area = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    safe_seed_area[
        border:h - border,
        border:w - border
    ] = 255

    red_seed = cv.bitwise_and(
        red_seed,
        safe_seed_area
    )

    red_pixels = cv.countNonZero(
        red_seed
    )

    roi_pixels = max(
        1,
        h * w
    )

    red_ratio = (
        red_pixels
        / roi_pixels
    )

    # ---------------------------------------------
    # Brown / rotten strawberry support
    #
    # A severely rotten strawberry may have a large brown/gray
    # section. GrabCut seeded mainly by RED pixels can otherwise
    # classify that rotten section as background.
    #
    # This mask is NOT used as a defect result here. It is only
    # used to help the Strawberry BODY mask include attached
    # rotten tissue.
    # ---------------------------------------------

    rotten_brown = cv.inRange(
        hsv,
        np.array([0, 25, 20]),
        np.array([30, 255, 175])
    )

    rotten_gray_dark = cv.inRange(
        hsv,
        np.array([0, 0, 20]),
        np.array([180, 150, 145])
    )

    rotten_candidate = cv.bitwise_or(
        rotten_brown,
        rotten_gray_dark
    )

    # Keep possible rotten tissue only in the central fruit area.
    rotten_candidate = cv.bitwise_and(
        rotten_candidate,
        probable_fg
    )

    rotten_candidate = cv.bitwise_and(
        rotten_candidate,
        safe_seed_area
    )

    if red_ratio >= 0.03:
        gc_mask[
            red_seed > 0
        ] = cv.GC_FGD

    else:
        # Unripe/green strawberry:
        # use a smaller central definite-foreground seed.
        centre_seed = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv.ellipse(
            centre_seed,
            (w // 2, h // 2),
            (
                max(4, int(w * 0.18)),
                max(4, int(h * 0.22))
            ),
            0,
            0,
            360,
            255,
            cv.FILLED
        )

        gc_mask[
            centre_seed > 0
        ] = cv.GC_FGD

    # ---------------------------------------------
    # GrabCut
    # ---------------------------------------------

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:
        cv.grabCut(
            blurred,
            gc_mask,
            None,
            bg_model,
            fg_model,
            6,
            cv.GC_INIT_WITH_MASK
        )

        mask = np.where(
            (gc_mask == cv.GC_FGD)
            | (gc_mask == cv.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

    except cv.error:
        # Safe fallback to the existing generic segmentation.
        mask, _ = create_fruit_mask(
            roi
        )

    # ---------------------------------------------
    # Recover attached rotten strawberry tissue
    #
    # Repeatedly grow from the already accepted Strawberry mask
    # into brown/gray candidate pixels that physically touch it.
    # This lets a large rotten half remain part of the fruit while
    # avoiding unrelated background that is disconnected.
    # ---------------------------------------------

    growth_kernel = np.ones(
        (9, 9),
        np.uint8
    )

    for _ in range(8):

        near_existing_fruit = cv.dilate(
            mask,
            growth_kernel,
            iterations=1
        )

        attached_rot = cv.bitwise_and(
            rotten_candidate,
            near_existing_fruit
        )

        new_mask = cv.bitwise_or(
            mask,
            attached_rot
        )

        if cv.countNonZero(
            cv.bitwise_xor(
                new_mask,
                mask
            )
        ) == 0:
            break

        mask = new_mask

    # ---------------------------------------------
    # Clean mask
    # ---------------------------------------------

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2
    )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask

    # Prefer a large component near the ROI centre.
    centre_x = w / 2.0
    centre_y = h / 2.0

    best_contour = None
    best_score = -1e9

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area < (h * w) * 0.03:
            continue

        moments = cv.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        distance = np.hypot(
            cx - centre_x,
            cy - centre_y
        )

        distance_ratio = (
            distance
            / max(
                1.0,
                np.hypot(
                    w / 2.0,
                    h / 2.0
                )
            )
        )

        area_ratio = (
            area
            / max(
                1.0,
                h * w
            )
        )

        score = (
            area_ratio * 2.0
            - distance_ratio
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        best_contour = max(
            contours,
            key=cv.contourArea
        )

    clean_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv.drawContours(
        clean_mask,
        [best_contour],
        -1,
        255,
        cv.FILLED
    )

    clean_mask = cv.morphologyEx(
        clean_mask,
        cv.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=1
    )

    return clean_mask


def refine_strawberry_defect_result(
    roi,
    defect_mask
):
    """
    Clip Strawberry defect detection to the Strawberry body only.

    Returns:
        output,
        cleaned_defect_mask,
        recalculated_percentage
    """

    strawberry_mask = create_strawberry_mask(
        roi
    )

    strawberry_pixels = cv.countNonZero(
        strawberry_mask
    )

    raw_defect_pixels = cv.countNonZero(
        defect_mask
    )

    print(
        f"[STRAWBERRY DEBUG] raw defect pixels="
        f"{raw_defect_pixels} | "
        f"strawberry pixels={strawberry_pixels}"
    )

    if strawberry_pixels == 0:
        # Extremely safe fallback.
        output = roi.copy()

        return (
            output,
            np.zeros(
                roi.shape[:2],
                dtype=np.uint8
            ),
            0.0
        )

    # Defect is allowed ONLY on the segmented strawberry.
    cleaned_defect = cv.bitwise_and(
        defect_mask,
        strawberry_mask
    )

    # Slightly remove boundary artifacts.
    distance = cv.distanceTransform(
        strawberry_mask,
        cv.DIST_L2,
        5
    )

    margin = max(
        2,
        int(
            min(
                strawberry_mask.shape
            ) * 0.008
        )
    )

    safe_surface = np.zeros_like(
        strawberry_mask
    )

    safe_surface[
        distance >= margin
    ] = 255

    cleaned_defect = cv.bitwise_and(
        cleaned_defect,
        safe_surface
    )

    # Morphological cleanup.
    cleaned_defect = cv.morphologyEx(
        cleaned_defect,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    cleaned_defect = cv.morphologyEx(
        cleaned_defect,
        cv.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Remove tiny seed/noise components.
    min_area = max(
        40,
        int(strawberry_pixels * 0.0015)
    )

    final_mask = np.zeros_like(
        cleaned_defect
    )

    contours, _ = cv.findContours(
        cleaned_defect,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv.contourArea(contour) >= min_area:

            cv.drawContours(
                final_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )

    defect_pixels = cv.countNonZero(
        final_mask
    )

    defect_percentage = (
        (defect_pixels / strawberry_pixels) * 100
        if strawberry_pixels > 0
        else 0.0
    )

    defect_percentage = min(
        defect_percentage,
        100.0
    )

    # IMPORTANT:
    # Start again from ORIGINAL ROI so old contours drawn on
    # soil/background are completely removed.
    output = roi.copy()

    final_contours, _ = cv.findContours(
        final_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in final_contours:

        if cv.contourArea(contour) >= min_area:

            cv.drawContours(
                output,
                [contour],
                -1,
                (0, 0, 255),
                2
            )

    return (
        output,
        final_mask,
        defect_percentage
    )


# =================================================
# 3. CALIBRATION
# =================================================

def calibrate_roi(roi, fruit_mask):
    """
    Brightness calibration using only pixels inside fruit.

    This reduces lighting differences while keeping the
    colour change conservative so existing HSV thresholds
    are not heavily disturbed.
    """

    if cv.countNonZero(fruit_mask) == 0:
        return roi.copy()


    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    ).astype(np.float32)

    value = hsv[:, :, 2]

    fruit_values = value[
        fruit_mask > 0
    ]


    if fruit_values.size == 0:
        return roi.copy()


    median_value = float(
        np.median(
            fruit_values
        )
    )


    if median_value <= 0:
        return roi.copy()


    # Conservative target brightness
    target_value = 145.0

    scale = (
        target_value
        / median_value
    )

    # Do not change lighting too aggressively
    scale = float(
        np.clip(
            scale,
            0.90,
            1.10
        )
    )


    value = np.clip(
        value * scale,
        0,
        255
    )

    hsv[:, :, 2] = value


    calibrated = cv.cvtColor(
        hsv.astype(np.uint8),
        cv.COLOR_HSV2BGR
    )


    return calibrated


# =================================================
# 4. EXTRACT FRUIT FROM BACKGROUND
# =================================================

def prepare_fruit_roi(roi):
    """
    Preprocessing pipeline used BEFORE defect analysis.

    IMPORTANT:
    - Preprocessing helps segmentation.
    - Segmentation extracts the fruit mask.
    - Calibration is used for the CNN/type check.
    - The existing defect detector still receives the ORIGINAL
      fruit ROI because its HSV thresholds were tuned on raw
      fruit colours. Passing a black-background calibrated image
      can make healthy green peel look defective.
    """

    processed = preprocess_roi(
        roi
    )

    fruit_mask, contour = create_fruit_mask(
        processed
    )

    calibrated = calibrate_roi(
        processed,
        fruit_mask
    )

    # Isolated calibrated crop for CNN/type classification only.
    cnn_roi = cv.bitwise_and(
        calibrated,
        calibrated,
        mask=fruit_mask
    )

    return (
        cnn_roi,
        fruit_mask,
        contour
    )


# =================================================
# MANGO / STRAWBERRY IDENTIFICATION HELPERS
# =================================================

def calculate_red_ratio(roi, mask=None):
    """
    Estimate how much of the fruit surface is strawberry-red.

    This is used only as a safety gate for Strawberry so a green
    or orange citrus fruit cannot be changed to Strawberry solely
    because the CNN is confident.
    """

    if roi is None or roi.size == 0:
        return 0.0

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    # Narrow red ranges. Orange peel is mostly outside these bands.
    red_low = cv.inRange(
        hsv,
        np.array([0, 90, 45]),
        np.array([8, 255, 255])
    )

    red_high = cv.inRange(
        hsv,
        np.array([170, 90, 45]),
        np.array([180, 255, 255])
    )

    red_mask = cv.bitwise_or(
        red_low,
        red_high
    )

    if (
        mask is not None
        and mask.shape[:2] == roi.shape[:2]
        and cv.countNonZero(mask) > 0
    ):
        red_mask = cv.bitwise_and(
            red_mask,
            mask
        )

        total = cv.countNonZero(
            mask
        )

    else:
        total = (
            roi.shape[0]
            * roi.shape[1]
        )

    if total <= 0:
        return 0.0

    return (
        cv.countNonZero(red_mask)
        / total
    )


def get_shape_values(
    contour,
    roi_width,
    roi_height
):
    """
    Return orientation-independent aspect ratio and circularity.
    """

    aspect = max(
        roi_width / max(1, roi_height),
        roi_height / max(1, roi_width)
    )

    circularity = None

    if contour is not None:

        try:
            (
                _,
                contour_aspect,
                contour_circularity
            ) = contour_shape_metrics(
                contour
            )

            if contour_aspect is not None:
                contour_aspect = float(
                    contour_aspect
                )

                aspect = max(
                    contour_aspect,
                    1.0 / max(
                        contour_aspect,
                        1e-6
                    )
                )

            circularity = contour_circularity

        except Exception:
            pass

    return (
        aspect,
        circularity
    )


def choose_cnn_override(
    yolo_confidence,
    cnn_type,
    cnn_conf,
    aspect,
    circularity,
    red_ratio
):
    """
    Safely allow Mango / Strawberry to replace a weak YOLO class.

    High-confidence YOLO Apple/Banana/Orange stays unchanged.
    """

    # IMPORTANT:
    # This YOLO was trained only on Apple / Banana / Orange.
    # Therefore a Mango or Strawberry can be classified with very
    # high confidence as one of those three classes.
    #
    # Do NOT trust YOLO confidence alone to block the friend's CNN.
    # Instead, allow Mango/Strawberry only when the CNN + safety
    # checks strongly agree.

    # Strawberry:
    # CNN + visible red surface + non-round-ish shape.
    strawberry_normal_ok = (
        cnn_type == "Strawberry"
        and cnn_conf >= STRAWBERRY_MIN_CONF
        and red_ratio >= STRAWBERRY_MIN_RED_RATIO
        and (
            circularity is None
            or circularity < 0.78
        )
    )

    strawberry_strong_red_ok = (
        cnn_type == "Strawberry"
        and cnn_conf >= STRAWBERRY_STRONG_RED_MIN_CONF
        and red_ratio >= STRAWBERRY_STRONG_RED_RATIO
        and (
            circularity is None
            or circularity < 0.78
        )
    )

    # Dedicated rule for unripe / green strawberry.
    #
    # Current real green-strawberry test:
    # CNN 98.94%, red 1.34%, aspect 1.123, circularity 0.530.
    #
    # A green orange is usually much more circular, so this branch
    # uses very high CNN confidence + low red + low circularity +
    # slight elongation.
    strawberry_green_ok = (
        cnn_type == "Strawberry"
        and cnn_conf >= STRAWBERRY_GREEN_MIN_CONF
        and red_ratio <= STRAWBERRY_GREEN_MAX_RED_RATIO
        and aspect >= STRAWBERRY_GREEN_MIN_ASPECT
        and circularity is not None
        and circularity < STRAWBERRY_GREEN_MAX_CIRCULARITY
    )

    # Rotten / damaged strawberries may be brown/gray and therefore
    # contain much less red than a healthy ripe strawberry.
    #
    # Safety:
    # earlier false Orange examples had circularity around 0.77-0.79,
    # so requiring < 0.72 helps keep those as Orange.
    strawberry_damaged_ok = (
        cnn_type == "Strawberry"
        and cnn_conf >= STRAWBERRY_DAMAGED_MIN_CONF
        and red_ratio >= STRAWBERRY_DAMAGED_MIN_RED_RATIO
        and red_ratio < STRAWBERRY_DAMAGED_MAX_RED_RATIO
        and aspect >= STRAWBERRY_DAMAGED_MIN_ASPECT
        and circularity is not None
        and circularity < STRAWBERRY_DAMAGED_MAX_CIRCULARITY
    )

    if (
        strawberry_normal_ok
        or strawberry_strong_red_ok
        or strawberry_green_ok
        or strawberry_damaged_ok
    ):
        return "strawberry"

    # Mango:
    # CNN + clearly oval / elongated fruit.
    if (
        cnn_type == "Mango"
        and cnn_conf >= MANGO_MIN_CONF
        and aspect >= MANGO_MIN_ASPECT
        and (
            circularity is None
            or circularity < MANGO_MAX_CIRCULARITY
        )
    ):
        return "mango"

    return None


_exact_defect_yolo_model = None


def get_exact_defect_yolo_model():
    global _exact_defect_yolo_model

    if _exact_defect_yolo_model is not None:
        return _exact_defect_yolo_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    from ultralytics import YOLO

    _exact_defect_yolo_model = YOLO(
        str(MODEL_PATH)
    )

    return _exact_defect_yolo_model


def run_exact_latest_defect_pipeline(uploaded_bgr):
    """
    Run the exact latest standalone test logic on a Streamlit image.

    Same image-processing/detection body as latest test_defect.py.
    """

    if not CUSTOM_MODULES_AVAILABLE:
        raise RuntimeError(
            CUSTOM_MODULE_ERROR
            or "Defect/ripeness module unavailable."
        )

    model = get_exact_defect_yolo_model()

    # Equivalent to cv.imread(...) result in the standalone test.
    image = uploaded_bgr.copy()

    if image is None or image.size == 0:
        raise ValueError("Invalid image.")

    display_image = image.copy()

    # =================================================
    # YOLO LOCALISATION
    # =================================================

    results = model.predict(
        source=image,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        agnostic_nms=True
    )


    detected = False
    fruit_count = 0

    # Store preliminary YOLO boxes so smaller duplicate / fragment
    # YOLO detections can be skipped.
    accepted_boxes = []

    # Store ONLY fruits that were actually accepted and processed.
    # The Mango/Strawberry object fallback must compare against this
    # list instead of every preliminary YOLO box.
    confirmed_boxes = []


    # =================================================
    # PRE-SCAN FOR STRONG STRAWBERRY EVIDENCE
    # =================================================
    #
    # Why:
    # YOLO only knows Apple / Banana / Orange. A badly mouldy strawberry
    # can therefore appear as one large Orange box, while the fallback
    # segmentation breaks that same fruit into two smaller Strawberry
    # fragments.
    #
    # We use those strong Strawberry fragments only as TYPE evidence.
    # The final localisation still uses the ONE large YOLO box.

    verified_strawberry_blobs = []

    try:
        prescan_blobs = segment_all_objects(
            image
        )

    except Exception as exc:
        print(
            f"Strawberry pre-scan segmentation error: {exc}"
        )
        prescan_blobs = []


    for prescan_blob in prescan_blobs:

        px, py, pw, ph = prescan_blob[
            "bbox"
        ]

        pbx1 = max(
            0,
            int(px)
        )

        pby1 = max(
            0,
            int(py)
        )

        pbx2 = min(
            image.shape[1],
            int(px + pw)
        )

        pby2 = min(
            image.shape[0],
            int(py + ph)
        )

        if (
            pbx2 <= pbx1
            or pby2 <= pby1
        ):
            continue

        prescan_roi = image[
            pby1:pby2,
            pbx1:pbx2
        ].copy()

        if prescan_roi.size == 0:
            continue

        prescan_local_mask = None
        prescan_cnn_roi = prescan_roi.copy()

        prescan_full_mask = prescan_blob.get(
            "mask"
        )

        if prescan_full_mask is not None:

            prescan_candidate_mask = prescan_full_mask[
                pby1:pby2,
                pbx1:pbx2
            ]

            if (
                prescan_candidate_mask.shape[:2]
                == prescan_roi.shape[:2]
            ):
                prescan_local_mask = prescan_candidate_mask

                prescan_cnn_roi = cv.bitwise_and(
                    prescan_roi,
                    prescan_roi,
                    mask=prescan_local_mask
                )

        try:
            (
                prescan_type,
                prescan_conf,
                _
            ) = classify_fruit_type_cnn(
                prescan_cnn_roi
            )

        except Exception:
            continue

        prescan_contour = prescan_blob.get(
            "contour"
        )

        (
            prescan_aspect,
            prescan_circularity
        ) = get_shape_values(
            prescan_contour,
            prescan_roi.shape[1],
            prescan_roi.shape[0]
        )

        prescan_red_ratio = calculate_red_ratio(
            prescan_roi,
            prescan_local_mask
        )

        prescan_override = choose_cnn_override(
            0.0,
            prescan_type,
            prescan_conf,
            prescan_aspect,
            prescan_circularity,
            prescan_red_ratio
        )

        if prescan_override == "strawberry":

            verified_strawberry_blobs.append(
                {
                    "bbox": (
                        pbx1,
                        pby1,
                        pbx2,
                        pby2
                    ),
                    "confidence": prescan_conf,
                    "red_ratio": prescan_red_ratio,
                    "aspect": prescan_aspect,
                    "circularity": prescan_circularity
                }
            )


    # =================================================
    # PROCESS EACH YOLO FRUIT
    # =================================================

    for result in results:

        for box in result.boxes:

            # -----------------------------------------
            # YOLO result
            # -----------------------------------------

            cls_id = int(
                box.cls.item()
            )

            raw_yolo_type = model.names[
                cls_id
            ].lower()

            yolo_confidence = float(
                box.conf.item()
            )


            # -----------------------------------------
            # Class-specific confidence filtering
            # -----------------------------------------

            # IMPORTANT:
            # Do NOT reject a weak Apple/Banana box yet.
            #
            # YOLO was trained only on Apple / Banana / Orange.
            # A real Strawberry or Mango can therefore appear as a
            # LOW-confidence Apple/Banana. We still need the YOLO box
            # for localisation so the friend's CNN can inspect that ROI.
            #
            # The original Apple/Banana minimum confidence is applied
            # AFTER the CNN has had a chance to correct the fruit type.

            # IMPORTANT:
            # No extra Orange threshold.
            # A badly rotten orange may have lower confidence.


            # -----------------------------------------
            # Bounding box
            # -----------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Keep coordinates valid
            image_h, image_w = image.shape[:2]

            x1 = max(
                0,
                min(x1, image_w - 1)
            )

            y1 = max(
                0,
                min(y1, image_h - 1)
            )

            x2 = max(
                x1 + 1,
                min(x2, image_w)
            )

            y2 = max(
                y1 + 1,
                min(y2, image_h)
            )


            # =================================================
            # REMOVE DUPLICATE / FRAGMENT YOLO BOXES
            # =================================================

            current_width = x2 - x1
            current_height = y2 - y1

            current_area = (
                current_width
                * current_height
            )

            current_center_x = (
                x1 + x2
            ) // 2

            current_center_y = (
                y1 + y2
            ) // 2

            duplicate_fragment = False

            for (
                old_x1,
                old_y1,
                old_x2,
                old_y2,
                old_area
            ) in accepted_boxes:

                center_inside_old = (
                    old_x1
                    <= current_center_x
                    <= old_x2
                    and
                    old_y1
                    <= current_center_y
                    <= old_y2
                )

                much_smaller = (
                    current_area
                    < old_area * 0.35
                )

                if (
                    center_inside_old
                    and
                    much_smaller
                ):

                    duplicate_fragment = True
                    break

            if duplicate_fragment:
                continue


            # -----------------------------------------
            # Remove extremely tiny false boxes
            # -----------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            box_area = current_area

            image_area = (
                image_w
                * image_h
            )

            area_ratio = (
                box_area / image_area
                if image_area > 0
                else 0.0
            )


            if area_ratio < MIN_BOX_AREA_RATIO:
                continue


            # Store this box only after it passes
            # confidence, duplicate and tiny-box filtering.
            accepted_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    current_area
                )
            )


            # -----------------------------------------
            # Raw ROI
            # -----------------------------------------

            raw_roi = image[
                y1:y2,
                x1:x2
            ].copy()

            if raw_roi.size == 0:
                continue


            # =================================================
            # PREPROCESSING + SEGMENTATION + CALIBRATION
            # =================================================

            (
                cnn_roi,
                fruit_mask,
                fruit_contour
            ) = prepare_fruit_roi(
                raw_roi
            )


            # If segmentation completely failed,
            # keep CNN input as the raw crop.
            if cv.countNonZero(fruit_mask) == 0:

                cnn_roi = raw_roi.copy()


            # =================================================
            # FRUIT TYPE
            # =================================================

            fruit_type = raw_yolo_type

            confidence = yolo_confidence


            # =================================================
            # FRUIT TYPE PRIORITY
            # =================================================
            #
            # High-confidence YOLO Apple/Banana/Orange stays unchanged.
            #
            # If YOLO confidence is weaker, friend's CNN may correct
            # the fruit to Mango / Strawberry, but only if colour and
            # shape checks also agree.

            cnn_type = None
            cnn_conf = 0.0

            (
                aspect,
                circularity
            ) = get_shape_values(
                fruit_contour,
                raw_roi.shape[1],
                raw_roi.shape[0]
            )

            red_ratio = calculate_red_ratio(
                raw_roi,
                fruit_mask
            )

            # Always ask the friend's type CNN to verify the ROI.
            #
            # Reason:
            # YOLO has only Apple / Banana / Orange classes, so an
            # unseen Strawberry or Mango can still receive 99%+ YOLO
            # confidence as Apple/Orange/Banana.
            try:
                cnn_type, cnn_conf, _ = (
                    classify_fruit_type_cnn(
                        cnn_roi
                    )
                )

            except Exception as exc:
                print(
                    f"Fruit type CNN error: {exc}"
                )

                cnn_type = None
                cnn_conf = 0.0

            cnn_override = choose_cnn_override(
                yolo_confidence,
                cnn_type,
                cnn_conf,
                aspect,
                circularity,
                red_ratio
            )

            print(
                f"CNN raw type: {cnn_type} | "
                f"Conf: {cnn_conf * 100:.2f}% | "
                f"Red: {red_ratio * 100:.2f}%"
            )

            if cnn_override is not None:
                fruit_type = cnn_override
                confidence = cnn_conf

            # -------------------------------------------------
            # LARGE YOLO ORANGE -> STRAWBERRY CORRECTION
            # -------------------------------------------------
            #
            # If the whole crop is too mouldy for the friend's CNN to
            # recognise cleanly, use strong Strawberry fragments INSIDE
            # this YOLO box as supporting evidence.
            #
            # The box is NOT split. We keep one fruit / one box.

            if (
                raw_yolo_type == "orange"
                and verified_strawberry_blobs
            ):

                strawberry_evidence = []

                for evidence in verified_strawberry_blobs:

                    (
                        ex1,
                        ey1,
                        ex2,
                        ey2
                    ) = evidence[
                        "bbox"
                    ]

                    evidence_area = max(
                        1,
                        (ex2 - ex1)
                        * (ey2 - ey1)
                    )

                    inter_x1 = max(
                        x1,
                        ex1
                    )

                    inter_y1 = max(
                        y1,
                        ey1
                    )

                    inter_x2 = min(
                        x2,
                        ex2
                    )

                    inter_y2 = min(
                        y2,
                        ey2
                    )

                    inter_w = max(
                        0,
                        inter_x2 - inter_x1
                    )

                    inter_h = max(
                        0,
                        inter_y2 - inter_y1
                    )

                    contained_ratio = (
                        (inter_w * inter_h)
                        / evidence_area
                    )

                    if contained_ratio >= 0.80:

                        strawberry_evidence.append(
                            evidence
                        )

                # Two independent strong Strawberry fragments inside
                # one Orange YOLO box are strong evidence that this is
                # one damaged Strawberry split by segmentation.
                #
                # One very large fragment is also enough.
                strong_fragment_count = len(
                    strawberry_evidence
                )

                large_fragment = False

                for evidence in strawberry_evidence:

                    (
                        ex1,
                        ey1,
                        ex2,
                        ey2
                    ) = evidence[
                        "bbox"
                    ]

                    fragment_area = (
                        (ex2 - ex1)
                        * (ey2 - ey1)
                    )

                    if fragment_area >= current_area * 0.28:
                        large_fragment = True
                        break

                # -------------------------------------------------
                # LOW-CONFIDENCE / OVERSIZED YOLO ORANGE
                # -------------------------------------------------
                #
                # If YOLO is weak, e.g. 36%, do NOT keep its very large
                # Orange box.  Use the verified Strawberry evidence to
                # build a tighter ONE-fruit box instead.
                #
                # This is different from the earlier mouldy Strawberry
                # case where YOLO was ~91% and its large box already
                # localised the fruit reasonably well.
                if (
                    yolo_confidence < 0.55
                    and strong_fragment_count >= 1
                ):

                    print(
                        f"Tightening weak YOLO {raw_yolo_type} box "
                        f"({yolo_confidence * 100:.1f}%) using "
                        f"{strong_fragment_count} verified Strawberry blob(s)."
                    )

                    tight_x1 = min(
                        evidence["bbox"][0]
                        for evidence
                        in strawberry_evidence
                    )

                    tight_y1 = min(
                        evidence["bbox"][1]
                        for evidence
                        in strawberry_evidence
                    )

                    tight_x2 = max(
                        evidence["bbox"][2]
                        for evidence
                        in strawberry_evidence
                    )

                    tight_y2 = max(
                        evidence["bbox"][3]
                        for evidence
                        in strawberry_evidence
                    )

                    # Small padding so the Strawberry edge is not cut.
                    tight_w = max(
                        1,
                        tight_x2 - tight_x1
                    )

                    tight_h = max(
                        1,
                        tight_y2 - tight_y1
                    )

                    pad_x = max(
                        4,
                        int(tight_w * 0.05)
                    )

                    pad_y = max(
                        4,
                        int(tight_h * 0.05)
                    )

                    x1 = max(
                        0,
                        tight_x1 - pad_x
                    )

                    y1 = max(
                        0,
                        tight_y1 - pad_y
                    )

                    x2 = min(
                        image_w,
                        tight_x2 + pad_x
                    )

                    y2 = min(
                        image_h,
                        tight_y2 + pad_y
                    )

                    current_area = max(
                        1,
                        (x2 - x1)
                        * (y2 - y1)
                    )

                    # IMPORTANT:
                    # Rebuild the ROI using the corrected Strawberry box.
                    raw_roi = image[
                        y1:y2,
                        x1:x2
                    ].copy()

                    fruit_type = "strawberry"

                    confidence = max(
                        evidence["confidence"]
                        for evidence
                        in strawberry_evidence
                    )

                    print(
                        "Weak YOLO Orange replaced by tight "
                        "Strawberry localisation."
                    )

                elif (
                    strong_fragment_count >= 2
                    or large_fragment
                ):

                    fruit_type = "strawberry"

                    confidence = max(
                        evidence["confidence"]
                        for evidence
                        in strawberry_evidence
                    )

                    print(
                        "Large YOLO Orange corrected to "
                        "Strawberry using contained CNN evidence."
                    )

            # -------------------------------------------------
            # Apply original YOLO confidence filters ONLY if
            # the fruit remained Apple / Banana after CNN check.
            # -------------------------------------------------

            if (
                fruit_type == "apple"
                and yolo_confidence < APPLE_MIN_CONF
            ):
                print(
                    f"Rejected weak Apple after CNN check: "
                    f"{yolo_confidence * 100:.2f}%"
                )
                continue

            if (
                fruit_type == "banana"
                and yolo_confidence < BANANA_MIN_CONF
            ):
                print(
                    f"Rejected weak Banana after CNN check: "
                    f"{yolo_confidence * 100:.2f}%"
                )
                continue

            # Helpful debugging only when CNN thinks this is one of
            # the two fallback-only species.
            if cnn_type in {
                "Mango",
                "Strawberry"
            }:
                print(
                    f"CNN candidate: {cnn_type} "
                    f"({cnn_conf * 100:.2f}%)"
                )

                if cnn_type == "Strawberry":
                    print(
                        f"Strawberry red ratio: "
                        f"{red_ratio * 100:.2f}%"
                    )

                print(
                    f"Shape aspect: {aspect:.3f}"
                )

                print(
                    f"Shape circularity: "
                    f"{circularity if circularity is not None else 'N/A'}"
                )


            # =================================================
            # DEFECT DETECTION
            # =================================================

            output, defect_mask, defect_percentage = (
                detect_defect(
                    raw_roi,
                    fruit_type
                )
            )

            # Strawberry-only segmentation correction.
            # Apple/Banana/Orange/Mango remain unchanged.
            if fruit_type == "strawberry":

                (
                    output,
                    defect_mask,
                    defect_percentage
                ) = refine_strawberry_defect_result(
                    raw_roi,
                    defect_mask
                )


            # -----------------------------------------
            # IMPORTANT:
            # Do NOT recalculate defect percentage here.
            #
            # Each fruit-specific defect detector already
            # calculates the percentage using its own
            # fruit mask. Orange especially may recover
            # severe mould as part of the fruit surface.
            #
            # Recalculating with this external GrabCut
            # mask can make the denominator too small and
            # produce an incorrect very high percentage.
            # -----------------------------------------


            # =================================================
            # RIPENESS
            # =================================================

            ripeness_result = classify_ripeness(
                raw_roi,
                fruit_type,
                defect_percentage
            )

            ripeness = ripeness_result[
                "ripeness"
            ]


            detected = True
            fruit_count += 1

            # This fruit really passed all checks and was processed.
            confirmed_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    current_area,
                    fruit_type
                )
            )


            # =================================================
            # PRINT RESULT
            # =================================================

            print(
                f"Fruit #{fruit_count}: "
                f"{fruit_type}"
            )

            print(
                f"Raw YOLO type: "
                f"{raw_yolo_type}"
            )

            print(
                f"YOLO confidence: "
                f"{yolo_confidence * 100:.2f}%"
            )

            if fruit_type in {
                "mango",
                "strawberry"
            }:

                print(
                    f"CNN override confidence: "
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


            # =================================================
            # DISPLAY
            #
            # Keep original background visually,
            # but copy processed defect result ONLY
            # inside the segmented fruit.
            # =================================================

            # Use the fruit-specific detector output directly.
    # Do NOT clip the red defect contour using the external
    # GrabCut segmentation mask.
            display_roi = output.copy()


            display_image[
                y1:y2,
                x1:x2
            ] = display_roi


            # -----------------------------------------
            # Draw YOLO bounding box
            # -----------------------------------------

            cv.rectangle(
                display_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -----------------------------------------
            # Label
            # -----------------------------------------

            label = (
                f"{fruit_type.title()} | "
                f"{ripeness} | "
                f"Defect: {defect_percentage:.1f}% | "
                f"Conf: {confidence * 100:.1f}%"
            )

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


    # =================================================
    # PER-OBJECT CNN FALLBACK FOR MANGO / STRAWBERRY
    # =================================================
    #
    # This runs even when YOLO already detected other fruits.
    # Only segmented objects NOT sufficiently covered by YOLO
    # are sent to friend's fruit-type CNN.

    try:
        blobs = segment_all_objects(
            image
        )

    except Exception as exc:
        print(
            f"Object segmentation fallback error: {exc}"
        )

        blobs = []


    for blob in blobs:

        x, y, w, h = blob[
            "bbox"
        ]

        blob_x1 = max(
            0,
            int(x)
        )

        blob_y1 = max(
            0,
            int(y)
        )

        blob_x2 = min(
            image.shape[1],
            int(x + w)
        )

        blob_y2 = min(
            image.shape[0],
            int(y + h)
        )

        if (
            blob_x2 <= blob_x1
            or blob_y2 <= blob_y1
        ):
            continue

        blob_area = (
            (blob_x2 - blob_x1)
            * (blob_y2 - blob_y1)
        )

        if blob_area <= 0:
            continue


        # -----------------------------------------
        # Skip object already covered by YOLO
        # -----------------------------------------

        already_detected = False

        for (
            yolo_x1,
            yolo_y1,
            yolo_x2,
            yolo_y2,
            _,
            confirmed_type
        ) in confirmed_boxes:

            inter_x1 = max(
                blob_x1,
                yolo_x1
            )

            inter_y1 = max(
                blob_y1,
                yolo_y1
            )

            inter_x2 = min(
                blob_x2,
                yolo_x2
            )

            inter_y2 = min(
                blob_y2,
                yolo_y2
            )

            inter_w = max(
                0,
                inter_x2 - inter_x1
            )

            inter_h = max(
                0,
                inter_y2 - inter_y1
            )

            overlap_ratio = (
                (inter_w * inter_h)
                / blob_area
            )

            blob_center_x = (
                blob_x1 + blob_x2
            ) / 2.0

            blob_center_y = (
                blob_y1 + blob_y2
            ) / 2.0

            yolo_center_x = (
                yolo_x1 + yolo_x2
            ) / 2.0

            yolo_center_y = (
                yolo_y1 + yolo_y2
            ) / 2.0

            center_distance = np.hypot(
                blob_center_x - yolo_center_x,
                blob_center_y - yolo_center_y
            )

            blob_diag = max(
                1.0,
                np.hypot(
                    blob_x2 - blob_x1,
                    blob_y2 - blob_y1
                )
            )

            same_center = (
                center_distance
                <= blob_diag * 0.28
            )

            # Skip only when this blob is strongly overlapping AND
            # centred on the same already-processed fruit.
            #
            # This prevents one large YOLO box from blocking another
            # nearby Strawberry.
            if (
                (
                    confirmed_type == "strawberry"
                    and overlap_ratio >= 0.75
                )
                or (
                    overlap_ratio >= 0.65
                    and same_center
                )
            ):
                already_detected = True
                break

        if already_detected:
            continue


        # -----------------------------------------
        # Object ROI
        # -----------------------------------------

        raw_roi = image[
            blob_y1:blob_y2,
            blob_x1:blob_x2
        ].copy()

        if raw_roi.size == 0:
            continue


        # -----------------------------------------
        # Isolated input for friend's CNN
        # -----------------------------------------

        cnn_roi = raw_roi.copy()
        local_mask = None

        full_mask = blob.get(
            "mask"
        )

        if full_mask is not None:

            candidate_mask = full_mask[
                blob_y1:blob_y2,
                blob_x1:blob_x2
            ]

            if (
                candidate_mask.shape[:2]
                == raw_roi.shape[:2]
            ):
                local_mask = candidate_mask

                cnn_roi = cv.bitwise_and(
                    raw_roi,
                    raw_roi,
                    mask=local_mask
                )


        try:
            cnn_type, cnn_conf, _ = (
                classify_fruit_type_cnn(
                    cnn_roi
                )
            )

        except Exception as exc:
            print(
                f"Fruit type CNN fallback error: {exc}"
            )

            continue


        fallback_contour = blob.get(
            "contour"
        )

        (
            aspect,
            circularity
        ) = get_shape_values(
            fallback_contour,
            raw_roi.shape[1],
            raw_roi.shape[0]
        )

        red_ratio = calculate_red_ratio(
            raw_roi,
            local_mask
        )


        # -----------------------------------------
        # Accept Mango / Strawberry only
        # -----------------------------------------

        fruit_type = None

        strawberry_normal_ok = (
            cnn_type == "Strawberry"
            and cnn_conf >= STRAWBERRY_MIN_CONF
            and red_ratio >= STRAWBERRY_MIN_RED_RATIO
            and (
                circularity is None
                or circularity < 0.78
            )
        )

        strawberry_strong_red_ok = (
            cnn_type == "Strawberry"
            and cnn_conf >= STRAWBERRY_STRONG_RED_MIN_CONF
            and red_ratio >= STRAWBERRY_STRONG_RED_RATIO
            and (
                circularity is None
                or circularity < 0.78
            )
        )

        strawberry_green_ok = (
            cnn_type == "Strawberry"
            and cnn_conf >= STRAWBERRY_GREEN_MIN_CONF
            and red_ratio <= STRAWBERRY_GREEN_MAX_RED_RATIO
            and aspect >= STRAWBERRY_GREEN_MIN_ASPECT
            and circularity is not None
            and circularity < STRAWBERRY_GREEN_MAX_CIRCULARITY
        )

        strawberry_damaged_ok = (
            cnn_type == "Strawberry"
            and cnn_conf >= STRAWBERRY_DAMAGED_MIN_CONF
            and red_ratio >= STRAWBERRY_DAMAGED_MIN_RED_RATIO
            and red_ratio < STRAWBERRY_DAMAGED_MAX_RED_RATIO
            and aspect >= STRAWBERRY_DAMAGED_MIN_ASPECT
            and circularity is not None
            and circularity < STRAWBERRY_DAMAGED_MAX_CIRCULARITY
        )

        if (
            strawberry_normal_ok
            or strawberry_strong_red_ok
            or strawberry_green_ok
            or strawberry_damaged_ok
        ):
            fruit_type = "strawberry"

        elif (
            cnn_type == "Mango"
            and cnn_conf >= MANGO_MIN_CONF
            and aspect >= MANGO_MIN_ASPECT
            and (
                circularity is None
                or circularity < MANGO_MAX_CIRCULARITY
            )
        ):
            fruit_type = "mango"

        else:
            continue


        confidence = cnn_conf


        # =================================================
        # DEFECT DETECTION
        # =================================================

        output, defect_mask, defect_percentage = (
            detect_defect(
                raw_roi,
                fruit_type
            )
        )

        if fruit_type == "strawberry":

            (
                output,
                defect_mask,
                defect_percentage
            ) = refine_strawberry_defect_result(
                raw_roi,
                defect_mask
            )


        # =================================================
        # RIPENESS
        # =================================================

        ripeness_result = classify_ripeness(
            raw_roi,
            fruit_type,
            defect_percentage
        )

        ripeness = ripeness_result[
            "ripeness"
        ]


        detected = True
        fruit_count += 1


        print(
            f"Fruit #{fruit_count}: "
            f"{fruit_type}"
        )

        print(
            f"CNN fallback confidence: "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Shape aspect: "
            f"{aspect:.3f}"
        )

        print(
            f"Shape circularity: "
            f"{circularity if circularity is not None else 'N/A'}"
        )

        if fruit_type == "strawberry":
            print(
                f"Red ratio: "
                f"{red_ratio * 100:.2f}%"
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
        # Put result into main image
        # -----------------------------------------

        display_image[
            blob_y1:blob_y2,
            blob_x1:blob_x2
        ] = output


        cv.rectangle(
            display_image,
            (blob_x1, blob_y1),
            (blob_x2, blob_y2),
            (0, 255, 0),
            2
        )


        label = (
            f"{fruit_type.title()} | "
            f"{ripeness} | "
            f"Defect: {defect_percentage:.1f}% | "
            f"CNN: {confidence * 100:.1f}%"
        )

        text_y = max(
            blob_y1 - 10,
            25
        )

        cv.putText(
            display_image,
            label,
            (blob_x1, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv.LINE_AA
        )


    if not detected:

        try:

            fallback_cnn_roi, fallback_mask, fallback_contour = (
                prepare_fruit_roi(
                    image
                )
            )

            if cv.countNonZero(
                fallback_mask
            ) == 0:

                fallback_cnn_roi = image.copy()

            fallback_type, fallback_conf, _ = (
                classify_fruit_type_cnn(
                    fallback_cnn_roi
                )
            )

            fallback_aspect = None
            fallback_circularity = None

            if fallback_contour is not None:

                try:

                    (
                        _,
                        fallback_aspect,
                        fallback_circularity
                    ) = contour_shape_metrics(
                        fallback_contour
                    )

                except Exception:

                    fallback_aspect = None
                    fallback_circularity = None

            fallback_fruit_type = None

            fallback_red_ratio = calculate_red_ratio(
                image,
                fallback_mask
            )

            fallback_aspect_safe = (
                max(
                    float(fallback_aspect),
                    1.0 / max(float(fallback_aspect), 1e-6)
                )
                if fallback_aspect is not None
                else 1.0
            )

            fallback_strawberry_normal = (
                fallback_type == "Strawberry"
                and fallback_conf >= STRAWBERRY_MIN_CONF
                and fallback_red_ratio >= STRAWBERRY_MIN_RED_RATIO
            )

            fallback_strawberry_green = (
                fallback_type == "Strawberry"
                and fallback_conf >= STRAWBERRY_GREEN_MIN_CONF
                and fallback_red_ratio <= STRAWBERRY_GREEN_MAX_RED_RATIO
                and fallback_aspect_safe >= STRAWBERRY_GREEN_MIN_ASPECT
                and fallback_circularity is not None
                and fallback_circularity < STRAWBERRY_GREEN_MAX_CIRCULARITY
            )

            fallback_strawberry_damaged = (
                fallback_type == "Strawberry"
                and fallback_conf >= STRAWBERRY_DAMAGED_MIN_CONF
                and fallback_red_ratio >= STRAWBERRY_DAMAGED_MIN_RED_RATIO
                and fallback_red_ratio < STRAWBERRY_DAMAGED_MAX_RED_RATIO
                and fallback_aspect_safe >= STRAWBERRY_DAMAGED_MIN_ASPECT
                and fallback_circularity is not None
                and fallback_circularity < STRAWBERRY_DAMAGED_MAX_CIRCULARITY
            )

            if (
                fallback_strawberry_normal
                or fallback_strawberry_green
                or fallback_strawberry_damaged
            ):

                fallback_fruit_type = "strawberry"

            elif (
                fallback_type == "Mango"
                and fallback_conf >= MANGO_MIN_CONF
                and fallback_aspect is not None
                and fallback_aspect >= MANGO_MIN_ASPECT
            ):

                fallback_fruit_type = "mango"


            if fallback_fruit_type is not None:

                output, defect_mask, defect_percentage = (
                    detect_defect(
                        image,
                        fallback_fruit_type
                    )
                )

                if fallback_fruit_type == "strawberry":

                    (
                        output,
                        defect_mask,
                        defect_percentage
                    ) = refine_strawberry_defect_result(
                        image,
                        defect_mask
                    )

                ripeness_result = classify_ripeness(
                    image,
                    fallback_fruit_type,
                    defect_percentage
                )

                ripeness = ripeness_result[
                    "ripeness"
                ]

                detected = True
                fruit_count = 1

                print(
                    f"Fruit #1: "
                    f"{fallback_fruit_type}"
                )

                print(
                    f"CNN fallback confidence: "
                    f"{fallback_conf * 100:.2f}%"
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

                display_image = output.copy()

                label = (
                    f"{fallback_fruit_type.title()} | "
                    f"{ripeness} | "
                    f"Defect: {defect_percentage:.1f}% | "
                    f"CNN: {fallback_conf * 100:.1f}%"
                )

                cv.putText(
                    display_image,
                    label,
                    (20, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA
                )

        except Exception as exc:

            print(
                f"CNN fallback error: {exc}"
            )


    if not detected:

        print(
            "No fruit detected."
        )

    return {
        "original": image,
        "annotated": display_image,
        "detected": bool(detected),
        "count": int(fruit_count),
    }


def parse_exact_defect_console(log_text):
    """Parse display metrics only; does not affect detection logic."""

    objects = []
    current = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()

        fruit_match = re.match(
            r"Fruit #(\d+):\s*(.+)",
            line,
            re.IGNORECASE,
        )

        if fruit_match:
            if current is not None:
                objects.append(current)

            current = {
                "index": int(fruit_match.group(1)) - 1,
                "fruit_type": fruit_match.group(2).strip().title(),
                "raw_yolo_type": None,
                "yolo_confidence": None,
                "cnn_confidence": None,
                "defect_percentage": None,
                "ripeness": None,
            }
            continue

        if current is None:
            continue

        lower = line.lower()

        if lower.startswith("raw yolo type:"):
            current["raw_yolo_type"] = (
                line.split(":", 1)[1]
                .strip()
                .title()
            )

        elif lower.startswith("yolo confidence:"):
            value = (
                line.split(":", 1)[1]
                .strip()
                .rstrip("%")
            )
            try:
                current["yolo_confidence"] = float(value)
            except ValueError:
                pass

        elif (
            lower.startswith("cnn override confidence:")
            or lower.startswith("cnn fallback confidence:")
        ):
            value = (
                line.split(":", 1)[1]
                .strip()
                .rstrip("%")
            )
            try:
                current["cnn_confidence"] = float(value)
            except ValueError:
                pass

        elif lower.startswith("defect percentage:"):
            value = (
                line.split(":", 1)[1]
                .strip()
                .rstrip("%")
            )
            try:
                current["defect_percentage"] = float(value)
            except ValueError:
                pass

        elif lower.startswith("ripeness:"):
            current["ripeness"] = (
                line.split(":", 1)[1]
                .strip()
            )

    if current is not None:
        objects.append(current)

    return objects


try:
    import ultralytics  # noqa: F401  (only used to check availability up front)
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Fruit Quality Inspection Dashboard", layout="wide")

DEFAULT_ERODE_PIXELS = 10
DEFAULT_YOLO_CONFIDENCE = 0.25
# Only used by the Stem Detection section's own ArUco-marker calibration —
# the main pipeline's calibration no longer uses marker size (manual
# cm-per-pixel only).
DEFAULT_MARKER_SIZE_CM = 5.0

DENOISE_METHOD = "median"
ENHANCE_METHOD = "clahe"


# ======================================================
# Helpers
# ======================================================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload_to_bgr(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def infer_freshness_from_filename(filename):
    """Best-effort Fresh/Rotten label from filename (e.g. "FreshApple (12).jpg")."""
    lower = filename.lower()
    if "fresh" in lower:
        return "Fresh"
    if "rotten" in lower or "spoiled" in lower or "stale" in lower:
        return "Rotten"
    return None


def stem_quality_tier(confidence):
    """Translate a raw confidence score into a plain-language tier."""
    if not confidence or confidence <= 0:
        return "No Reliable Stem Detected"
    if confidence >= 0.5:
        return "High Confidence"
    return "Review Recommended"


def summary_to_text(summary):
    parts = []
    for fruit_type, qualities in summary.items():
        quality_text = ", ".join(f"{n} {q}" for q, n in qualities.items())
        parts.append(f"{fruit_type}: {quality_text}")
    return "; ".join(parts) if parts else "No fruit detected"


# ======================================================
# Sidebar — minimal configuration
# ======================================================
st.sidebar.title("Inspection Settings")

IMPLEMENTED_TECHNIQUE = "Colour Feature Extraction"
TECHNIQUE_OPTIONS = [
    IMPLEMENTED_TECHNIQUE,
    "Morphological and Texture Feature Extraction",
    "Defect Detection",
    "Stem Detection",
    "All of the above",
]
selected_technique = st.sidebar.selectbox(
    "Technique / method (report section)",
    TECHNIQUE_OPTIONS,
    index=0,
    help="Matches the report's methodology sections. Only Colour Feature "
         "Extraction is implemented in this app (LAB chroma-distance "
         "segmentation + YOLO + CNN) — the others are placeholders for "
         "teammates' modules and don't change what actually runs below.",
)
if selected_technique not in (
    IMPLEMENTED_TECHNIQUE,
    "Stem Detection",
    "Defect Detection",
):
    st.sidebar.caption(
        f"ℹ️ “{selected_technique}” isn't implemented in this module yet — "
        f"running the same {IMPLEMENTED_TECHNIQUE} pipeline (LAB + YOLO + CNN) below."
    )

if selected_technique == "Defect Detection":
    if CUSTOM_MODULES_AVAILABLE:
        st.sidebar.success(
            "Exact latest defect pipeline loaded"
        )
    else:
        st.sidebar.error(
            "Defect detection module unavailable"
        )

        if CUSTOM_MODULE_ERROR:
            st.sidebar.caption(
                CUSTOM_MODULE_ERROR
            )

# ======================================================
# Stem Detection — self-contained section, runs instead of the Colour
# Feature Extraction pipeline below.
# ======================================================
if selected_technique == "Stem Detection":
    from stem_detection import calibration as stem_calib
    from stem_detection import metrics as stem_metrics
    from stem_detection import preprocessing as stem_pp
    from stem_detection import report as stem_report
    from stem_detection.detector import StemDetector

    st.sidebar.subheader("Stem Detection settings")
    stem_fruit_type = "Fruit"
    stem_method = "Automatic"

    # Prefer the locally trained V4 weights, fall back to V3 then models/best.pt.
    _stem_model_candidates = [
        "runs/detect/fruit_stem_detector_v4/weights/best.pt",
        "runs/detect/fruit_stem_detector_v3/weights/best.pt",
        "models/best.pt",
    ]
    stem_model_path = next((p for p in _stem_model_candidates if os.path.isfile(p)), _stem_model_candidates[0])
    stem_yolo_confidence = 0.10
    if not YOLO_AVAILABLE:
        st.sidebar.warning(
            "YOLO is unavailable, so automatic stem detection will use the "
            "traditional image-processing fallback only."
        )

    @st.cache_resource
    def _get_stem_detector(model_path: str) -> StemDetector:
        return StemDetector(model_path=model_path)

    stem_detector = _get_stem_detector(stem_model_path)
    if YOLO_AVAILABLE and not stem_detector.yolo_ready:
        st.sidebar.warning(
            f"No YOLO stem model found at `{stem_model_path}`. Automatic mode will "
            "continue with the traditional image-processing fallback."
        )

    stem_denoise_method = "median"
    stem_enhance_method = "clahe"

    st.sidebar.divider()
    stem_want_calibration = st.sidebar.checkbox("Measure physical size (cm)", value=False, key="stem_want_calib")
    stem_calib_mode = None
    stem_marker_cm = DEFAULT_MARKER_SIZE_CM
    stem_manual_ratio = None
    if stem_want_calibration:
        stem_calib_mode = st.sidebar.radio(
            "How is scale determined?",
            ["Auto-detect ArUco marker in photo", "I know my cm-per-pixel ratio"],
            key="stem_calib_mode",
        )
        if stem_calib_mode.startswith("Auto"):
            stem_marker_cm = st.sidebar.number_input(
                "Marker side length (cm)", value=DEFAULT_MARKER_SIZE_CM, min_value=0.1, step=0.5, key="stem_marker_cm",
            )
        else:
            stem_manual_ratio = st.sidebar.number_input(
                "cm per pixel", value=0.02, min_value=0.0001, step=0.001, format="%.4f", key="stem_manual_ratio",
            )

    def _get_stem_calibration(image):
        if not stem_want_calibration:
            return stem_calib.uncalibrated()
        if stem_calib_mode.startswith("Auto"):
            return stem_calib.calibrate(image, marker_size_cm=stem_marker_cm)
        return stem_calib.manual_scale(stem_manual_ratio)

    st.title("Stem Detection")
    st.caption(
        "Automatically localises apple, banana and orange stems/crowns/calyxes using "
        "a combined YOLO + classical image-processing pipeline."
    )

    stem_tab_images, stem_tab_benchmark = st.tabs(
        ["Image inspection", "Method benchmark"]
    )

    # --- Image inspection ---
    with stem_tab_images:
        stem_uploaded_files = st.file_uploader(
            "Select one or more images", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True, key="stem_image_uploader",
        )
        stem_images_to_process = []
        if stem_uploaded_files:
            for f in stem_uploaded_files:
                img = read_upload_to_bgr(f)
                if img is not None:
                    stem_images_to_process.append((f.name, img))

        # Automatic mode can still run when the YOLO model is unavailable,
        # because detector.py contains a traditional fallback.
        stem_run_disabled = len(stem_images_to_process) == 0
        stem_run_button = st.button(
            "Run detection", type="primary", disabled=stem_run_disabled, key="stem_run_images",
        )

        if "stem_image_results" not in st.session_state:
            st.session_state["stem_image_results"] = []

        if stem_run_button:
            stem_results = []
            stem_progress = st.progress(0.0, text="Running detection...")
            for i, (name, img) in enumerate(stem_images_to_process):
                processed = stem_pp.preprocess(img, stem_denoise_method, stem_enhance_method)
                calibration_result = _get_stem_calibration(img)
                detections, elapsed, method_used = stem_detector.detect(
                    processed, stem_fruit_type, stem_method, stem_yolo_confidence, skip_preprocess=True,
                )
                annotated = stem_detector.annotate(processed, detections)
                if calibration_result.marker_corners is not None:
                    annotated = stem_calib.draw_marker_overlay(annotated, calibration_result)
                stem_results.append({
                    "filename": name, "original": img, "processed": processed, "annotated": annotated,
                    "detections": detections, "fruit_type": stem_fruit_type, "method": stem_method,
                    "method_used": method_used,
                    "candidate_mask": stem_detector.traditional_mask(processed),
                    "calibration": calibration_result, "processing_ms": elapsed * 1000,
                })
                stem_progress.progress((i + 1) / len(stem_images_to_process), text=f"Processed {name}")
            stem_progress.empty()
            st.session_state["stem_image_results"] = stem_results

        stem_image_results = st.session_state["stem_image_results"]

        if stem_image_results:
            st.divider()
            st.header("Summary")
            stem_total = sum(len(r["detections"]) for r in stem_image_results)
            stem_found_in = sum(1 for r in stem_image_results if r["detections"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Photos inspected", len(stem_image_results))
            c2.metric("Stems detected", stem_total)
            c3.metric("Photos with a stem found", f"{stem_found_in}/{len(stem_image_results)}")
            c4.metric("Avg. processing time", f"{np.mean([r['processing_ms'] for r in stem_image_results]):.0f} ms")

            stem_table_rows = []
            for r in stem_image_results:
                best = max(r["detections"], key=lambda d: d.confidence) if r["detections"] else None
                size_cm = "-"
                if best is not None and r["calibration"].is_calibrated:
                    x1, y1, x2, y2 = best.bbox
                    w_cm = r["calibration"].px_to_cm(x2 - x1)
                    h_cm = r["calibration"].px_to_cm(y2 - y1)
                    size_cm = f"{w_cm:.1f} x {h_cm:.1f}"
                stem_table_rows.append({
                    "Image": r["filename"], "Method": r["method"],
                    "Detected via": r["method_used"].capitalize(),
                    "Stems found": len(r["detections"]),
                    "Best confidence": f"{best.confidence:.0%}" if best else "-",
                    "Quality": stem_quality_tier(best.confidence if best else None),
                    "Size (cm)": size_cm, "Time (ms)": f"{r['processing_ms']:.0f}",
                })
            stem_df = pd.DataFrame(stem_table_rows)
            st.dataframe(stem_df, use_container_width=True)

            stem_csv_bytes = stem_df.to_csv(index=False).encode("utf-8")
            stem_dl1, stem_dl2 = st.columns(2)
            stem_dl1.download_button("Download results as CSV", stem_csv_bytes, "stem_detection_results.csv", "text/csv")

            if stem_dl2.button("Generate PDF report", key="stem_gen_pdf"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    stem_report.generate_report(stem_image_results, output_path=tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.session_state["stem_pdf_bytes"] = f.read()
                os.unlink(tmp.name)
            if "stem_pdf_bytes" in st.session_state:
                stem_dl2.download_button(
                    "Download PDF report", st.session_state["stem_pdf_bytes"],
                    "stem_detection_report.pdf", "application/pdf",
                )

            stem_freshness_rows = []
            for r in stem_image_results:
                freshness = infer_freshness_from_filename(r["filename"])
                if freshness is None:
                    continue
                best = max(r["detections"], key=lambda d: d.confidence) if r["detections"] else None
                stem_freshness_rows.append({
                    "freshness": freshness,
                    "stem_detected": bool(r["detections"]),
                    "best_confidence": best.confidence if best else 0.0,
                })

            if stem_freshness_rows:
                st.divider()
                st.header("Stem visibility vs. freshness")
                st.caption("Freshness inferred from filename (e.g. \"FreshApple\"). Images without a hint are excluded.")
                stem_fresh_df = pd.DataFrame(stem_freshness_rows)
                stem_fresh_summary = stem_fresh_df.groupby("freshness").agg(
                    stem_detection_rate=("stem_detected", "mean"),
                    mean_confidence=("best_confidence", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
                    photos=("stem_detected", "size"),
                ).reset_index()
                st.dataframe(stem_fresh_summary, use_container_width=True)

                fresh_chart1, fresh_chart2 = st.columns(2)
                fresh_chart1.caption("Stem detection rate by freshness")
                fresh_chart1.bar_chart(stem_fresh_summary.set_index("freshness")["stem_detection_rate"])
                fresh_chart2.caption("Mean stem confidence by freshness")
                fresh_chart2.bar_chart(stem_fresh_summary.set_index("freshness")["mean_confidence"])

                fresh_row = stem_fresh_summary[stem_fresh_summary["freshness"] == "Fresh"]
                rotten_row = stem_fresh_summary[stem_fresh_summary["freshness"] == "Rotten"]
                if not fresh_row.empty and not rotten_row.empty:
                    rate_diff = fresh_row["stem_detection_rate"].iloc[0] - rotten_row["stem_detection_rate"].iloc[0]
                    if abs(rate_diff) >= 0.1:
                        direction = "higher" if rate_diff > 0 else "lower"
                        st.info(f"Stem detection rate was {abs(rate_diff):.0%} {direction} on Fresh vs Rotten photos.")
                    else:
                        st.info("Stem detection rate was similar between Fresh and Rotten photos.")

            st.divider()
            st.header("Per-image detail")
            for r in stem_image_results:
                title = f"{r['filename']} - {len(r['detections'])} stem(s) found" if r["detections"] \
                    else f"{r['filename']} - No visible stem detected"
                with st.expander(title):
                    best = max(r["detections"], key=lambda d: d.confidence) if r["detections"] else None
                    quality = stem_quality_tier(best.confidence if best else None)
                    if quality == "High Confidence":
                        st.success(f"{quality} — detected via {r['method_used'].capitalize()}")
                    elif quality == "Review Recommended":
                        st.warning(f"{quality} — detected via {r['method_used'].capitalize()}")
                    else:
                        st.error(quality)

                    ic1, ic2 = st.columns(2)
                    ic1.image(bgr_to_rgb(r["original"]), caption="Original", use_container_width=True)
                    ic2.image(bgr_to_rgb(r["annotated"]), caption="Stem detection (bbox + contour)",
                               use_container_width=True)
                    if r["calibration"].is_calibrated:
                        st.caption(f"Calibration: {r['calibration'].method} - {r['calibration'].confidence}")

                    if st.toggle("View Detection Details (pipeline steps)", key=f"stem_pipeline_{r['filename']}"):
                        pc1, pc2, pc3, pc4 = st.columns(4)
                        pc1.image(bgr_to_rgb(r["original"]), caption="1. Original", use_container_width=True)
                        pc2.image(bgr_to_rgb(r["processed"]), caption="2. Preprocessed (denoise + enhance)",
                                   use_container_width=True)
                        pc3.image(r["candidate_mask"], caption="3. Candidate regions (stem-colour mask)",
                                   use_container_width=True)
                        pc4.image(bgr_to_rgb(r["annotated"]), caption="4. Final detection",
                                   use_container_width=True)
        else:
            st.info("Upload one or more images, then click **Run detection**.")

    # --- Method benchmark (Mode A comparative evaluation) ---
    with stem_tab_benchmark:
        st.write(
            "Compares three stem-detection techniques on the same images — Traditional "
            "(classical image processing), YOLO (deep learning), and Hybrid (both combined) "
            "— to show which one performs best."
        )

        stem_bench_images = [(r["filename"], "Fruit", r["original"]) for r in stem_image_results]

        if not stem_bench_images:
            st.info("Upload and run images in **Image inspection** first, then come back here to benchmark them.")
        else:
            if st.button("Run benchmark", type="primary", key="stem_run_bench"):
                with st.spinner("Running all methods..."):
                    stem_bench_df = stem_metrics.run_benchmark(
                        stem_bench_images, methods=("Traditional", "YOLO", "Hybrid"),
                        detector=stem_detector, yolo_confidence=stem_yolo_confidence,
                    )
                    stem_bench_visuals = []
                    for stem_bv_name, stem_bv_fruit, stem_bv_img in stem_bench_images:
                        stem_bv_trad, _, _ = stem_detector.detect_traditional(stem_bv_img, stem_bv_fruit)
                        stem_bv_yolo, _ = stem_detector.detect_yolo(stem_bv_img, stem_bv_fruit, stem_yolo_confidence)
                        stem_bv_hybrid, _ = stem_detector.detect_hybrid(stem_bv_img, stem_bv_fruit, stem_yolo_confidence)
                        stem_bench_visuals.append({
                            "filename": stem_bv_name,
                            "Traditional": stem_detector.annotate(stem_bv_img, stem_bv_trad),
                            "YOLO": stem_detector.annotate(stem_bv_img, stem_bv_yolo),
                            "Hybrid": stem_detector.annotate(stem_bv_img, stem_bv_hybrid),
                        })
                st.session_state["stem_bench_df"] = stem_bench_df
                st.session_state["stem_bench_summary"] = stem_metrics.summarize(stem_bench_df)
                st.session_state["stem_bench_visuals"] = stem_bench_visuals

            if "stem_bench_summary" in st.session_state:
                st.subheader("Method comparison")
                stem_summary = st.session_state["stem_bench_summary"]
                st.dataframe(stem_summary, use_container_width=True)

                stem_chart1, stem_chart2 = st.columns(2)
                stem_chart1.caption("Detection rate by method")
                stem_chart1.bar_chart(stem_summary.set_index("method")["detection_rate"])
                stem_chart2.caption("Mean processing time (ms) by method")
                stem_chart2.bar_chart(stem_summary.set_index("method")["mean_processing_ms"])

                stem_top_rate = stem_summary["detection_rate"].max()
                stem_tied_top = stem_summary[stem_summary["detection_rate"] == stem_top_rate]
                stem_best_method = stem_tied_top.loc[stem_tied_top["mean_processing_ms"].idxmin(), "method"]
                st.success(f"Highest detection rate on this set: **{stem_best_method}**")

                stem_analysis_text = stem_metrics.generate_analysis_text(stem_summary)
                st.subheader("Analysis")
                st.markdown(stem_analysis_text)

                st.subheader("Visual comparison")
                for stem_bv in st.session_state.get("stem_bench_visuals", []):
                    with st.expander(stem_bv["filename"]):
                        stem_bv_c1, stem_bv_c2, stem_bv_c3 = st.columns(3)
                        stem_bv_c1.image(bgr_to_rgb(stem_bv["Traditional"]), caption="Traditional",
                                          use_container_width=True)
                        stem_bv_c2.image(bgr_to_rgb(stem_bv["YOLO"]), caption="YOLO", use_container_width=True)
                        stem_bv_c3.image(bgr_to_rgb(stem_bv["Hybrid"]), caption="Hybrid", use_container_width=True)

                with st.expander("Per-image, per-method detail"):
                    st.dataframe(
                        st.session_state["stem_bench_df"].drop(columns=["fruit_type"]),
                        use_container_width=True,
                    )

                if st.button("Generate benchmark PDF report", key="stem_bench_pdf_btn"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        stem_report.generate_report(
                            [], output_path=tmp.name, benchmark_summary=stem_summary,
                            title="Stem Detection Method Benchmark", analysis_text=stem_analysis_text,
                        )
                        with open(tmp.name, "rb") as f:
                            stem_bench_pdf_bytes = f.read()
                    os.unlink(tmp.name)
                    st.download_button(
                        "Download benchmark PDF", stem_bench_pdf_bytes,
                        "stem_detection_benchmark.pdf", "application/pdf", key="stem_bench_pdf_dl",
                    )

    st.stop()

st.sidebar.divider()

if not YOLO_AVAILABLE:
    st.sidebar.error(
        "ultralytics (YOLO) isn't installed — detection can't run. "
        "Run `pip install ultralytics` and restart the app."
    )

if not os.path.isdir("color_knn_models"):
    st.sidebar.warning(
        "No color_knn_models/*.joblib found. Run `python train_color_knn.py` to train the colour-feature "
        "KNN model per fruit type — until then, Quality will show as unavailable."
    )

st.sidebar.divider()
want_measurements = st.sidebar.checkbox("Measure physical size (cm)", value=False)
manual_cm_per_pixel = None
if want_measurements:
    manual_cm_per_pixel = st.sidebar.number_input(
        "cm per pixel", value=0.02, min_value=0.0001, step=0.001, format="%.4f",
        help="Known scale for your camera setup (e.g. derived once from a ruler photo).",
    )

with st.sidebar.expander("Perspective rectification (optional)"):
    st.caption(
        "Straightens the image plane before measurement if the camera isn't "
        "perfectly perpendicular to the inspection surface."
    )
    want_rectify = st.checkbox("Enable perspective rectification", value=False)
    rectify_mode = "Auto-detect (largest rectangle in photo)"
    rectify_points = None
    if want_rectify:
        rectify_mode = st.radio("How to find the 4 corners?",
                                 ["Auto-detect (largest rectangle in photo)", "Enter coordinates manually"])
        if rectify_mode.startswith("Auto"):
            st.caption(
                "Looks for the largest 4-sided shape in each photo (e.g. a tray, "
                "card, or table edge) and uses it as the reference. Less reliable "
                "than a coded marker — no unique identity to lock onto, so it can "
                "pick the wrong shape in a cluttered frame. Falls back to no "
                "rectification for a photo if nothing suitable is found."
            )
        else:
            st.caption(
                "Pixel coordinates of the 4 corners of a flat reference region "
                "(e.g. the tray/table edges), any order."
            )
            rc1, rc2 = st.columns(2)
            x1 = rc1.number_input("Corner 1 — x", value=0, step=1)
            y1 = rc2.number_input("Corner 1 — y", value=0, step=1)
            x2 = rc1.number_input("Corner 2 — x", value=100, step=1)
            y2 = rc2.number_input("Corner 2 — y", value=0, step=1)
            x3 = rc1.number_input("Corner 3 — x", value=100, step=1)
            y3 = rc2.number_input("Corner 3 — y", value=100, step=1)
            x4 = rc1.number_input("Corner 4 — x", value=0, step=1)
            y4 = rc2.number_input("Corner 4 — y", value=100, step=1)
            rectify_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

with st.sidebar.expander("Advanced settings"):
    erode_pixels = st.slider("Mask erosion (px)", 0, 30, DEFAULT_ERODE_PIXELS,
                              help="Trims mixed-color boundary patches between fruit and background.")
    yolo_confidence = st.slider("YOLO confidence threshold", 0.05, 0.9, DEFAULT_YOLO_CONFIDENCE, step=0.05,
                                 help="Lower catches more heavily-occluded fruit at the cost of more false positives.")


def get_calibration(image):
    if not want_measurements:
        return calib.uncalibrated()
    return calib.manual_scale(manual_cm_per_pixel)


# ======================================================
# Main — image ingestion
# ======================================================
st.title("Fruit Quality Inspection Dashboard")
st.caption("Detects every fruit in each photo, classifies its type and quality, and reports the results.")

images_to_process = []  # list of (name, bgr_image)

uploaded_files = st.file_uploader(
    "Select one or more images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True
)
if uploaded_files:
    for f in uploaded_files:
        img = read_upload_to_bgr(f)
        if img is not None:
            images_to_process.append((f.name, img))

run_button = st.button(
    "Run Inspection", type="primary",
    disabled=(len(images_to_process) == 0 or not YOLO_AVAILABLE),
)
if not YOLO_AVAILABLE and images_to_process:
    st.warning("Can't run — ultralytics (YOLO) isn't installed. See the sidebar for install instructions.")

if "results" not in st.session_state:
    st.session_state["results"] = []

if run_button:
    results = []
    progress = st.progress(0.0, text="Running pipeline...")
    for i, (name, img) in enumerate(images_to_process):

        # ==================================================
        # TIFANY — EXACT LATEST DEFECT TEST
        # ==================================================
        if selected_technique == "Defect Detection":

            try:
                console_buffer = io.StringIO()

                with contextlib.redirect_stdout(
                    console_buffer
                ):
                    latest_result = (
                        run_exact_latest_defect_pipeline(
                            img
                        )
                    )

                console_log = (
                    console_buffer.getvalue()
                )

                parsed_objects = (
                    parse_exact_defect_console(
                        console_log
                    )
                )

                out = {
                    "original": latest_result["original"],
                    "annotated": latest_result["annotated"],
                    "objects": parsed_objects,
                    "count": latest_result["count"],
                    "summary": {},
                    "calibration_method": "not used",
                    "calibration_confidence": "not used",
                    "exact_defect_console": console_log,
                }

            except Exception as exc:
                st.error(
                    f"Defect Detection failed: {exc}"
                )

                out = {
                    "original": img,
                    "annotated": img.copy(),
                    "objects": [],
                    "count": 0,
                    "summary": {},
                    "calibration_method": "not used",
                    "calibration_confidence": "not used",
                    "exact_defect_console": "",
                }

        # ==================================================
        # FRIEND PIPELINE — unchanged
        # ==================================================
        else:
            working_img = img

            if want_rectify:
                if rectify_mode.startswith("Auto"):
                    auto_quad = (
                        calib.detect_reference_quad(
                            working_img
                        )
                    )

                    if auto_quad is not None:
                        working_img = (
                            calib.rectify_perspective(
                                working_img,
                                auto_quad,
                            )
                        )

                elif rectify_points is not None:
                    working_img = (
                        calib.rectify_perspective(
                            working_img,
                            rectify_points,
                        )
                    )

            calibration_result = get_calibration(
                working_img
            )

            out = inspect_image_yolo(
                working_img,
                calibration=calibration_result,
                denoise_method=DENOISE_METHOD,
                enhance_method=ENHANCE_METHOD,
                erode_pixels=erode_pixels,
                yolo_confidence=yolo_confidence,
            )

        out["filename"] = name
        results.append(out)

        progress.progress(
            (i + 1) / len(images_to_process),
            text=f"Processed {name}",
        )
    progress.empty()
    st.session_state["results"] = results

results = st.session_state["results"]

# ======================================================
# Dashboard — summary
# ======================================================
if results:

    # ==================================================
    # TIFANY — EXACT LATEST DEFECT VIEW
    # ==================================================
    if selected_technique == "Defect Detection":

        st.divider()
        st.header("Defect Detection")

        st.caption(
            "Runs the exact latest standalone defect-test logic "
            "on the uploaded image."
        )

        for result in results:

            st.subheader(
                result.get(
                    "filename",
                    "Image",
                )
            )

            left, right = st.columns(2)

            left.image(
                bgr_to_rgb(
                    result["original"]
                ),
                caption="Original",
                width="stretch",
            )

            right.image(
                bgr_to_rgb(
                    result["annotated"]
                ),
                caption=(
                    "Defect result "
                    "(same processing flow as latest separate test)"
                ),
                width="stretch",
            )

            objects = result.get(
                "objects",
                [],
            )

            if not objects:
                st.warning(
                    "No fruit detected."
                )

            for obj in objects:

                fruit_number = (
                    obj.get("index", 0)
                    + 1
                )

                fruit_type = (
                    obj.get("fruit_type")
                    or "Unknown"
                )

                st.markdown(
                    f"### Fruit #{fruit_number} — "
                    f"{fruit_type}"
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Fruit Type",
                    fruit_type,
                )

                confidence = (
                    obj.get("cnn_confidence")
                    if obj.get("cnn_confidence") is not None
                    else obj.get("yolo_confidence")
                )

                m2.metric(
                    "Confidence",
                    (
                        f"{confidence:.2f}%"
                        if confidence is not None
                        else "N/A"
                    ),
                )

                defect_pct = obj.get(
                    "defect_percentage"
                )

                m3.metric(
                    "Defect %",
                    (
                        f"{defect_pct:.2f}%"
                        if defect_pct is not None
                        else "N/A"
                    ),
                )

                m4.metric(
                    "Ripeness",
                    obj.get("ripeness") or "N/A",
                )

                raw_yolo = obj.get(
                    "raw_yolo_type"
                )

                if raw_yolo:
                    st.caption(
                        f"Raw YOLO type: {raw_yolo}"
                    )

                st.divider()

            with st.expander(
                "Separate-test console output"
            ):
                st.code(
                    result.get(
                        "exact_defect_console",
                        "",
                    )
                )

        # Prevent friend summary from re-processing the custom results.
        st.stop()

    # ==================================================
    # FRIEND DASHBOARD — unchanged
    # ==================================================
    st.divider()
    st.header("Summary")

    all_objects = [obj for r in results for obj in r["objects"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Photos inspected", len(results))
    col2.metric("Fruits detected", len(all_objects))
    fresh_n = sum(1 for o in all_objects if o.get("label") == "Fresh")
    rotten_n = sum(1 for o in all_objects if o.get("label") == "Rotten")
    col3.metric("Fresh", fresh_n)
    col4.metric("Rotten", rotten_n)

    chart_col1, chart_col2 = st.columns(2)
    fruit_type_counts = pd.Series([o.get("fruit_type") or "Unknown" for o in all_objects]).value_counts()
    chart_col1.caption("By fruit type")
    chart_col1.bar_chart(fruit_type_counts)
    label_counts = pd.Series([o.get("label") or "Unclassified" for o in all_objects]).value_counts()
    chart_col2.caption("By quality")
    chart_col2.bar_chart(label_counts)

    st.subheader("Every fruit detected")
    table_rows = []
    for r in results:
        for obj in r["objects"]:
            table_rows.append({
                "Image": r["filename"],
                "#": obj["index"] + 1,
                "Fruit Type": obj.get("fruit_type") or "—",
                "Type Conf": f"{obj.get('fruit_type_confidence', 0) * 100:.1f}%" if obj.get("fruit_type") else "—",
                "Quality": obj.get("label") or "—",
                "Quality Conf": f"{obj.get('confidence', 0) * 100:.1f}%" if obj.get("label") else "—",
                "Wound %": f"{obj.get('defect_fraction', 0) * 100:.1f}%",
                "Width (cm)": f"{obj['width_cm']:.2f}" if obj.get("width_cm") is not None else "—",
                "Height (cm)": f"{obj['height_cm']:.2f}" if obj.get("height_cm") is not None else "—",
                "Area (cm^2)": f"{obj['area_cm2']:.2f}" if obj.get("area_cm2") is not None else "—",
                "Width (px)": f"{obj.get('width_px', 0):.0f}",
                "Height (px)": f"{obj.get('height_px', 0):.0f}",
            })
    df = pd.DataFrame(table_rows)
    st.dataframe(df, width="stretch")

    csv = df.to_csv(index=False).encode("utf-8")
    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button("Download results as CSV", csv, "inspection_results.csv", "text/csv")

    if dl_col2.button("Generate PDF report"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            report_mod.generate_report(results, output_path=tmp.name)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
        st.session_state["pdf_bytes"] = pdf_bytes

    if "pdf_bytes" in st.session_state:
        dl_col2.download_button(
            "Download PDF report", st.session_state["pdf_bytes"],
            "inspection_report.pdf", "application/pdf",
        )

    st.divider()
    st.header("Per-photo detail")
    for r in results:
        with st.expander(f"{r['filename']}  —  {r['count']} fruit(s): {summary_to_text(r['summary'])}"):
            c1, c2 = st.columns(2)
            c1.image(bgr_to_rgb(r["original"]), caption="Original", width="stretch")
            c2.image(bgr_to_rgb(r["annotated"]), caption="Detected fruits (bbox + contour + label)", width="stretch")

            if not r["objects"]:
                st.warning("No fruit detected in this photo.")
                continue

            st.markdown(f"**Separated fruit ({r['count']}):**")
            crop_cols = st.columns(len(r["objects"]))
            for obj, col in zip(r["objects"], crop_cols):
                caption = f"#{obj['index'] + 1} {obj.get('fruit_type') or '?'} {obj.get('label') or '?'}"
                col.image(bgr_to_rgb(obj["crop_isolated"]), caption=caption, width="stretch")

            for obj in r["objects"]:
                st.markdown(f"**Fruit #{obj['index'] + 1}**")
                cols = st.columns(4)
                cls = obj.get("classification")

                cols[0].write(f"Type: **{obj.get('fruit_type') or 'N/A'}** "
                               f"({obj.get('fruit_type_confidence', 0) * 100:.0f}%)")
                cols[1].write(f"Quality: **{obj.get('label') or 'N/A'}** "
                               f"({obj.get('confidence', 0) * 100:.0f}%)")
                if obj.get("width_cm") is not None:
                    cols[2].write(f"Size: {obj['width_cm']:.1f} × {obj['height_cm']:.1f} cm")
                    cols[3].write(f"Area: {obj['area_cm2']:.1f} cm²")
                else:
                    cols[2].write(f"Size: {obj.get('width_px', 0):.0f} × {obj.get('height_px', 0):.0f} px")
                    cols[3].write(f"Area: {obj.get('area_px', 0):.0f} px²")

                if cls is not None and cls.error:
                    st.caption(f"⚠️ {cls.error}")
                st.divider()

            st.caption(f"Calibration: {r.get('calibration_method')} ({r.get('calibration_confidence')})")
else:
    st.info("Upload images or point to a folder, then click **Run Inspection**.")