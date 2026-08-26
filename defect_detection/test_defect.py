from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from defect_detection import detect_defect
from ripeness_detection import classify_ripeness


# =================================================
# PATHS
# =================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Friend functions are only USED, not modified
from colorDetection import classify_fruit_type_cnn
from segmentation import (
    contour_shape_metrics,
    segment_all_objects,
)


IMAGE_PATH = CURRENT_DIR / "rotten_banana.png"

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
APPLE_MIN_CONF = 0.65
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

    # ------------------------------------------------------
    # HEALTHY RIPE STRAWBERRY OVERLAP / SHADOW SAFETY
    # ------------------------------------------------------
    #
    # When two strawberries overlap, the neighbouring rotten fruit
    # can enter the YOLO crop of a healthy red strawberry. That dark
    # patch often touches the rectangular ROI edge and was previously
    # counted as a defect on the healthy fruit.
    #
    # Apply this only when the strawberry itself is strongly red.
    # Damaged strawberries with reduced red coverage keep the normal
    # detector so genuine large rot is not removed.

    strawberry_red_ratio = calculate_red_ratio(
        roi,
        strawberry_mask
    )

    if strawberry_red_ratio >= 0.70:

        border_margin = max(
            5,
            int(min(final_mask.shape) * 0.025)
        )

        border_clean = np.zeros_like(
            final_mask
        )

        border_contours, _ = cv.findContours(
            final_mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        fh, fw = final_mask.shape

        for contour in border_contours:

            x, y, cw, ch = cv.boundingRect(
                contour
            )

            touches_roi_edge = (
                x <= border_margin
                or y <= border_margin
                or (x + cw) >= (fw - border_margin)
                or (y + ch) >= (fh - border_margin)
            )

            # On a strongly red/healthy-looking strawberry, a large
            # dark component entering from the ROI edge is much more
            # likely to belong to an overlapping neighbour/shadow.
            if touches_roi_edge:
                continue

            cv.drawContours(
                border_clean,
                [contour],
                -1,
                255,
                cv.FILLED
            )

        final_mask = border_clean

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


# =================================================
# LOAD MODEL
# =================================================

model = YOLO(
    str(MODEL_PATH)
)


# =================================================
# LOAD IMAGE
# =================================================

image = cv.imread(
    str(IMAGE_PATH)
)

if image is None:

    print(
        f"Image not found: {IMAGE_PATH}"
    )

    raise SystemExit


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

        print(
            f"Shape aspect ALL: {aspect:.3f}"
        )

        print(
            f"Shape circularity ALL: "
            f"{circularity if circularity is not None else 'N/A'}"
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

        # -------------------------------------------------
        # SECOND TYPE CHECK ON ORIGINAL ROI
        # -------------------------------------------------
        # The friend's CNN may have been trained on normal photos.
        # A black-background segmented crop can sometimes make a
        # Mango look like Orange. Re-check the untouched YOLO crop.
        raw_cnn_type = None
        raw_cnn_conf = 0.0

        try:
            raw_cnn_type, raw_cnn_conf, _ = (
                classify_fruit_type_cnn(
                    raw_roi
                )
            )
        except Exception:
            raw_cnn_type = None
            raw_cnn_conf = 0.0

        print(
            f"CNN raw-ROI type: {raw_cnn_type} | "
            f"Conf: {raw_cnn_conf * 100:.2f}%"
        )

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
        # GREEN STRAWBERRY / WEAK ORANGE SPECIAL CASE
        # -------------------------------------------------
        # Real unripe Strawberry sample:
        #   YOLO Orange       = 56.20%
        #   processed CNN     = Strawberry 98.02%
        #   raw-ROI CNN       = Strawberry 99.97%
        #   red ratio         = 0.00%
        #   aspect            = 1.218
        #   circularity       = 0.743
        if (
            fruit_type == "orange"
            and raw_yolo_type == "orange"
            and yolo_confidence < 0.70
            and cnn_type == "Strawberry"
            and cnn_conf >= 0.975
            and raw_cnn_type == "Strawberry"
            and raw_cnn_conf >= 0.995
            and red_ratio <= 0.10
            and aspect >= 1.15
            and circularity is not None
            and circularity < 0.76
        ):
            fruit_type = "strawberry"
            confidence = max(
                cnn_conf,
                raw_cnn_conf
            )

            print(
                "Weak Orange corrected to green Strawberry "
                f"(YOLO={yolo_confidence * 100:.2f}%, "
                f"CNN={cnn_conf * 100:.2f}%, "
                f"rawCNN={raw_cnn_conf * 100:.2f}%, "
                f"aspect={aspect:.3f}, "
                f"circularity={circularity:.3f})."
            )

        # -------------------------------------------------
        # DAMAGED STRAWBERRY / APPLE OVERRIDE
        # -------------------------------------------------
        # Real rotten Strawberry sample:
        #   YOLO Apple        = 91.70%
        #   processed CNN     = Strawberry 94.77%
        #   raw-ROI CNN       = Strawberry 99.58%
        #   red ratio         = 39.14%
        #   aspect            = 1.273
        #   circularity       = 0.684
        #
        # The normal damaged-Strawberry rule requires processed CNN
        # confidence >= 97%, so this real Strawberry was left as Apple.
        # Use the raw-ROI CNN as extra evidence, but keep strong
        # shape/red safety checks so normal apples are not changed.
        if (
            fruit_type == "apple"
            and raw_yolo_type == "apple"
            and cnn_type == "Strawberry"
            and cnn_conf >= 0.90
            and raw_cnn_type == "Strawberry"
            and raw_cnn_conf >= 0.99
            and 0.15 <= red_ratio < 0.70
            and aspect >= 1.15
            and circularity is not None
            and circularity < 0.72
        ):
            fruit_type = "strawberry"
            confidence = max(
                cnn_conf,
                raw_cnn_conf
            )

            print(
                "Apple corrected to damaged Strawberry "
                f"(YOLO={yolo_confidence * 100:.2f}%, "
                f"CNN={cnn_conf * 100:.2f}%, "
                f"rawCNN={raw_cnn_conf * 100:.2f}%, "
                f"red={red_ratio * 100:.2f}%, "
                f"aspect={aspect:.3f}, "
                f"circularity={circularity:.3f})."
            )

        # -------------------------------------------------
        # RAW-ROI CNN MANGO RECHECK
        # -------------------------------------------------
        # Trust a strong Mango result from the original, unmasked crop.
        # This helps Mango images that the segmented/calibrated CNN crop
        # incorrectly calls Orange.
        if (
            fruit_type not in {"mango", "strawberry"}
            # Mango is an unseen class for YOLO, but raw-ROI CNN alone
            # is not enough: background / banana fragments can also be
            # called Mango with very high confidence.
            #
            # Require BOTH CNN views to support Mango, plus a
            # Mango-like oval shape.
            and raw_yolo_type in {"orange", "banana"}
            and raw_cnn_type == "Mango"
            and raw_cnn_conf >= 0.95
            and cnn_type == "Mango"
            and cnn_conf >= 0.80
            and aspect >= 1.15
            and circularity is not None
            and circularity < 0.90
        ):
            fruit_type = "mango"
            confidence = max(
                raw_cnn_conf,
                cnn_conf
            )

            print(
                "Raw-ROI CNN corrected fruit to Mango "
                f"(rawCNN={raw_cnn_conf * 100:.2f}%, "
                f"CNN={cnn_conf * 100:.2f}%, "
                f"aspect={aspect:.3f}, "
                f"circularity={circularity:.3f})."
            )

        # -------------------------------------------------
        # BANANA -> MANGO DISAGREEMENT FALLBACK
        # -------------------------------------------------
        # Rotten Mango sample:
        # YOLO Banana 80.16%, processed CNN Orange 76.82%,
        # aspect 1.125, circularity 0.782, red 19.40%.
        #
        # A true banana is normally much more elongated/curved.
        # Only apply when the two models DISAGREE and the object is
        # compact/oval rather than banana-shaped.
        if (
            fruit_type == "banana"
            and raw_yolo_type == "banana"
            and yolo_confidence <= 0.85
            and cnn_type == "Orange"
            and cnn_conf <= 0.85
            and 0.10 <= red_ratio <= 0.35
            and aspect <= 1.25
            and circularity is not None
            and 0.70 <= circularity < 0.85
        ):
            fruit_type = "mango"
            confidence = max(
                yolo_confidence,
                cnn_conf
            )

            print(
                "Banana -> Mango disagreement fallback applied "
                f"(aspect={aspect:.3f}, "
                f"circularity={circularity:.3f}, "
                f"red={red_ratio * 100:.2f}%)."
            )

        # -------------------------------------------------
        # LOW-RED WEAK-ORANGE -> MANGO FALLBACK
        # -------------------------------------------------
        # This handles a Mango that BOTH YOLO and the friend's CNN
        # call Orange, but neither model is very confident.
        #
        # Current real Mango sample:
        #   YOLO Orange      = 87.58%
        #   CNN processed    = 81.79%
        #   CNN raw ROI      = 79.68%
        #   red ratio        = 1.80%
        #   aspect           = 1.045
        #   circularity      = 0.759
        #
        # Safety:
        # - keep this only for weak Orange agreement
        # - require almost no red/orange-red surface
        # - require a noticeably non-round contour
        if (
            fruit_type == "orange"
            and raw_yolo_type == "orange"
            and cnn_type == "Orange"
            and yolo_confidence < 0.90
            and cnn_conf < 0.85
            and (
                raw_cnn_type is None
                or raw_cnn_type == "Orange"
            )
            and raw_cnn_conf < 0.85
            and red_ratio <= 0.08
            and circularity is not None
            and circularity < 0.77
        ):
            fruit_type = "mango"
            confidence = max(
                yolo_confidence,
                cnn_conf,
                raw_cnn_conf
            )

            print(
                "Weak low-red Orange -> Mango fallback applied "
                f"(YOLO={yolo_confidence * 100:.2f}%, "
                f"CNN={cnn_conf * 100:.2f}%, "
                f"rawCNN={raw_cnn_conf * 100:.2f}%, "
                f"red={red_ratio * 100:.2f}%, "
                f"circularity={circularity:.3f})."
            )

        # -------------------------------------------------
        # MANGO SHAPE FALLBACK
        # -------------------------------------------------
        # Special case:
        # YOLO only knows Apple/Banana/Orange, and the friend's
        # CNN can occasionally also call a Mango "Orange".
        #
        # Use a conservative shape fallback only when BOTH models
        # say Orange but the segmented fruit is clearly elongated
        # and less circular than a normal round orange.
        #
        # Current real Mango sample:
        #   aspect      = 1.238
        #   circularity = 0.777
        #   CNN Orange confidence = 90.86%
        if (
            cnn_override is None
            and raw_yolo_type == "orange"
            and cnn_type == "Orange"
            # Strong raw-ROI Orange evidence is a veto.
            # Current real mouldy Orange:
            #   YOLO Orange = 98.25%
            #   raw-ROI CNN Orange = 100.00%
            # Even if segmentation makes the shape elongated,
            # do NOT convert it to Mango.
            and not (
                raw_cnn_type == "Orange"
                and raw_cnn_conf >= 0.95
            )
            and circularity is not None
            and (
                # Green / yellow Mango case:
                # little strawberry-red, moderately elongated.
                (
                    cnn_conf <= 0.93
                    and red_ratio <= 0.15
                    and aspect >= 1.23
                    and circularity < 0.79
                )
                or
                # Ripe / damaged Mango case:
                # red/orange colour can be high, so rely on a
                # distinctly elongated + less-circular fruit shape.
                (
                    aspect >= 1.24
                    and circularity < 0.76
                )
            )
        ):
            fruit_type = "mango"
            confidence = cnn_conf

            print(
                "Orange -> Mango shape fallback applied "
                f"(aspect={aspect:.3f}, "
                f"circularity={circularity:.3f}, "
                f"red={red_ratio * 100:.2f}%)"
            )

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

        # Skip segmented fragments that are already part of a
        # confirmed YOLO fruit.
        #
        # The old rule required >=65% overlap AND almost the same
        # centre. In an apple cluster, segmentation can return only
        # one small piece of an already-detected apple, so its centre
        # shifts and the piece was incorrectly sent to the Mango /
        # Strawberry fallback CNN.
        #
        # Keep Strawberry handling conservative, but for confirmed
        # Apple/Banana/Orange also suppress a smaller contained piece
        # when most of that blob lies inside the YOLO fruit box.
        if (
            (
                confirmed_type == "strawberry"
                and overlap_ratio >= 0.75
            )
            or (
                confirmed_type in {"apple", "banana", "orange"}
                and overlap_ratio >= 0.45
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


# =================================================
# SHOW RESULT
# =================================================

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