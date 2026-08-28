import cv2 as cv
import numpy as np


# ==========================================================
# Fruit Mask
# ==========================================================

def get_fruit_mask(roi):
    """
    Separates fruit from most of the background.

    White = fruit
    Black = background
    """

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = np.zeros(
        saturation.shape,
        dtype=np.uint8
    )

    mask[
        (saturation > 25)
        & (value > 20)
    ] = 255

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask

    largest = max(
        contours,
        key=cv.contourArea
    )

    fruit_mask = np.zeros_like(
        mask
    )

    cv.drawContours(
        fruit_mask,
        [largest],
        -1,
        255,
        cv.FILLED
    )

    return fruit_mask


# ==========================================================
# Suppress ROI Corner Squares
# ==========================================================

def suppress_bbox_corners(
    mask,
    corner_ratio=0.12,
    min_corner_px=8
):
    """
    Zeroes out a small square in each of the four corners of
    a YOLO crop, BEFORE any contour extraction / morphology.

    Why this exists:

    A YOLO bounding box is a rectangle drawn around a round
    (or near-round) fruit. Geometrically, the four literal
    corners of that rectangle are essentially always
    background -- a round fruit that actually reached into a
    sharp 90-degree corner of its own bounding box would mean
    the box was mis-fit in the first place.

    Left alone, a warm-toned or bright background patch
    sitting in one of those corners can pass a loose colour
    threshold (e.g. the orange "peel" HSV range) and then get
    bridged onto the real fruit blob by a large closing
    kernel. Once that happens the corner becomes part of the
    "fruit" contour and everything downstream -- including
    defect detection -- treats it as real peel.

    Blanking the corner squares up front prevents that fusion
    from ever happening, without touching the actual fruit
    body (which never legitimately occupies those squares).
    """

    h, w = mask.shape[:2]

    size = max(
        min_corner_px,
        int(min(h, w) * corner_ratio)
    )

    cleaned = mask.copy()

    cleaned[0:size, 0:size] = 0
    cleaned[0:size, w - size:w] = 0
    cleaned[h - size:h, 0:size] = 0
    cleaned[h - size:h, w - size:w] = 0

    return cleaned


# ==========================================================
# Remove False Defects at ROI Boundary
# ==========================================================

def remove_border_components(
    mask,
    margin=5,
    min_area=50,
    border_overlap_ratio=None,
    kill_corner_blocks=False,
    corner_block_ratio=0.10,
    kill_flush_rectangles=False,
    rect_extent_thresh=0.85,
    rect_min_span_ratio=0.05
):
    """
    Removes detected defect regions that touch
    the YOLO bounding-box boundary.

    border_overlap_ratio=None (default):
        Original behaviour. A component is dropped if its
        BOUNDING BOX merely touches the boundary strip.
        Good for apples, where real defects are small bruises
        that never legitimately reach the crop edge.

    border_overlap_ratio=<0..1>:
        A component is dropped only if that FRACTION of its
        own pixels sit inside the boundary strip. This matters
        for a fruit like a tightly-cropped orange, where a
        large real defect (e.g. mould covering half the peel)
        can have a bounding box that grazes the crop edge even
        though almost none of its pixels are actually there --
        the old bounding-box check would wrongly discard the
        whole defect. Small background-bleed artifacts (a
        triangle sitting right in a bbox corner) are almost
        entirely inside the strip and still get removed.

    kill_corner_blocks=True:
        Extra, unconditional safety net that runs BEFORE the
        ratio check. It drops any component whose bounding box
        touches TWO ADJACENT crop edges at once (i.e. it sits
        in an actual rectangular corner of the ROI), as long as
        that bounding box is reasonably large relative to the
        image. A real defect on a round fruit essentially never
        forms a shape that hugs a right-angle image corner --
        only leaked/background regions do. This catches large
        background blobs that would otherwise survive the
        border_overlap_ratio check because they extend well
        past the boundary strip.

    kill_flush_rectangles=True:
        A second, more general safety net for background/shadow
        leakage that only touches ONE crop edge (so
        kill_corner_blocks doesn't catch it) -- e.g. a patch of
        table shadow under the fruit that got fused into the
        fruit mask and sits flush against the bottom of the ROI,
        without reaching a corner.

        A component is dropped if ALL of the following hold:
          - it touches at least one border edge
          - it is "extent"-high: area / (bbox_w * bbox_h) is
            close to 1, i.e. it is nearly a solid filled
            rectangle rather than an irregular blob
          - it spans a non-trivial chunk of the image along the
            touched edge (not just a tiny sliver)

        Real mould/rot/bruise regions are organic and scalloped
        -- they essentially never form a clean filled rectangle
        with a straight edge exactly flush with the crop
        boundary. That specific geometry is the signature of a
        region that was cut off by, or fused onto the fruit via,
        the ROI boundary itself.
    """

    cleaned = np.zeros_like(
        mask
    )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    h, w = mask.shape

    if border_overlap_ratio is not None:

        border_strip = np.zeros_like(mask)
        border_strip[:margin, :] = 255
        border_strip[h - margin:, :] = 255
        border_strip[:, :margin] = 255
        border_strip[:, w - margin:] = 255

    corner_size = int(min(h, w) * corner_block_ratio)

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area < min_area:
            continue

        x, y, cw, ch = cv.boundingRect(
            contour
        )

        if kill_corner_blocks:

            touches_left = x <= margin
            touches_right = (x + cw) >= w - margin
            touches_top = y <= margin
            touches_bottom = (y + ch) >= h - margin

            in_a_corner = (
                (touches_left or touches_right)
                and (touches_top or touches_bottom)
            )

            # Only treat it as a "corner block" if it's
            # actually sized like one (not a small real
            # blemish that happens to graze one corner tip).
            big_enough = (
                cw >= corner_size
                and ch >= corner_size
            )

            if in_a_corner and big_enough:
                continue

        if kill_flush_rectangles:

            touches_left = x <= margin
            touches_right = (x + cw) >= w - margin
            touches_top = y <= margin
            touches_bottom = (y + ch) >= h - margin

            touches_any_edge = (
                touches_left
                or touches_right
                or touches_top
                or touches_bottom
            )

            bbox_area = cw * ch

            extent = (
                (area / bbox_area)
                if bbox_area > 0 else 0.0
            )

            min_span = int(
                min(h, w) * rect_min_span_ratio
            )

            spans_enough = (
                cw >= min_span
                and ch >= min_span
            )

            if (
                touches_any_edge
                and extent >= rect_extent_thresh
                and spans_enough
            ):
                continue

        if border_overlap_ratio is not None:

            component_mask = np.zeros_like(mask)

            cv.drawContours(
                component_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )

            total_px = cv.countNonZero(component_mask)

            border_px = cv.countNonZero(
                cv.bitwise_and(component_mask, border_strip)
            )

            overlap = (border_px / total_px) if total_px else 0.0

            touches_border = overlap >= border_overlap_ratio

        else:

            touches_border = (
                x <= margin
                or y <= margin
                or x + cw >= w - margin
                or y + ch >= h - margin
            )

        if not touches_border:

            cv.drawContours(
                cleaned,
                [contour],
                -1,
                255,
                cv.FILLED
            )

    return cleaned


# ==========================================================
# Smooth a Binary Mask's Boundary (visual only)
# ==========================================================

def smooth_mask_boundary(
    mask,
    blur_ksize=9,
    close_ksize=5
):
    """
    Softens the staircase/pixelated edge of a binary mask by
    blurring + re-thresholding, then a light closing pass.

    This does NOT change *which* regions are flagged as
    defect -- it only rounds the boundary of regions that are
    already present. It's applied as a final, purely cosmetic
    step so it's safe to bolt onto the very end of a single
    fruit's pipeline (orange) without affecting the earlier
    detection logic or any other fruit type.
    """

    if cv.countNonZero(mask) == 0:
        return mask

    blurred = cv.GaussianBlur(
        mask,
        (blur_ksize, blur_ksize),
        0
    )

    _, smoothed = cv.threshold(
        blurred,
        127,
        255,
        cv.THRESH_BINARY
    )

    kernel = np.ones(
        (close_ksize, close_ksize),
        np.uint8
    )

    smoothed = cv.morphologyEx(
        smoothed,
        cv.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return smoothed


# ==========================================================
# General Brown / Dark Defect Detection
# Used for Apple
# ==========================================================

def calculate_defect(
    roi,
    lower_brown,
    upper_brown,
    dark_threshold
):

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    fruit_mask = get_fruit_mask(
        roi
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # Slightly remove fruit edge
    fruit_mask = cv.erode(
        fruit_mask,
        kernel,
        iterations=2
    )


    # ------------------------------------------------------
    # Brown regions
    # ------------------------------------------------------

    brown_mask = cv.inRange(
        hsv,
        lower_brown,
        upper_brown
    )


    # ------------------------------------------------------
    # Dark / black regions
    # ------------------------------------------------------

    value = hsv[:, :, 2]

    dark_mask = np.zeros(
        value.shape,
        dtype=np.uint8
    )

    dark_mask[
        value < dark_threshold
    ] = 255


    # ------------------------------------------------------
    # Combine defects
    # ------------------------------------------------------

    defect_mask = cv.bitwise_or(
        brown_mask,
        dark_mask
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        fruit_mask
    )


    # ------------------------------------------------------
    # Morphological processing
    # ------------------------------------------------------

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        fruit_mask
    )


    # ------------------------------------------------------
    # Remove false ROI boundary defects
    # ------------------------------------------------------

    defect_mask = remove_border_components(
        defect_mask,
        margin=8,
        min_area=50
    )


    # ------------------------------------------------------
    # Calculate percentage
    # ------------------------------------------------------

    fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    defect_pixels = cv.countNonZero(
        defect_mask
    )

    if fruit_pixels == 0:

        defect_percentage = 0.0

    else:

        defect_percentage = (
            defect_pixels
            / fruit_pixels
        ) * 100

        defect_percentage = min(
            defect_percentage,
            100.0
        )


    # ------------------------------------------------------
    # Draw defects
    # ------------------------------------------------------

    output = roi.copy()

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv.contourArea(contour) > 50:

            cv.drawContours(
                output,
                [contour],
                -1,
                (0, 0, 255),
                2
            )


    return (
        output,
        defect_mask,
        defect_percentage
    )


# ==========================================================
# Banana Defect Detection
# ==========================================================

def detect_banana_defect(roi):
    """
    Robust banana-only defect detection.

    Designed for common YOLO banana crops containing:
    - yellow / ripe bananas
    - green / unripe bananas
    - heavily spotted bananas
    - dark / overripe bananas
    - brown / black rot
    - gray / white mould
    - shadows
    - a hand or warm background touching the banana
    - tight YOLO crops where the fruit reaches the ROI border

    IMPORTANT:
    Only the banana pipeline is changed.
    Apple and orange detection remain untouched.
    """

    if roi is None or roi.size == 0:
        empty_mask = np.zeros((1, 1), dtype=np.uint8)
        return roi, empty_mask, 0.0

    # ------------------------------------------------------
    # Blur / colour spaces
    # ------------------------------------------------------

    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    lab = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2LAB
    )

    ycrcb = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2YCrCb
    )

    gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    lab_b = lab[:, :, 2]

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    h, w = hue.shape

    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    # ======================================================
    # 1. ROBUST BANANA FRUIT MASK
    # ======================================================
    #
    # The old version started with:
    #
    #     saturation > 20 and value > 10
    #
    # which is too broad and can join a hand / table / shadow
    # to the banana when they touch.
    #
    # This version:
    # 1. Finds high-confidence banana-coloured seed pixels.
    # 2. Suppresses likely skin only for boundary construction.
    # 3. Keeps the dominant seed BEFORE any aggressive closing.
    # 4. Grows only through plausible banana pixels.
    # 5. Uses a fallback for very dark / overripe bananas.
    # 6. Fills enclosed dark spots after the boundary is found.


    # ------------------------------------------------------
    # 1A. High-confidence yellow / pale-yellow / green seed
    # ------------------------------------------------------

    yellow_core = (
        (hue >= 15)
        & (hue <= 40)
        & (saturation >= 55)
        & (value >= 35)
        & (lab_b >= 145)
        & (cb <= 108)
    )

    pale_yellow_core = (
        (hue >= 18)
        & (hue <= 38)
        & (saturation >= 25)
        & (saturation < 120)
        & (value >= 80)
        & (lab_b >= 148)
        & (cb <= 112)
    )

    green_core = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 45)
        & (value >= 25)
        & (cb <= 118)
    )

    core_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    core_mask[
        yellow_core
        | pale_yellow_core
        | green_core
    ] = 255


    # ------------------------------------------------------
    # 1B. Suppress likely hand / warm skin-like background
    # ------------------------------------------------------
    #
    # This is ONLY for finding the banana boundary.
    # It does not directly remove defect pixels from the fruit.

    skin_like = (
        (hue <= 30)
        & (saturation >= 20)
        & (saturation <= 200)
        & (cr >= 130)
        & (cr <= 185)
        & (cb >= 100)
        & (cb <= 140)
        & (lab_b <= 165)
    )

    skin_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    skin_mask[
        skin_like
    ] = 255

    # Strong banana evidence overrides the skin rule.
    skin_mask[
        core_mask > 0
    ] = 0


    # ------------------------------------------------------
    # 1C. Clean seed and keep dominant component FIRST
    # ------------------------------------------------------

    core_mask = cv.morphologyEx(
        core_mask,
        cv.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    (
        component_count,
        component_labels,
        component_stats,
        component_centroids
    ) = cv.connectedComponentsWithStats(
        core_mask,
        connectivity=8
    )

    seed_mask = np.zeros_like(
        core_mask
    )

    if component_count > 1:

        centre_x = w / 2.0
        centre_y = h / 2.0

        diagonal = max(
            1.0,
            float(
                np.hypot(
                    w,
                    h
                )
            )
        )

        minimum_seed_area = max(
            20,
            int(
                h
                * w
                * 0.0005
            )
        )

        best_label = None
        best_score = -1.0

        for label in range(
            1,
            component_count
        ):

            area = component_stats[
                label,
                cv.CC_STAT_AREA
            ]

            if area < minimum_seed_area:
                continue

            cx, cy = component_centroids[
                label
            ]

            centre_distance = (
                np.hypot(
                    cx - centre_x,
                    cy - centre_y
                )
                / diagonal
            )

            # Area is the main factor.
            # Centre location is only a small tie-breaker.
            score = (
                float(area)
                * (
                    1.0
                    - 0.35
                    * centre_distance
                )
            )

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is not None:

            seed_mask[
                component_labels
                == best_label
            ] = 255


    # ------------------------------------------------------
    # 1D. Create broader banana candidate
    # ------------------------------------------------------
    #
    # Includes yellow/green peel plus dark rotten peel.
    # Pale mould is accepted only when it has rough texture.

    broad_yellow = (
        (hue >= 8)
        & (hue <= 45)
        & (saturation >= 22)
        & (value >= 18)
        & (cb <= 125)
    )

    broad_green = (
        (hue >= 30)
        & (hue <= 100)
        & (saturation >= 25)
        & (value >= 18)
        & (cb <= 130)
    )

    dark_peel = (
        (value <= 115)
        & (saturation >= 18)
        & (cb <= 132)
    )

    boundary_laplacian = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    boundary_texture = np.clip(
        np.abs(
            boundary_laplacian
        ),
        0,
        255
    ).astype(
        np.uint8
    )

    _, boundary_texture_mask = cv.threshold(
        boundary_texture,
        11,
        255,
        cv.THRESH_BINARY
    )

    boundary_texture_mask = cv.dilate(
        boundary_texture_mask,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    pale_mould_candidate = (
        (saturation <= 110)
        & (value >= 40)
        & (value <= 240)
        & (boundary_texture_mask > 0)
    )

    candidate_mask = np.zeros_like(
        core_mask
    )

    candidate_mask[
        broad_yellow
        | broad_green
        | dark_peel
        | pale_mould_candidate
    ] = 255

    # Remove likely skin/background, but keep strong banana seed.
    candidate_mask[
        skin_mask > 0
    ] = 0

    candidate_mask[
        core_mask > 0
    ] = 255

    # Remove obvious blue/cyan background.
    blue_background = (
        (hue >= 85)
        & (hue <= 135)
        & (saturation > 35)
    )

    candidate_mask[
        blue_background
    ] = 0

    candidate_mask = cv.morphologyEx(
        candidate_mask,
        cv.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )


    # ------------------------------------------------------
    # 1E. Fallback for very dark / severely overripe banana
    # ------------------------------------------------------

    seed_pixels = cv.countNonZero(
        seed_mask
    )

    minimum_useful_seed = max(
        80,
        int(
            h
            * w
            * 0.01
        )
    )

    if seed_pixels < minimum_useful_seed:

        (
            fallback_count,
            fallback_labels,
            fallback_stats,
            _
        ) = cv.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8
        )

        if fallback_count > 1:

            fallback_label = (
                1
                + np.argmax(
                    fallback_stats[
                        1:,
                        cv.CC_STAT_AREA
                    ]
                )
            )

            seed_mask[
                fallback_labels
                == fallback_label
            ] = 255


    # ------------------------------------------------------
    # 1F. Region-grow only through plausible banana pixels
    # ------------------------------------------------------

    grown_mask = seed_mask.copy()

    grow_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    for _ in range(
        max(
            h,
            w
        )
    ):

        expanded_mask = cv.dilate(
            grown_mask,
            grow_kernel,
            iterations=1
        )

        expanded_mask = cv.bitwise_and(
            expanded_mask,
            candidate_mask
        )

        if np.array_equal(
            expanded_mask,
            grown_mask
        ):
            break

        grown_mask = expanded_mask


    # ------------------------------------------------------
    # 1G. Keep final largest fruit body
    # ------------------------------------------------------

    grown_mask = cv.morphologyEx(
        grown_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (5, 5),
            np.uint8
        ),
        iterations=2
    )

    (
        grown_count,
        grown_labels,
        grown_stats,
        _
    ) = cv.connectedComponentsWithStats(
        grown_mask,
        connectivity=8
    )

    fruit_mask = np.zeros_like(
        grown_mask
    )

    if grown_count > 1:

        fruit_label = (
            1
            + np.argmax(
                grown_stats[
                    1:,
                    cv.CC_STAT_AREA
                ]
            )
        )

        fruit_mask[
            grown_labels
            == fruit_label
        ] = 255


    # ------------------------------------------------------
    # 1H. Fill enclosed dark spots / holes
    # ------------------------------------------------------
    #
    # Padding ensures floodFill starts in true background
    # even when the banana touches the YOLO crop edge.

    padded_mask = cv.copyMakeBorder(
        fruit_mask,
        1,
        1,
        1,
        1,
        cv.BORDER_CONSTANT,
        value=0
    )

    flooded = padded_mask.copy()

    flood_helper = np.zeros(
        (
            padded_mask.shape[0] + 2,
            padded_mask.shape[1] + 2
        ),
        dtype=np.uint8
    )

    cv.floodFill(
        flooded,
        flood_helper,
        (0, 0),
        255
    )

    enclosed_holes = cv.bitwise_not(
        flooded
    )

    padded_mask = cv.bitwise_or(
        padded_mask,
        enclosed_holes
    )

    fruit_mask = padded_mask[
        1:-1,
        1:-1
    ]

    fruit_mask = cv.morphologyEx(
        fruit_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (5, 5),
            np.uint8
        ),
        iterations=1
    )

    # Only a tiny edge erosion.
    fruit_mask = cv.erode(
        fruit_mask,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )


    # ======================================================
    # 2. Brown Defect
    # ======================================================
    brown_mask = cv.inRange(
        hsv,
        np.array(
            [3, 70, 20]
        ),
        np.array(
            [16, 255, 140]
        )
    )


    # ======================================================
    # 3. Dark Brown Defect
    # ======================================================
    dark_brown_mask = cv.inRange(
        hsv,
        np.array(
            [0, 50, 0]
        ),
        np.array(
            [25, 255, 70]
        )
    )


    # ======================================================
    # 4. Very Black Defect
    # ======================================================
    very_black_mask = cv.inRange(
        hsv,
        np.array(
            [0, 20, 0]
        ),
        np.array(
            [180, 255, 35]
        )
    )


    # ======================================================
    # 5. Gray / White Mould
    # ======================================================
    mold_mask = cv.inRange(
        hsv,
        np.array(
            [0, 0, 45]
        ),
        np.array(
            [180, 100, 220]
        )
    )

    mold_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mold_mask = cv.morphologyEx(
        mold_mask,
        cv.MORPH_CLOSE,
        mold_kernel,
        iterations=2
    )

    mold_mask = cv.morphologyEx(
        mold_mask,
        cv.MORPH_OPEN,
        mold_kernel,
        iterations=1
    )


    # ======================================================
    # 6. Healthy Yellow Peel
    # ======================================================
    yellow_healthy_mask = cv.inRange(
        hsv,
        np.array(
            [12, 35, 100]
        ),
        np.array(
            [38, 255, 255]
        )
    )


    # ======================================================
    # 7. Yellow Peel Under Shadow
    # ======================================================
    yellow_shadow_mask = cv.inRange(
        hsv,
        np.array(
            [17, 30, 35]
        ),
        np.array(
            [38, 255, 110]
        )
    )


    # ======================================================
    # 8. Healthy Green Peel
    # ======================================================
    green_healthy_mask = cv.inRange(
        hsv,
        np.array(
            [35, 40, 40]
        ),
        np.array(
            [90, 255, 255]
        )
    )


    # ======================================================
    # 9. Combine Defects
    # ======================================================
    dark_mask = cv.bitwise_or(
        dark_brown_mask,
        very_black_mask
    )

    defect_mask = cv.bitwise_or(
        brown_mask,
        dark_mask
    )

    # Add mould
    defect_mask = cv.bitwise_or(
        defect_mask,
        mold_mask
    )


    # ------------------------------------------------------
    # Remove healthy yellow
    # ------------------------------------------------------
    defect_mask = cv.bitwise_and(
        defect_mask,
        cv.bitwise_not(
            yellow_healthy_mask
        )
    )


    # ------------------------------------------------------
    # Remove yellow shadow
    # ------------------------------------------------------
    defect_mask = cv.bitwise_and(
        defect_mask,
        cv.bitwise_not(
            yellow_shadow_mask
        )
    )


    # ------------------------------------------------------
    # Remove healthy green
    # ------------------------------------------------------
    defect_mask = cv.bitwise_and(
        defect_mask,
        cv.bitwise_not(
            green_healthy_mask
        )
    )


    # ======================================================
    # 10. Inner Fruit Mask
    # ======================================================
    inner_fruit_mask = cv.erode(
        fruit_mask,
        kernel,
        iterations=1
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        inner_fruit_mask
    )

    # Preserve a copy BEFORE opening. Small real banana freckles/spots
    # can disappear during morphology, so we recover only compact,
    # dark components from this source later.
    compact_spot_source = defect_mask.copy()


    # ======================================================
    # 11. Morphological Processing
    # ======================================================
    # IMPORTANT: keep this modest. A large CLOSE can connect many
    # individual banana spots into one unnatural long red region.
    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        inner_fruit_mask
    )


    # ======================================================
    # 12. Component cleanup + small spot recovery
    # ======================================================
    fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    min_defect_area = max(
        12,
        int(fruit_pixels * 0.00035)
    )

    # Accurate distance-to-boundary even when the banana touches the
    # YOLO crop border. OpenCV distanceTransform does not treat pixels
    # outside the image as background, so pad with a black frame first.
    padded_for_distance = cv.copyMakeBorder(
        fruit_mask,
        1, 1, 1, 1,
        cv.BORDER_CONSTANT,
        value=0
    )

    boundary_distance = cv.distanceTransform(
        padded_for_distance,
        cv.DIST_L2,
        5
    )[1:-1, 1:-1]

    edge_band = max(
        10,
        int(min(h, w) * 0.085)
    )

    def keep_banana_component(contour, source_mask, min_area, allow_small=False):
        area = cv.contourArea(contour)
        if area < min_area:
            return False

        x, y, cw, ch = cv.boundingRect(contour)
        elongation = max(cw, ch) / max(1.0, float(min(cw, ch)))

        component_mask = np.zeros_like(source_mask)
        cv.drawContours(component_mask, [contour], -1, 255, cv.FILLED)
        component_pixels = cv.countNonZero(component_mask)
        if component_pixels == 0:
            return False

        component_distances = boundary_distance[component_mask > 0]
        edge_fraction = float(np.mean(component_distances < edge_band))

        # False hand/contact shadows typically form a long strip following
        # the banana edge. Reject only when BOTH strongly edge-hugging and
        # elongated, so compact genuine defects near the edge still survive.
        area_fraction = component_pixels / max(1.0, float(fruit_pixels))
        edge_strip_artifact = (
            edge_fraction >= 0.82
            and elongation >= 2.6
            and area_fraction < 0.08
        )

        if edge_strip_artifact:
            return False

        # Anything almost completely cut by the crop edge and very narrow
        # is normally ROI/background leakage.
        touches_roi = (
            x <= 1 or y <= 1
            or x + cw >= w - 1
            or y + ch >= h - 1
        )
        if touches_roi and edge_fraction >= 0.90 and elongation >= 2.2:
            return False

        if allow_small:
            bbox_area = max(1, cw * ch)
            extent = area / bbox_area
            if extent < 0.18:
                return False

        return True

    # ---------------- main / larger defect regions ----------------
    cleaned_main = np.zeros_like(defect_mask)
    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if keep_banana_component(
            contour,
            defect_mask,
            min_defect_area
        ):
            cv.drawContours(
                cleaned_main,
                [contour],
                -1,
                255,
                cv.FILLED
            )

    # ---------------- recover genuine small dark spots ----------------
    # Use the pre-opening source, then keep only compact components. This
    # recovers small black/brown banana spots without re-introducing the
    # long hand-contact shadow.
    spot_mask = cv.bitwise_and(
        compact_spot_source,
        cv.inRange(
            hsv,
            np.array([0, 45, 0]),
            np.array([35, 255, 115])
        )
    )

    spot_mask = cv.bitwise_and(
        spot_mask,
        inner_fruit_mask
    )

    spot_mask = cv.morphologyEx(
        spot_mask,
        cv.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    cleaned_spots = np.zeros_like(spot_mask)
    spot_min_area = max(
        4,
        int(fruit_pixels * 0.00010)
    )
    spot_max_area = max(
        80,
        int(fruit_pixels * 0.012)
    )

    spot_contours, _ = cv.findContours(
        spot_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in spot_contours:
        area = cv.contourArea(contour)
        if area > spot_max_area:
            continue
        if keep_banana_component(
            contour,
            spot_mask,
            spot_min_area,
            allow_small=True
        ):
            cv.drawContours(
                cleaned_spots,
                [contour],
                -1,
                255,
                cv.FILLED
            )

    defect_mask = cv.bitwise_or(
        cleaned_main,
        cleaned_spots
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        inner_fruit_mask
    )

    # ======================================================
    # 13. Calculate Defect Percentage
    # ======================================================
    defect_pixels = cv.countNonZero(
        defect_mask
    )

    if fruit_pixels == 0:
        defect_percentage = 0.0
    else:
        defect_percentage = (
            defect_pixels
            / fruit_pixels
        ) * 100

        defect_percentage = min(
            defect_percentage,
            100.0
        )


    # ======================================================
    # 14. Draw Defects
    # ======================================================
    # DISPLAY ONLY:
    # Draw the real defect contours, then hide only contour pixels that
    # are too close to the outer banana boundary. This avoids the long
    # artificial red line that can follow the fruit/ROI edge.
    output = roi.copy()

    # Reuse the padded distance map from cleanup so ROI-border pixels
    # are correctly considered close to the banana boundary too.
    display_distance = boundary_distance

    h, w = fruit_mask.shape

    display_margin = max(
        7,
        int(min(h, w) * 0.045)
    )

    display_inner_mask = np.zeros_like(
        fruit_mask
    )

    display_inner_mask[
        display_distance >= display_margin
    ] = 255

    contour_line_mask = np.zeros_like(
        defect_mask
    )

    original_defect_contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in original_defect_contours:
        if cv.contourArea(contour) >= min_defect_area:
            cv.drawContours(
                contour_line_mask,
                [contour],
                -1,
                255,
                2
            )

    contour_line_mask = cv.bitwise_and(
        contour_line_mask,
        display_inner_mask
    )

    output[
        contour_line_mask > 0
    ] = (0, 0, 255)


    return (
        output,
        defect_mask,
        defect_percentage
    )

# ==========================================================
# Apple Defect Detection
# ==========================================================

# ==========================================================
# Apple Wrinkle / Texture Detection
# ==========================================================

def detect_apple_wrinkles(
    roi,
    fruit_mask
):
    """
    Detect meaningful apple wrinkle / shrivel clusters while avoiding
    normal red/yellow/green skin colour transitions.

    Key idea:
    - Wrinkles are narrow dark ridges caused mostly by LIGHTNESS change.
    - Natural apple pigmentation changes colour/chroma as well.
    - Therefore we use a Hessian ridge test in Lab space and require
      local colour/chroma similarity before a pixel can become a wrinkle.
    - Only dense local clusters are kept; isolated texture is discarded.
    """

    # ------------------------------------------------------
    # 1. Lab colour space
    # ------------------------------------------------------

    smoothed = cv.GaussianBlur(
        roi,
        (3, 3),
        0
    )

    lab = cv.cvtColor(
        smoothed,
        cv.COLOR_BGR2LAB
    )

    lightness, channel_a, channel_b = cv.split(
        lab
    )


    # ------------------------------------------------------
    # 2. Hessian ridge response
    # ------------------------------------------------------
    # A real crease behaves like a thin dark ridge. One Hessian
    # eigenvalue becomes strong while the other stays much smaller.

    light_float = cv.GaussianBlur(
        lightness,
        (5, 5),
        1.2
    ).astype(np.float32)

    dxx = cv.Sobel(
        light_float,
        cv.CV_32F,
        2,
        0,
        ksize=3
    )

    dyy = cv.Sobel(
        light_float,
        cv.CV_32F,
        0,
        2,
        ksize=3
    )

    dxy = cv.Sobel(
        light_float,
        cv.CV_32F,
        1,
        1,
        ksize=3
    )

    trace = dxx + dyy

    root = np.sqrt(
        (dxx - dyy) ** 2
        + 4.0 * (dxy ** 2)
    )

    eigen_1 = (trace - root) / 2.0
    eigen_2 = (trace + root) / 2.0

    swap = np.abs(eigen_1) > np.abs(eigen_2)

    eigen_small = np.where(
        swap,
        eigen_2,
        eigen_1
    )

    eigen_large = np.where(
        swap,
        eigen_1,
        eigen_2
    )

    ridge_ratio = (
        np.abs(eigen_small)
        / (np.abs(eigen_large) + 1e-3)
    )


    # ------------------------------------------------------
    # 3. Reject normal colour-pattern edges
    # ------------------------------------------------------
    # A wrinkle should be darker than its local neighbourhood but
    # should not strongly change Lab chroma values.

    local_l = cv.GaussianBlur(
        lightness,
        (17, 17),
        0
    ).astype(np.int16)

    local_a = cv.GaussianBlur(
        channel_a,
        (17, 17),
        0
    ).astype(np.int16)

    local_b = cv.GaussianBlur(
        channel_b,
        (17, 17),
        0
    ).astype(np.int16)

    light_drop = (
        local_l
        - lightness.astype(np.int16)
    )

    a_change = np.abs(
        channel_a.astype(np.int16)
        - local_a
    )

    b_change = np.abs(
        channel_b.astype(np.int16)
        - local_b
    )

    wrinkle_candidate = np.zeros_like(
        lightness
    )

    wrinkle_candidate[
        (eigen_large > 20)
        & (ridge_ratio < 0.35)
        & (light_drop >= 5)
        & (a_change <= 10)
        & (b_change <= 12)
    ] = 255


    # ------------------------------------------------------
    # 4. Keep wrinkle analysis away from outer apple edge
    # ------------------------------------------------------

    # IMPORTANT: pad with black before distanceTransform.  When a tight
    # YOLO crop cuts through the apple, OpenCV otherwise does not know
    # that pixels just outside the ROI are background.  That can make
    # border-touching apple pixels look falsely "deep" inside the fruit.
    padded_fruit = cv.copyMakeBorder(
        fruit_mask,
        1, 1, 1, 1,
        cv.BORDER_CONSTANT,
        value=0
    )

    distance = cv.distanceTransform(
        padded_fruit,
        cv.DIST_L2,
        5
    )[1:-1, 1:-1]

    h, w = fruit_mask.shape

    wrinkle_margin = max(
        7,
        int(min(h, w) * 0.050)
    )

    inner_mask = np.zeros_like(
        fruit_mask
    )

    inner_mask[
        distance >= wrinkle_margin
    ] = 255

    wrinkle_candidate = cv.bitwise_and(
        wrinkle_candidate,
        inner_mask
    )

    wrinkle_candidate = cv.morphologyEx(
        wrinkle_candidate,
        cv.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1
    )


    # ------------------------------------------------------
    # 5. Require a LOCAL CLUSTER of crease pixels
    # ------------------------------------------------------
    # A single pigment boundary or reflection line is not enough.
    # Shriveled skin produces many nearby ridge pixels.

    ridge_float = (
        wrinkle_candidate > 0
    ).astype(np.float32)

    local_density = cv.boxFilter(
        ridge_float,
        -1,
        (21, 21),
        normalize=True
    )

    wrinkle_region = np.zeros_like(
        wrinkle_candidate
    )

    wrinkle_region[
        (local_density >= 0.20)
        & (inner_mask > 0)
    ] = 255

    wrinkle_region = cv.morphologyEx(
        wrinkle_region,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    wrinkle_region = cv.morphologyEx(
        wrinkle_region,
        cv.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1
    )


    # ------------------------------------------------------
    # 6. Remove tiny / unrealistic wrinkle clusters
    # ------------------------------------------------------

    fruit_pixels = max(
        1,
        cv.countNonZero(inner_mask)
    )

    minimum_cluster_area = max(
        30,
        int(fruit_pixels * 0.0003)
    )

    cleaned_mask = np.zeros_like(
        wrinkle_region
    )

    contours, _ = cv.findContours(
        wrinkle_region,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area < minimum_cluster_area:
            continue

        x, y, cw, ch = cv.boundingRect(
            contour
        )

        bbox_area = max(
            1,
            cw * ch
        )

        extent = area / bbox_area

        # A huge, almost solid block is more likely to be a colour
        # patch than a wrinkle network.
        if (
            area > fruit_pixels * 0.06
            and extent > 0.65
        ):
            continue

        cv.drawContours(
            cleaned_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )


    wrinkle_pixels = cv.countNonZero(
        cleaned_mask
    )

    wrinkle_percentage = (
        wrinkle_pixels
        / fruit_pixels
    ) * 100.0

    # Ignore very weak global wrinkle signal. This prevents a smooth
    # patterned apple from being treated as shrivelled because of a few
    # isolated skin lines.
    if wrinkle_percentage < 0.35:

        cleaned_mask[:] = 0
        wrinkle_percentage = 0.0


    return (
        cleaned_mask,
        wrinkle_percentage
    )


# ==========================================================
# Apple Stem / Calyx Suppression
# ==========================================================

def get_apple_stem_calyx_mask(
    roi,
    fruit_mask
):
    """
    Suppresses the small natural dark stem/calyx regions near the top
    or bottom centre of an apple.

    Large dark regions are NOT removed, so genuine rot near the top or
    bottom can still be detected.
    """

    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    candidate = (
        ((value <= 95) & (saturation >= 15))
        |
        (
            (hue >= 20)
            & (hue <= 110)
            & (saturation >= 30)
            & (value <= 140)
        )
    )

    candidate_mask = np.zeros_like(
        fruit_mask
    )

    candidate_mask[
        candidate
    ] = 255

    candidate_mask = cv.bitwise_and(
        candidate_mask,
        fruit_mask
    )


    fruit_contours, _ = cv.findContours(
        fruit_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not fruit_contours:
        return np.zeros_like(fruit_mask)

    largest = max(
        fruit_contours,
        key=cv.contourArea
    )

    fx, fy, fw, fh = cv.boundingRect(
        largest
    )


    # ------------------------------------------------------
    # Top / bottom centre search zones
    # ------------------------------------------------------

    zone = np.zeros_like(
        fruit_mask
    )

    top_x1 = fx + int(fw * 0.18)
    top_x2 = fx + int(fw * 0.82)
    top_y2 = min(
        fruit_mask.shape[0],
        fy + int(fh * 0.24)
    )

    zone[
        fy:top_y2,
        top_x1:top_x2
    ] = 255

    bottom_x1 = fx + int(fw * 0.25)
    bottom_x2 = fx + int(fw * 0.75)
    bottom_y1 = max(
        0,
        fy + fh - int(fh * 0.16)
    )
    bottom_y2 = min(
        fruit_mask.shape[0],
        fy + fh
    )

    zone[
        bottom_y1:bottom_y2,
        bottom_x1:bottom_x2
    ] = 255

    candidate_mask = cv.bitwise_and(
        candidate_mask,
        zone
    )

    candidate_mask = cv.morphologyEx(
        candidate_mask,
        cv.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )


    # ------------------------------------------------------
    # Keep only small stem/calyx-like components
    # ------------------------------------------------------

    fruit_pixels = max(
        1,
        cv.countNonZero(fruit_mask)
    )

    maximum_area = (
        fruit_pixels * 0.012
    )

    suppression_mask = np.zeros_like(
        fruit_mask
    )

    contours, _ = cv.findContours(
        candidate_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area < 10:
            continue

        if area > maximum_area:
            continue

        x, y, cw, ch = cv.boundingRect(
            contour
        )

        if cw > fw * 0.22:
            continue

        if ch > fh * 0.22:
            continue

        centre_y = y + (ch / 2.0)

        in_top_zone = (
            centre_y <= fy + fh * 0.24
        )

        in_bottom_zone = (
            centre_y >= fy + fh * 0.84
        )

        if not (
            in_top_zone
            or in_bottom_zone
        ):
            continue

        cv.drawContours(
            suppression_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )


    suppression_mask = cv.dilate(
        suppression_mask,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    return suppression_mask


# ==========================================================
# Apple Defect Detection
# ==========================================================

def detect_apple_defect(roi):
    """
    Robust apple defect detection.

    Handles:
    - brown rot / bruising
    - black / very dark damage
    - dark red / purple bruising
    - olive / green rot
    - gray / white mould with rough texture
    - green mould
    - genuine wrinkle / shrivel clusters

    Designed to reduce false positives caused by:
    - normal red/yellow/green apple striping
    - smooth colour transitions
    - reflections
    - fruit boundary shadows
    - natural stem / calyx regions
    """

    # ======================================================
    # 1. Fruit Mask
    # ======================================================

    fruit_mask = get_fruit_mask(
        roi
    )

    if cv.countNonZero(fruit_mask) == 0:

        return (
            roi.copy(),
            fruit_mask,
            0.0
        )


    # Slightly remove only the extreme outer boundary.
    fruit_mask = cv.erode(
        fruit_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )


    # ======================================================
    # 2. Apple Interior Safe Mask
    # ======================================================

    # Pad before distanceTransform so a tight YOLO crop is treated
    # correctly.  Without this, an apple touching the ROI border can
    # produce false red contours along the crop edge.
    padded_fruit = cv.copyMakeBorder(
        fruit_mask,
        1, 1, 1, 1,
        cv.BORDER_CONSTANT,
        value=0
    )

    distance = cv.distanceTransform(
        padded_fruit,
        cv.DIST_L2,
        5
    )[1:-1, 1:-1]

    h, w = fruit_mask.shape

    colour_margin = max(
        8,
        int(min(h, w) * 0.050)
    )

    colour_safe_mask = np.zeros_like(
        fruit_mask
    )

    colour_safe_mask[
        distance >= colour_margin
    ] = 255


    # ======================================================
    # 3. Preprocessing
    # ======================================================

    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]


    # ------------------------------------------------------
    # Local darkness
    # ------------------------------------------------------
    # A bruise/rot patch is often darker than its local surrounding.

    local_value = cv.GaussianBlur(
        value,
        (31, 31),
        0
    )

    local_darkness = (
        local_value.astype(np.int16)
        - value.astype(np.int16)
    )


    # ------------------------------------------------------
    # Rough texture
    # ------------------------------------------------------
    # Mould / damaged tissue is normally rougher than smooth healthy
    # peel. This is only used together with colour conditions.

    gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    laplacian = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    texture_strength = np.abs(
        laplacian
    )

    rough_texture = np.zeros_like(
        gray
    )

    rough_texture[
        texture_strength >= 16
    ] = 255

    rough_texture = cv.dilate(
        rough_texture,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    rough = (
        rough_texture > 0
    )


    # ======================================================
    # 4. Colour / Damage Conditions
    # ======================================================

    # ------------------------------------------------------
    # Brown rot / bruise
    # ------------------------------------------------------
    # Healthy orange/red apple striping is not accepted from colour
    # alone. It must be strongly dark OR locally dark + rough.

    brown_hue = (
        (hue >= 4)
        & (hue <= 30)
        & (saturation >= 45)
    )

    brown_rot = (
        brown_hue
        & (
            # Very dark brown can be accepted with only a modest local
            # contrast; medium brown must also be locally dark and rough.
            (
                (value <= 100)
                & (local_darkness >= 5)
            )
            |
            (
                (value <= 145)
                & (local_darkness >= 14)
                & rough
            )
        )
    )


    # ------------------------------------------------------
    # Black / very dark damage
    # ------------------------------------------------------

    black_damage = (
        (value <= 82)
        & (saturation >= 15)
    )


    # ------------------------------------------------------
    # Dark red bruise
    # ------------------------------------------------------

    red_bruise = (
        (
            (hue <= 12)
            | (hue >= 170)
        )
        & (saturation >= 55)
        & (value <= 115)
        & (
            # Healthy red apple peel can be naturally dark.  Require
            # stronger local contrast or roughness unless it is extremely
            # dark.
            (
                (value <= 88)
                & (local_darkness >= 4)
            )
            |
            (
                (local_darkness >= 14)
                & rough
            )
        )
    )


    # ------------------------------------------------------
    # Purple / deep-red bruise
    # ------------------------------------------------------

    purple_bruise = (
        (hue >= 135)
        & (hue <= 179)
        & (saturation >= 40)
        & (value <= 115)
        & (
            (local_darkness >= 10)
            | (
                (value <= 88)
                & rough
            )
        )
    )


    # ------------------------------------------------------
    # Olive / green-brown rot
    # ------------------------------------------------------

    olive_rot = (
        (hue >= 20)
        & (hue <= 95)
        & (saturation >= 25)
        & (value <= 118)
        & (
            (local_darkness >= 8)
            | rough
        )
    )


    # ------------------------------------------------------
    # Local dark bruise not covered well by fixed hue
    # ------------------------------------------------------

    local_bruise = (
        (local_darkness >= 24)
        & (value <= 165)
        & (saturation >= 30)
        & rough
    )


    # ------------------------------------------------------
    # Gray / white mould
    # ------------------------------------------------------
    # Low saturation alone is NOT enough because reflections are also
    # low-saturation. Rough texture and a local lightness change are
    # required.

    gray_mould = (
        (saturation <= 95)
        & (value >= 45)
        & (value <= 205)
        & rough
        & (local_darkness >= 5)
    )


    # ------------------------------------------------------
    # Green mould / dark green diseased tissue
    # ------------------------------------------------------

    green_mould = (
        (hue >= 28)
        & (hue <= 105)
        & (saturation >= 20)
        & (saturation <= 180)
        & (value <= 145)
        & rough
        & (
            (local_darkness >= 5)
            | (value <= 105)
        )
    )


    # ======================================================
    # 5. Combine Colour Defects
    # ======================================================

    colour_candidate = (
        brown_rot
        | black_damage
        | red_bruise
        | purple_bruise
        | olive_rot
        | local_bruise
        | gray_mould
        | green_mould
    )

    colour_mask = np.zeros_like(
        fruit_mask
    )

    colour_mask[
        colour_candidate
    ] = 255

    colour_mask = cv.bitwise_and(
        colour_mask,
        colour_safe_mask
    )


    # Conservative morphology:
    # remove isolated noise, then connect nearby pixels inside the same
    # damaged patch without merging the whole apple texture.

    colour_mask = cv.morphologyEx(
        colour_mask,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    colour_mask = cv.morphologyEx(
        colour_mask,
        cv.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1
    )


    # ======================================================
    # 6. Adaptive Component Filtering
    # ======================================================

    fruit_pixels = max(
        1,
        cv.countNonZero(colour_safe_mask)
    )

    minimum_area = max(
        25,
        int(fruit_pixels * 0.0003)
    )

    cleaned_colour_mask = np.zeros_like(
        colour_mask
    )

    colour_contours, _ = cv.findContours(
        colour_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in colour_contours:

        area = cv.contourArea(
            contour
        )

        if area < 8:
            continue

        component_mask = np.zeros_like(
            colour_mask
        )

        cv.drawContours(
            component_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )

        mean_value = cv.mean(
            value,
            mask=component_mask
        )[0]

        x, y, cw, ch = cv.boundingRect(
            contour
        )

        bbox_area = max(
            1,
            cw * ch
        )

        extent = area / bbox_area

        # Tiny regions are kept only when they are truly dark.
        if (
            area < minimum_area
            and mean_value > 76
        ):
            continue

        # Thin, weak streaks are usually healthy pigmentation / lighting.
        if (
            area < fruit_pixels * 0.002
            and extent < 0.13
            and mean_value > 92
        ):
            continue

        cv.drawContours(
            cleaned_colour_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )


    # ------------------------------------------------------
    # Boundary-contact cleanup
    # ------------------------------------------------------
    # A tight crop, stem shadow, or background contact can create a long
    # red contour that follows the apple outline.  Reject only components
    # that live almost entirely in the outer boundary band.  A genuine
    # defect that starts at the edge but extends meaningfully inward is kept.

    boundary_band = max(
        7,
        int(min(h, w) * 0.060)
    )

    deep_band = max(
        boundary_band + 3,
        int(min(h, w) * 0.095)
    )

    boundary_cleaned = np.zeros_like(
        cleaned_colour_mask
    )

    boundary_contours, _ = cv.findContours(
        cleaned_colour_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in boundary_contours:

        area = cv.contourArea(contour)

        if area < 8:
            continue

        x, y, cw, ch = cv.boundingRect(
            contour
        )

        component_mask = np.zeros_like(
            cleaned_colour_mask
        )

        cv.drawContours(
            component_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )

        component_distances = distance[
            component_mask > 0
        ]

        if component_distances.size == 0:
            continue

        edge_fraction = float(
            np.mean(component_distances < boundary_band)
        )

        deep_fraction = float(
            np.mean(component_distances >= deep_band)
        )

        touches_roi = (
            x <= 1
            or y <= 1
            or x + cw >= w - 1
            or y + ch >= h - 1
        )

        elongation = max(cw, ch) / max(
            1.0,
            float(min(cw, ch))
        )

        # Typical crop/fruit-outline artifact: almost all pixels are in
        # the boundary band and the shape does not penetrate into the apple.
        if (
            edge_fraction >= 0.78
            and deep_fraction <= 0.10
            and elongation >= 1.45
        ):
            continue

        # If it actually touches the rectangular ROI edge, be a little
        # stricter.  Real damage is still kept when enough of the component
        # extends into the deep interior.
        if (
            touches_roi
            and edge_fraction >= 0.65
            and deep_fraction <= 0.18
        ):
            continue

        cv.drawContours(
            boundary_cleaned,
            [contour],
            -1,
            255,
            cv.FILLED
        )

    colour_mask = boundary_cleaned


    # ======================================================
    # 7. Wrinkle / Shrivel Detection
    # ======================================================

    (
        wrinkle_mask,
        wrinkle_percentage
    ) = detect_apple_wrinkles(
        roi,
        fruit_mask
    )


    # ======================================================
    # 8. Remove Natural Stem / Calyx
    # ======================================================

    stem_calyx_mask = get_apple_stem_calyx_mask(
        roi,
        fruit_mask
    )

    colour_mask = cv.bitwise_and(
        colour_mask,
        cv.bitwise_not(
            stem_calyx_mask
        )
    )

    wrinkle_mask = cv.bitwise_and(
        wrinkle_mask,
        cv.bitwise_not(
            stem_calyx_mask
        )
    )


    # ======================================================
    # 9. Final Defect Mask
    # ======================================================

    defect_mask = cv.bitwise_or(
        colour_mask,
        wrinkle_mask
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        fruit_mask
    )


    # ======================================================
    # 10. Defect Percentage
    # ======================================================

    fruit_pixels_total = cv.countNonZero(
        fruit_mask
    )

    defect_pixels = cv.countNonZero(
        defect_mask
    )

    if fruit_pixels_total == 0:

        defect_percentage = 0.0

    else:

        defect_percentage = (
            defect_pixels
            / fruit_pixels_total
        ) * 100.0

        defect_percentage = min(
            defect_percentage,
            100.0
        )


    # ======================================================
    # 11. Draw Defect Contours
    # ======================================================

    output = roi.copy()

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    minimum_draw_area = max(
        12,
        int(fruit_pixels_total * 0.00008)
    )

    # Draw into a temporary line mask, then suppress only display pixels
    # that sit right on the fruit boundary.  This prevents a visually ugly
    # red outline even when a valid defect happens to touch the edge.
    contour_line_mask = np.zeros_like(
        defect_mask
    )

    for contour in contours:

        if cv.contourArea(contour) < minimum_draw_area:
            continue

        cv.drawContours(
            contour_line_mask,
            [contour],
            -1,
            255,
            2
        )

    display_margin = max(
        3,
        int(min(h, w) * 0.022)
    )

    display_safe_mask = np.zeros_like(
        fruit_mask
    )

    display_safe_mask[
        distance >= display_margin
    ] = 255

    contour_line_mask = cv.bitwise_and(
        contour_line_mask,
        display_safe_mask
    )

    output[
        contour_line_mask > 0
    ] = (0, 0, 255)


    return (
        output,
        defect_mask,
        defect_percentage
    )


# ==========================================================
# Orange Wrinkle / Shrivel / Dry Peel Detection
# ==========================================================

def detect_orange_wrinkles(
    roi,
    fruit_mask,
    calyx_mask=None
):
    """
    Detects overripe orange surface texture:

    1. Strong wrinkle / shrivel lines
    2. Pale / white dried peel areas

    Normal smooth orange peel and bright reflections
    are reduced by requiring pale regions to also
    contain rough texture.
    """

    # ======================================================
    # 1. PREPROCESSING
    # ======================================================

    gray = cv.cvtColor(
        roi,
        cv.COLOR_BGR2GRAY
    )

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    # Blur normal small orange-peel texture
    blurred = cv.GaussianBlur(
        gray,
        (7, 7),
        0
    )


    # ======================================================
    # 2. CREATE INNER FRUIT MASK
    # ======================================================

    edge_kernel = np.ones(
        (9, 9),
        np.uint8
    )

    # Ignore outer fruit boundary
    inner_mask = cv.erode(
        fruit_mask,
        edge_kernel,
        iterations=2
    )


    # ======================================================
    # 3. STRONG WRINKLE LINE DETECTION
    # ======================================================

    edges = cv.Canny(
        blurred,
        90,
        180
    )

    edges = cv.bitwise_and(
        edges,
        inner_mask
    )


    # Connect nearby wrinkle lines
    wrinkle_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    wrinkle_line_mask = cv.morphologyEx(
        edges,
        cv.MORPH_CLOSE,
        wrinkle_kernel,
        iterations=1
    )

    # Slightly thicken wrinkle lines
    wrinkle_line_mask = cv.dilate(
        wrinkle_line_mask,
        wrinkle_kernel,
        iterations=1
    )

    wrinkle_line_mask = cv.bitwise_and(
        wrinkle_line_mask,
        inner_mask
    )


    # ======================================================
    # 4. REMOVE SMALL NORMAL PEEL TEXTURE
    # ======================================================

    cleaned_wrinkle_lines = np.zeros_like(
        wrinkle_line_mask
    )

    contours, _ = cv.findContours(
        wrinkle_line_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        # Only keep stronger wrinkle structures
        if area > 60:

            # --------------------------------------------------
            # FIX: reject thin, elongated single strokes.
            #
            # A genuine wrinkle/shrivel area is a network of
            # many short, crossing creases -- after the close +
            # dilate above, that network fills in to a moderately
            # dense cluster. A stray Canny edge from a specular
            # highlight on glossy peel (very common on citrus) is
            # instead a single long, thin, mostly-empty curve: it
            # can clear the area>60 bar while covering only a
            # sliver of its own bounding box.
            #
            # "Extent" (contour area / bbox area) captures this:
            # a filled cluster has moderate-to-high extent, a
            # thin worm-like line has very low extent. Require a
            # minimum extent so isolated highlight lines are
            # dropped while real wrinkle clusters pass through.
            # --------------------------------------------------

            bx, by, bw, bh = cv.boundingRect(
                contour
            )

            bbox_area = bw * bh

            extent = (
                area / bbox_area
                if bbox_area > 0 else 0.0
            )

            if extent < 0.18:
                continue

            cv.drawContours(
                cleaned_wrinkle_lines,
                [contour],
                -1,
                255,
                cv.FILLED
            )


    # ======================================================
    # 5. PALE / WHITE DRIED PEEL CANDIDATE
    # ======================================================

    # Dried peel tends to become less saturated
    # and lighter / whitish.
    pale_dry_mask = cv.inRange(
        hsv,
        np.array([0, 0, 75]),
        np.array([180, 125, 255])
    )

    # Must stay inside orange
    pale_dry_mask = cv.bitwise_and(
        pale_dry_mask,
        inner_mask
    )


    # ======================================================
    # 6. ROUGH TEXTURE DETECTION
    # ======================================================

    # Laplacian highlights rough / cracked texture.
    laplacian = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    texture_strength = np.abs(
        laplacian
    )

    texture_strength = np.clip(
        texture_strength,
        0,
        255
    ).astype(np.uint8)


    # Strong texture only
    _, texture_mask = cv.threshold(
        texture_strength,
        12,
        255,
        cv.THRESH_BINARY
    )


    # Expand texture so an old dry patch becomes
    # a region rather than many tiny lines.
    texture_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    texture_mask = cv.dilate(
        texture_mask,
        texture_kernel,
        iterations=2
    )

    texture_mask = cv.bitwise_and(
        texture_mask,
        inner_mask
    )


    # ======================================================
    # 7. CONFIRM DRIED PALE PEEL
    # ======================================================

    # Important:
    # Pale colour alone is NOT enough.
    # It must also contain rough texture.
    dry_peel_mask = cv.bitwise_and(
        pale_dry_mask,
        texture_mask
    )


    dry_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # Fill nearby cracks / dried regions
    dry_peel_mask = cv.morphologyEx(
        dry_peel_mask,
        cv.MORPH_CLOSE,
        dry_kernel,
        iterations=2
    )

    dry_peel_mask = cv.morphologyEx(
        dry_peel_mask,
        cv.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    dry_peel_mask = cv.bitwise_and(
        dry_peel_mask,
        inner_mask
    )


    # ======================================================
    # 8. REMOVE VERY SMALL DRY PATCH NOISE
    # ======================================================

    cleaned_dry_mask = np.zeros_like(
        dry_peel_mask
    )

    dry_contours, _ = cv.findContours(
        dry_peel_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in dry_contours:

        area = cv.contourArea(
            contour
        )

        # Keep meaningful dry regions
        if area > 80:

            cv.drawContours(
                cleaned_dry_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )


    # ======================================================
    # 9. COMBINE WRINKLES + DRIED PEEL
    # ======================================================

    wrinkle_mask = cv.bitwise_or(
        cleaned_wrinkle_lines,
        cleaned_dry_mask
    )


    # Keep everything inside valid fruit
    wrinkle_mask = cv.bitwise_and(
        wrinkle_mask,
        inner_mask
    )


    # ======================================================
    # 10. REMOVE CALYX / STEM
    # ======================================================

    if calyx_mask is not None:

        wrinkle_mask = cv.bitwise_and(
            wrinkle_mask,
            cv.bitwise_not(
                calyx_mask
            )
        )


    # ======================================================
    # 11. FINAL CLEANING
    # ======================================================

    wrinkle_mask = cv.morphologyEx(
        wrinkle_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    wrinkle_mask = cv.bitwise_and(
        wrinkle_mask,
        inner_mask
    )


    return wrinkle_mask

# ==========================================================
# Orange Defect Detection
# ==========================================================

def detect_orange_defect(roi):
    """
    Detect orange surface defects.

    Detects:
    - brown / rotten areas
    - black / dark damage
    - minor dark blemishes
    - white / gray mould
    - pale green mould
    - dark green mould

    All defects are restricted to the actual
    detected orange boundary.
    """

    # ======================================================
    # 1. PREPROCESSING
    # ======================================================

    blurred = cv.GaussianBlur(
        roi,
        (7, 7),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    # ======================================================
    # 2. DETECT ACTUAL ORANGE BOUNDARY
    # ======================================================

    # Include orange, yellow and green orange peel.
    peel_mask = cv.inRange(
        hsv,
        np.array([0, 30, 25]),
        np.array([95, 255, 255])
    )

    # ------------------------------------------------------
    # FIX: strip the literal ROI corner squares BEFORE any
    # closing happens.
    #
    # A YOLO crop is a rectangle around a round fruit, so the
    # four sharp corners of that rectangle are essentially
    # always background. If a corner happens to be warm-toned
    # (common under indoor lighting) it can pass this loose
    # peel threshold on its own. The big closing kernel below
    # (previously 11x11 x3) is then strong enough to visually
    # BRIDGE that isolated corner blob onto the real orange
    # blob, so cv.findContours sees them as one shape and the
    # background gets baked into "fruit" from this point on --
    # nothing downstream can undo that.
    #
    # Blanking the corner squares first means a warm corner
    # can no longer connect to the fruit, regardless of how
    # aggressive the closing is.
    # ------------------------------------------------------
    peel_mask = suppress_bbox_corners(
        peel_mask,
        corner_ratio=0.12
    )

    # Connect nearby peel regions.
    # (kernel size/iterations reduced from 11x11 x3 -- that
    # was aggressive enough to bridge gaps of ~20px, which is
    # more than enough to fuse a corner blob onto the fruit
    # even after corner suppression catches most cases. This
    # is a second, smaller safety margin.)
    peel_mask = cv.morphologyEx(
        peel_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (9, 9),
            np.uint8
        ),
        iterations=2
    )

    # Remove small background noise.
    peel_mask = cv.morphologyEx(
        peel_mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # ======================================================
    # 3. FIND LARGEST FRUIT CONTOUR
    # ======================================================

    contours, _ = cv.findContours(
        peel_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    fruit_mask = np.zeros(
        roi.shape[:2],
        dtype=np.uint8
    )

    if not contours:

        return (
            roi.copy(),
            fruit_mask,
            0.0
        )


    largest_contour = max(
        contours,
        key=cv.contourArea
    )


    # ------------------------------------------------------
    # UPDATED (v2):
    # peel_mask only keeps saturated orange/yellow/green
    # pixels. A large desaturated concavity (e.g. a mould
    # patch) can bite a chunk out of the fruit's own outer
    # boundary rather than leaving a simple interior hole
    # closing can bridge -- that chunk then gets treated as
    # "not fruit" and every downstream defect mask silently
    # ignores it, even though it's clearly diseased peel.
    #
    # A naive convexHull() "fixes" that by filling every
    # concavity -- but a rectangular YOLO crop around a round
    # fruit ALWAYS has real background in its four corners,
    # and the hull straightens across those corners too,
    # wrongly turning a wedge of background into "fruit".
    # Shadow in that wedge then gets misread as a defect.
    #
    # Fix: only accept a hull-added region as real fruit if
    # it's a true INTERIOR concavity (fully surrounded by
    # peel, not touching the crop edge). Hull-added regions
    # that touch the crop boundary are corner background and
    # are discarded, leaving the original (non-hull) contour
    # there.
    # ------------------------------------------------------

    raw_contour_mask = np.zeros_like(
        fruit_mask
    )

    cv.drawContours(
        raw_contour_mask,
        [largest_contour],
        -1,
        255,
        cv.FILLED
    )

    fruit_hull = cv.convexHull(
        largest_contour
    )

    hull_mask = np.zeros_like(
        fruit_mask
    )

    cv.drawContours(
        hull_mask,
        [fruit_hull],
        -1,
        255,
        cv.FILLED
    )

    # Regions the hull added on top of the raw contour.
    hull_gap = cv.bitwise_and(
        hull_mask,
        cv.bitwise_not(raw_contour_mask)
    )

    safe_gap = remove_border_components(
        hull_gap,
        margin=3,
        min_area=20
    )

    fruit_mask = cv.bitwise_or(
        raw_contour_mask,
        safe_gap
    )

# ======================================================
# GEOMETRIC ORANGE BOUNDARY REFINEMENT
# ======================================================
# Orange is approximately elliptical.
# Use ellipse as a geometric constraint so that
# table shadows / drooping tails cannot become fruit.

    # Fallback geometry if ellipse fitting is unavailable.
    ellipse_mask = hull_mask.copy()

    if len(largest_contour) >= 5:

        center, axes, angle = cv.fitEllipse(
            largest_contour
        )

        # Slightly enlarge the fitted ellipse so genuine
        # peel near the boundary is not removed.
        expanded_ellipse = (
            center,
            (
                axes[0] * 1.04,
                axes[1] * 1.04
            ),
            angle
        )

        ellipse_mask = np.zeros_like(
            fruit_mask
        )

        cv.ellipse(
            ellipse_mask,
            expanded_ellipse,
            255,
            cv.FILLED
        )

        # Actual segmentation AND expected orange geometry
        fruit_mask = cv.bitwise_and(
            fruit_mask,
            ellipse_mask
        )

    # ======================================================
    # 3b. RECOVER SEVERE WHITE / GRAY MOULD
    # ======================================================
    #
    # A badly rotten orange can lose most of its normal orange
    # colour. The original peel mask is saturation-based, so a
    # large gray/white mould patch may be excluded from the
    # fruit boundary itself. Once excluded, later defect masks
    # can never count it.
    #
    # Recover only LARGE, rough, low-saturation regions that:
    #   1. sit inside the expected orange geometry, and
    #   2. are connected to / immediately beside the already
    #      detected fruit body.
    #
    # This keeps the fix conservative and avoids simply filling
    # the entire YOLO rectangle or ellipse as fruit.

    recovery_gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    low_sat_mould_candidate = cv.inRange(
        hsv,
        np.array([0, 0, 45]),
        np.array([180, 115, 235])
    )

    pale_green_recovery = cv.inRange(
        hsv,
        np.array([25, 5, 45]),
        np.array([105, 125, 225])
    )

    recovery_candidate = cv.bitwise_or(
        low_sat_mould_candidate,
        pale_green_recovery
    )

    # Rough mould has strong local texture. This reduces
    # smooth gray background/highlight regions.
    recovery_laplacian = cv.Laplacian(
        recovery_gray,
        cv.CV_32F,
        ksize=3
    )

    recovery_texture_strength = np.abs(
        recovery_laplacian
    )

    recovery_texture_strength = np.clip(
        recovery_texture_strength,
        0,
        255
    ).astype(np.uint8)

    _, recovery_texture_mask = cv.threshold(
        recovery_texture_strength,
        10,
        255,
        cv.THRESH_BINARY
    )

    recovery_texture_mask = cv.dilate(
        recovery_texture_mask,
        np.ones(
            (5, 5),
            np.uint8
        ),
        iterations=2
    )

    recovery_candidate = cv.bitwise_and(
        recovery_candidate,
        recovery_texture_mask
    )

    # Never recover outside the expected orange geometry.
    recovery_candidate = cv.bitwise_and(
        recovery_candidate,
        ellipse_mask
    )

    # Connect nearby mould fragments so one severe patch is
    # evaluated as a region rather than many tiny pieces.
    recovery_candidate = cv.morphologyEx(
        recovery_candidate,
        cv.MORPH_CLOSE,
        np.ones(
            (7, 7),
            np.uint8
        ),
        iterations=2
    )

    # Existing fruit acts as the seed. A valid recovered mould
    # region must touch this expanded seed.
    recovery_seed = cv.dilate(
        fruit_mask,
        np.ones(
            (11, 11),
            np.uint8
        ),
        iterations=2
    )

    recovered_mold_mask = np.zeros_like(
        fruit_mask
    )

    recovery_contours, _ = cv.findContours(
        recovery_candidate,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    expected_orange_pixels = cv.countNonZero(
        ellipse_mask
    )

    min_recovery_area = max(
        80,
        int(expected_orange_pixels * 0.005)
    )

    for recovery_contour in recovery_contours:

        recovery_area = cv.contourArea(
            recovery_contour
        )

        if recovery_area < min_recovery_area:
            continue

        component_mask = np.zeros_like(
            fruit_mask
        )

        cv.drawContours(
            component_mask,
            [recovery_contour],
            -1,
            255,
            cv.FILLED
        )

        seed_overlap = cv.countNonZero(
            cv.bitwise_and(
                component_mask,
                recovery_seed
            )
        )

        if seed_overlap == 0:
            continue

        cv.drawContours(
            recovered_mold_mask,
            [recovery_contour],
            -1,
            255,
            cv.FILLED
        )

    # Recovered severe mould is part of the fruit surface.
    fruit_mask = cv.bitwise_or(
        fruit_mask,
        recovered_mold_mask
    )

    fruit_mask = cv.bitwise_and(
        fruit_mask,
        ellipse_mask
    )

    # ======================================================
    # 4. CREATE SAFE REGION INSIDE ORANGE
    # ======================================================

    # Slightly shrink actual boundary so fruit-edge
    # shadows are not interpreted as defects.
    boundary_kernel = np.ones(
        (9, 9),
        np.uint8
    )

# ======================================================
# Create safe area inside actual orange boundary
# ======================================================

    distance = cv.distanceTransform(
        fruit_mask,
        cv.DIST_L2,
        5
    )

    # Ignore approximately outer 3% of fruit region
    h, w = fruit_mask.shape
    boundary_distance = max(
        10,
        int(min(h, w) * 0.03)
    )

    inside_orange_mask = np.zeros_like(
        fruit_mask
    )

    inside_orange_mask[
        distance >= boundary_distance
    ] = 255

    # ======================================================
    # Extra-safe surface region for brown/dark defects
    # ======================================================

    surface_distance = max(
        18,
        int(min(h, w) * 0.06)
    )

    surface_mask = np.zeros_like(
        fruit_mask
    )

    surface_mask[
        distance >= surface_distance
    ] = 255

    # ======================================================
# COLOUR-AWARE CALYX / STEM MASK
# ======================================================
# Calyx is normally strongly saturated olive/brown.
# White/gray mould usually has much lower saturation.
# Only inspect the upper portion of the orange.

    x, y, fw, fh = cv.boundingRect(
        largest_contour
    )

    top_region = np.zeros_like(
        fruit_mask
    )

    top_limit = min(
        y + int(fh * 0.30),
        fruit_mask.shape[0]
    )

    top_region[
        y:top_limit,
        x:x + fw
    ] = 255


    # Strongly saturated olive / brown calyx colours
    calyx_colour = cv.inRange(
        hsv,
        np.array([5, 120, 20]),
        np.array([70, 255, 190])
    )

    calyx_candidate = cv.bitwise_and(
        calyx_colour,
        top_region
    )

    calyx_candidate = cv.bitwise_and(
        calyx_candidate,
        fruit_mask
    )


    calyx_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    calyx_candidate = cv.morphologyEx(
        calyx_candidate,
        cv.MORPH_OPEN,
        calyx_kernel,
        iterations=1
    )

    calyx_candidate = cv.morphologyEx(
        calyx_candidate,
        cv.MORPH_CLOSE,
        calyx_kernel,
        iterations=1
    )


    # Keep only the most likely calyx component
    calyx_mask = np.zeros_like(
        fruit_mask
    )

    calyx_contours, _ = cv.findContours(
        calyx_candidate,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if calyx_contours:

        fruit_area = cv.contourArea(
            largest_contour
        )

        target_x = x + fw / 2
        target_y = y + fh * 0.08

        best_contour = None
        best_distance = float("inf")

        for contour in calyx_contours:

            area = cv.contourArea(
                contour
            )

            # Calyx should be relatively small
            if area < 20:
                continue

            if area > fruit_area * 0.03:
                continue

            # ------------------------------------------------
            # FIX: reject irregular/elongated fragments.
            #
            # calyx_colour's hue range (5-70) overlaps heavily
            # with brown_mask's scar/rot range (0-25), so a
            # small piece of a brown scar sitting in the top
            # 30% of the fruit can pass every earlier check and
            # get punched out of the defect mask as if it were
            # the stem cap -- producing a false "hole" ring in
            # the middle of a real defect.
            #
            # A true calyx cap is compact/roughly round. A
            # fragment carved out of a scar by morphology is
            # typically irregular. Reject low-circularity
            # candidates before they can even be considered.
            # ------------------------------------------------

            perimeter = cv.arcLength(
                contour,
                True
            )

            circularity = (
                (4 * np.pi * area) / (perimeter ** 2)
                if perimeter > 0 else 0.0
            )

            if circularity < 0.35:
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

            distance_to_top = (
                (cx - target_x) ** 2
                + (cy - target_y) ** 2
            )

            if distance_to_top < best_distance:

                best_distance = distance_to_top
                best_contour = contour


        if best_contour is not None:

            cv.drawContours(
                calyx_mask,
                [best_contour],
                -1,
                255,
                cv.FILLED
            )

    # ======================================================
    # 5. LOCAL DARK BLEMISH DETECTION
    # ======================================================

    value = hsv[:, :, 2]

    # Estimate surrounding/local brightness
    local_brightness = cv.GaussianBlur(
        value,
        (31, 31),
        0
    )

    # Difference between surrounding surface
    # and current pixel
    dark_difference = cv.subtract(
        local_brightness,
        value
    )

    relative_dark_mask = np.zeros_like(
        value,
        dtype=np.uint8
    )

    # A blemish should be locally darker, not merely darker
    # because it is near the curved edge of the orange.
    #
    # FIX: two changes to stop this mask alone from flagging
    # plain shading near the fruit's rim (worst at the bottom,
    # where curvature falls away from the light source) as a
    # standalone defect:
    #
    #   1. Gated against surface_mask (outer ~6% trimmed)
    #      instead of inside_orange_mask (outer ~3% trimmed).
    #      Curvature shading is strongest right at the rim, so
    #      the wider trim removes it while leaving genuine
    #      interior blemishes untouched.
    #
    #   2. Threshold raised from 18 to 26. A shallow, gradual
    #      curvature gradient rarely exceeds this; a real
    #      bruise/blemish is a much sharper local drop.
    relative_dark_mask[
        (surface_mask > 0)
        & (dark_difference > 26)
    ] = 255

    blemish_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    relative_dark_mask = cv.morphologyEx(
        relative_dark_mask,
        cv.MORPH_OPEN,
        blemish_kernel,
        iterations=1
    )

    relative_dark_mask = cv.morphologyEx(
        relative_dark_mask,
        cv.MORPH_CLOSE,
        blemish_kernel,
        iterations=2
    )

    relative_dark_mask = cv.bitwise_and(
        relative_dark_mask,
        inside_orange_mask
    )


    # ======================================================
    # 6. BROWN / ROTTEN AREA
    # ======================================================

    brown_mask = cv.inRange(
        hsv,
        np.array([0, 70, 20]),
        np.array([25, 255, 160])
    )

    # ======================================================
# Sensitive small brown blemish confirmation
# ======================================================

    brown_local_dark = np.zeros_like(
        dark_difference,
        dtype=np.uint8
    )

    # FIX: threshold raised from 10 to 22, matching the same
    # reasoning as relative_dark_mask above. A threshold of 10
    # is shallow enough that ordinary contact-shadow (where the
    # fruit rests against the table/background, which is often
    # itself a warm/brown tone) confirms as "rot" the instant
    # it overlaps brown_mask's colour range. Real brown rot is
    # a much sharper local drop than ambient contact shadow.
    brown_local_dark[
        dark_difference > 20
    ] = 255

    confirmed_brown_mask = cv.bitwise_and(
        brown_mask,
        brown_local_dark
    )

    confirmed_brown_mask = cv.bitwise_and(
        confirmed_brown_mask,
        surface_mask
    )


    # ======================================================
    # 7. BLACK / VERY DARK DAMAGE
    # ======================================================

    dark_mask = cv.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 75])
    )


    # ======================================================
    # 8. WHITE / GRAY MOULD
    # ======================================================

    white_gray_mold = cv.inRange(
        hsv,
        np.array([0, 0, 80]),
        np.array([180, 90, 245])
    )

    # ======================================================
    # Severe mould detection
    # For oranges that have lost most normal orange colour
    # ======================================================

    gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    severe_mold_colour = cv.inRange(
        hsv,
        np.array([0, 0, 45]),
        np.array([180, 125, 240])
    )

    lap = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    texture = np.abs(
        lap
    )

    texture = np.clip(
        texture,
        0,
        255
    ).astype(np.uint8)

    _, rough_texture = cv.threshold(
        texture,
        10,
        255,
        cv.THRESH_BINARY
    )

    rough_texture = cv.dilate(
        rough_texture,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    severe_mold_mask = cv.bitwise_and(
        severe_mold_colour,
        rough_texture
    )


    # ------------------------------------------------------
    # Restrict severe mould to central area of YOLO ROI
    # instead of relying on normal orange peel segmentation
    # ------------------------------------------------------

    roi_h, roi_w = severe_mold_mask.shape

    central_mask = np.zeros_like(
        severe_mold_mask
    )

    cv.ellipse(
        central_mask,
        (
            roi_w // 2,
            roi_h // 2
        ),
        (
            int(roi_w * 0.46),
            int(roi_h * 0.46)
        ),
        0,
        0,
        360,
        255,
        cv.FILLED
    )

    severe_mold_mask = cv.bitwise_and(
        severe_mold_mask,
        central_mask
    )

    severe_mold_mask = cv.morphologyEx(
        severe_mold_mask,
        cv.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )


    # ======================================================
    # 9. PALE GREEN MOULD
    # ======================================================

    pale_green_mold = cv.inRange(
        hsv,
        np.array([30, 5, 70]),
        np.array([95, 90, 230])
    )


    # ======================================================
    # 10. HEALTHY GREEN PEEL
    # ======================================================

    # Saturated green is more likely normal unripe peel.
    healthy_green = cv.inRange(
        hsv,
        np.array([30, 100, 40]),
        np.array([95, 255, 255])
    )


    # ======================================================
    # 10b. SMOOTH SPECULAR HIGHLIGHT / REFLECTION EXCLUSION
    # ======================================================
    #
    # Apply this ONLY when the orange is strongly green/unripe.
    # This prevents bright smooth reflections on glossy green
    # peel from being mistaken for white/gray mould or dry peel.
    #
    # Real mould is normally rough/irregular, while a reflection
    # is usually:
    #   - very bright,
    #   - low saturation,
    #   - locally brighter than nearby peel,
    #   - relatively smooth.
    #
    # This block is intentionally conservative so mature orange
    # mould cases are not changed.

    highlight_fruit_pixels = cv.countNonZero(
        inside_orange_mask
    )

    healthy_green_inside = cv.bitwise_and(
        healthy_green,
        inside_orange_mask
    )

    healthy_green_pixels = cv.countNonZero(
        healthy_green_inside
    )

    green_surface_ratio = (
        healthy_green_pixels / highlight_fruit_pixels
        if highlight_fruit_pixels > 0
        else 0.0
    )

    # Strongly green peel means this is an unripe/green orange.
    # We use a more conservative defect path for this condition
    # so normal green shading, glossy highlights and peel texture
    # are not treated like overripe damage.
    # Use the conservative green/unripe defect path even when
    # the fruit is partly green / partly yellow-orange.
    # Mixed-colour unripe citrus often has only a modest amount of
    # pure green pixels but still should not use the aggressive
    # mature-orange defect thresholds.
    is_green_orange = (
        green_surface_ratio >= 0.08
    )

    highlight_exclusion = np.zeros_like(
        inside_orange_mask
    )

    if is_green_orange:

        # --------------------------------------------------
        # GREEN-ORANGE GLOSS / REFLECTION CANDIDATES
        # --------------------------------------------------
        #
        # A healthy green orange often has a large pale-green /
        # whitish glossy patch. Citrus peel texture means that
        # patch is NOT always perfectly smooth, so the previous
        # "texture <= 18" rule was too strict and allowed the
        # highlight to become false mould.
        #
        # Build two conservative reflection candidates:
        #   1. very bright low-saturation white glare
        #   2. bright desaturated GREEN glare
        #
        # Both must also be locally brighter than surrounding peel.

        white_glare = cv.inRange(
            hsv,
            np.array([0, 0, 205]),
            np.array([180, 90, 255])
        )

        green_glare = cv.inRange(
            hsv,
            np.array([25, 20, 185]),
            np.array([70, 155, 255])
        )

        bright_difference = cv.subtract(
            value,
            local_brightness
        )

        locally_bright = np.zeros_like(
            value,
            dtype=np.uint8
        )

        locally_bright[
            bright_difference >= 8
        ] = 255

        # Smoothness is still useful for white glare, but no longer
        # mandatory for the green-glare branch because citrus peel
        # itself naturally has texture.
        smooth_texture = np.zeros_like(
            texture,
            dtype=np.uint8
        )

        smooth_texture[
            texture <= 24
        ] = 255

        smooth_white_glare = cv.bitwise_and(
            white_glare,
            smooth_texture
        )

        highlight_exclusion = cv.bitwise_or(
            green_glare,
            smooth_white_glare
        )

        highlight_exclusion = cv.bitwise_and(
            highlight_exclusion,
            locally_bright
        )

        # --------------------------------------------------
        # GLOBAL BRIGHTNESS GLARE CHECK
        # --------------------------------------------------
        #
        # A large glossy reflection can form a broad bright plateau.
        # The centre of that plateau may NOT be much brighter than
        # its immediate neighbourhood, so a purely local-brightness
        # test misses it.
        #
        # Compare against the median brightness of the fruit itself.
        # On a healthy green orange, a reflection is typically much
        # brighter and less saturated than the surrounding peel.

        green_surface_values = value[
            inside_orange_mask > 0
        ]

        median_green_value = (
            float(np.median(green_surface_values))
            if green_surface_values.size > 0
            else 128.0
        )

        global_glare = np.zeros_like(
            value,
            dtype=np.uint8
        )

        sat_channel = hsv[:, :, 1]

        global_glare[
            (value >= min(245, median_green_value + 28))
            & (sat_channel <= 170)
        ] = 255

        # Very rough bright pixels may still be true fuzzy mould.
        # Keep those available to the mould detector instead of
        # automatically treating every bright low-saturation pixel
        # as reflection.
        very_rough = np.zeros_like(
            texture,
            dtype=np.uint8
        )

        very_rough[
            texture >= 48
        ] = 255

        global_glare = cv.bitwise_and(
            global_glare,
            cv.bitwise_not(
                very_rough
            )
        )

        highlight_exclusion = cv.bitwise_or(
            highlight_exclusion,
            global_glare
        )

        highlight_exclusion = cv.bitwise_and(
            highlight_exclusion,
            inside_orange_mask
        )

        # Fill the reflection interior and cover its immediate
        # smooth boundary without spreading far across the peel.
        highlight_exclusion = cv.morphologyEx(
            highlight_exclusion,
            cv.MORPH_CLOSE,
            np.ones(
                (5, 5),
                np.uint8
            ),
            iterations=1
        )

        highlight_exclusion = cv.dilate(
            highlight_exclusion,
            np.ones(
                (3, 3),
                np.uint8
            ),
            iterations=1
        )

        highlight_exclusion = cv.bitwise_and(
            highlight_exclusion,
            inside_orange_mask
        )


    # ======================================================
    # 11. DARK GREEN MOULD
    # ======================================================

    dark_green_mold = cv.inRange(
        hsv,
        np.array([30, 40, 25]),
        np.array([95, 180, 170])
    )

    # Must also be unusually dark compared with
    # the rest of this orange.
    dark_green_mold = cv.bitwise_and(
        dark_green_mold,
        relative_dark_mask
    )

        # ======================================================
    # NORMAL ORANGE MOULD = STRICT ROUGH MOULD ONLY
    # ======================================================
    #
    # The broad severe/recovery masks are intentionally sensitive
    # so a badly rotten orange can still be recovered. They must
    # NOT enter the normal ripe-orange path, otherwise pale rough
    # peel around the stem/top can become a large false defect.
    #
    # For an ordinary orange, white/gray/pale mould must also have
    # clearly rough local texture.

    normal_rough_texture = np.zeros_like(
        texture,
        dtype=np.uint8
    )

    normal_rough_texture[
        texture >= 22
    ] = 255

    normal_rough_texture = cv.dilate(
        normal_rough_texture,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    normal_white_gray_mold = cv.bitwise_and(
        white_gray_mold,
        normal_rough_texture
    )

    normal_pale_green_mold = cv.bitwise_and(
        pale_green_mold,
        normal_rough_texture
    )

    mold_mask = cv.bitwise_or(
        normal_white_gray_mold,
        normal_pale_green_mold
    )

    # IMPORTANT:
    # severe_mold_mask and recovered_mold_mask are NOT added here.
    # They remain available for the later severe-only branch.


    # ======================================================
    # EXPAND CONFIRMED MOULD INTO ADJACENT GREEN MOULD
    # ======================================================

    # Green / olive mould candidate
    green_mold_candidate = cv.inRange(
        hsv,
        np.array([25, 20, 25]),
        np.array([100, 200, 210])
    )

    # Existing CONFIRMED normal mould acts as the seed.
    # Use the stricter rough-texture mould masks so smooth pale
    # peel / stem glare cannot seed a large false green-mould area.
    mold_seed = cv.bitwise_or(
        normal_white_gray_mold,
        normal_pale_green_mold
    )

    mold_seed = cv.bitwise_or(
        mold_seed,
        dark_green_mold
    )

    # Expand seed slightly so neighbouring mould pixels
    # can be included.
    #
    # NOTE: kept deliberately small (was 9x9 x2 ~= 18px
    # reach). Near the bottom/rim of a round fruit, natural
    # shading from surface curvature can pass the loose
    # green_mold_candidate colour range even on healthy peel.
    # A large reach let confirmed mould "pull in" that shaded
    # healthy peel as if it were connected disease, adding a
    # halo of false defect beyond the real mould blob. 5x5 x1
    # (~4-5px reach) still bridges small gaps in genuine mould
    # texture without reaching into the shading gradient.
    grow_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    expanded_seed = cv.dilate(
        mold_seed,
        grow_kernel,
        iterations=1
    )

    # Only accept green pixels that are connected /
    # very close to already confirmed mould
    connected_green_mold = cv.bitwise_and(
        green_mold_candidate,
        expanded_seed
    )

    # Keep inside the orange only
    connected_green_mold = cv.bitwise_and(
        connected_green_mold,
        inside_orange_mask
    )


    # ======================================================
    # 12. BUILD MOULD MASK
    # ======================================================


    # Remove normal saturated green peel.
    mold_mask = cv.bitwise_and(
        mold_mask,
        cv.bitwise_not(
            healthy_green
        )
    )

    # Put dark green mould back.
    mold_mask = cv.bitwise_or(
        mold_mask,
        dark_green_mold
    )

    mold_mask = cv.bitwise_or(
        mold_mask,
        connected_green_mold
    )

    # On strongly green oranges, remove only smooth bright
    # reflection pixels. Rough white/gray mould remains.
    mold_mask = cv.bitwise_and(
        mold_mask,
        cv.bitwise_not(
            highlight_exclusion
        )
    )


    mold_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mold_mask = cv.morphologyEx(
        mold_mask,
        cv.MORPH_CLOSE,
        mold_kernel,
        iterations=2
    )

    mold_mask = cv.morphologyEx(
        mold_mask,
        cv.MORPH_OPEN,
        mold_kernel,
        iterations=1
    )


    # CRITICAL:
    # Mould must stay inside actual orange.
    mold_mask = cv.bitwise_and(
        mold_mask,
        inside_orange_mask
    )

    # ======================================================
    # 12b. CONSERVATIVE MOULD FILTER FOR GREEN / UNRIPE ORANGE
    # ======================================================
    #
    # Green/unripe oranges are glossy and often have strong
    # highlights and natural green shading. The general severe
    # mould recovery is useful for mature rotten oranges, but is
    # too permissive for a strongly green fruit.
    #
    # For green oranges only:
    # - disable broad severe/recovered mould contribution
    # - keep only rough white/gray mould
    # - keep locally-dark green mould
    # - reject smooth bright highlights
    # - remove small/thin text-like or watermark-like components
    #
    # Mature / ripe / shrivelled / severe orange conditions use
    # the original logic unchanged.

    green_safe_mold_mask = np.zeros_like(
        mold_mask
    )

    if is_green_orange:

        # --------------------------------------------------
        # STRICT GREEN / MIXED-COLOUR MOULD CONFIRMATION
        # --------------------------------------------------
        #
        # Healthy unripe oranges naturally contain pale yellow-green
        # transition patches and strong glossy peel texture. Those
        # must not be treated as mould.
        #
        # White/gray mould therefore needs:
        #   - low saturation
        #   - clearly rough texture
        #   - not a detected highlight
        #
        # Pale-green mould must also be locally darker than the
        # surrounding peel; bright ripening patches are rejected.

        green_rough_strict = np.zeros_like(
            texture,
            dtype=np.uint8
        )

        green_rough_strict[
            texture >= 30
        ] = 255

        green_white_low_sat = cv.inRange(
            hsv,
            np.array([0, 0, 70]),
            np.array([180, 75, 245])
        )

        green_white_mold = cv.bitwise_and(
            green_white_low_sat,
            green_rough_strict
        )

        green_white_mold = cv.bitwise_and(
            green_white_mold,
            cv.bitwise_not(
                highlight_exclusion
            )
        )

        green_pale_mold = cv.bitwise_and(
            pale_green_mold,
            green_rough_strict
        )

        green_pale_mold = cv.bitwise_and(
            green_pale_mold,
            relative_dark_mask
        )

        # Dark green mould already requires local darkness.
        green_dark_mold = dark_green_mold.copy()

        green_safe_mold_mask = cv.bitwise_or(
            green_white_mold,
            green_pale_mold
        )

        green_safe_mold_mask = cv.bitwise_or(
            green_safe_mold_mask,
            green_dark_mold
        )

        # Stay farther away from the curved fruit edge.
        green_safe_mold_mask = cv.bitwise_and(
            green_safe_mold_mask,
            surface_mask
        )

        green_safe_mold_mask = cv.bitwise_and(
            green_safe_mold_mask,
            cv.bitwise_not(
                calyx_mask
            )
        )

        # Clean very small isolated texture / watermark fragments.
        green_safe_mold_mask = cv.morphologyEx(
            green_safe_mold_mask,
            cv.MORPH_OPEN,
            np.ones(
                (3, 3),
                np.uint8
            ),
            iterations=1
        )

        green_safe_mold_mask = cv.morphologyEx(
            green_safe_mold_mask,
            cv.MORPH_CLOSE,
            np.ones(
                (3, 3),
                np.uint8
            ),
            iterations=1
        )

        green_clean = np.zeros_like(
            green_safe_mold_mask
        )

        green_contours, _ = cv.findContours(
            green_safe_mold_mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        green_fruit_pixels = max(
            1,
            cv.countNonZero(
                fruit_mask
            )
        )

        green_min_area = max(
            100,
            int(
                green_fruit_pixels
                * 0.0025
            )
        )

        for green_contour in green_contours:

            green_area = cv.contourArea(
                green_contour
            )

            if green_area < green_min_area:
                continue

            gx, gy, gw, gh = cv.boundingRect(
                green_contour
            )

            green_bbox_area = max(
                1,
                gw * gh
            )

            green_extent = (
                green_area
                / green_bbox_area
            )

            # Thin text/watermark strokes and long highlight edges
            # usually have very low extent. Real mould patches are
            # more region-like.
            if green_extent < 0.18:
                continue

            cv.drawContours(
                green_clean,
                [green_contour],
                -1,
                255,
                cv.FILLED
            )

        mold_mask = green_clean
        green_safe_mold_mask = green_clean

    # ======================================================
    # 13. NON-MOULD DEFECTS
    # ======================================================

    # confirmed_brown_mask was already created above
    # using brown colour + dark_difference > 10


    # ------------------------------------------------------
    # Expand confirmed brown rot into nearby brown areas
    # ------------------------------------------------------

    # Same reasoning as the mould seed growth above: a large
    # reach here let confirmed brown rot pull in shaded-but-
    # healthy peel near the fruit's rim/bottom as false "rot".
    brown_grow_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    expanded_brown_seed = cv.dilate(
        confirmed_brown_mask,
        brown_grow_kernel,
        iterations=1
    )

    # Only grow into pixels that are still brown-coloured
    connected_brown_rot = cv.bitwise_and(
        brown_mask,
        expanded_brown_seed
    )

    connected_brown_rot = cv.bitwise_and(
        connected_brown_rot,
        surface_mask
    )


    # ------------------------------------------------------
    # Very dark / black defects
    # ------------------------------------------------------

    confirmed_dark_mask = cv.bitwise_and(
        dark_mask,
        surface_mask
    )


    # ------------------------------------------------------
    # Combine brown + dark + strong local blemishes
    # ------------------------------------------------------

    non_mold_mask = cv.bitwise_or(
        confirmed_brown_mask,
        connected_brown_rot
    )

    non_mold_mask = cv.bitwise_or(
        non_mold_mask,
        confirmed_dark_mask
    )

    non_mold_mask = cv.bitwise_or(
        non_mold_mask,
        relative_dark_mask
    )


    # Stay inside safe fruit surface
    non_mold_mask = cv.bitwise_and(
        non_mold_mask,
        surface_mask
    )


    # Remove calyx / stem
    non_mold_mask = cv.bitwise_and(
        non_mold_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )
    # ======================================================
    # 14. COMBINE ALL DEFECTS
    # ======================================================

    defect_mask = cv.bitwise_or(
        non_mold_mask,
        mold_mask
    )


    # Final hard restriction:
    # anything outside actual fruit = impossible defect.
    defect_mask = cv.bitwise_and(
        defect_mask,
        inside_orange_mask
    )

    # Do NOT put severe mould back yet.
    # It is intentionally kept separate until the later
    # severe-coverage validation step.

    # Still keep normal defects within central expected fruit region
    defect_mask = cv.bitwise_and(
        defect_mask,
        central_mask
    )


    # ======================================================
    # 15. MORPHOLOGICAL CLEANING
    # ======================================================

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_CLOSE,
        np.ones(
            (5, 5),
            np.uint8
        ),
        iterations=2
    )


    # Keep again inside the orange after morphology.
    defect_mask = cv.bitwise_and(
        defect_mask,
        inside_orange_mask
    )


    # ======================================================
    # 15b. FILL FULLY-ENCLOSED HOLES
    # ------------------------------------------------------
    # A genuine defect blob can have internal brightness/
    # colour variation -- part of a scar can be noticeably
    # lighter than its rim. The confirmation gates above
    # (brown_local_dark, dark thresholds, etc.) are tuned to
    # reject shallow ambient shading at the fruit's outer
    # edge, but that same threshold can also reject a lighter
    # interior patch of a real defect, leaving a false "hole"
    # -- a ring/donut shape -- in the middle of a solid scar.
    #
    # A hole that is completely surrounded by already-
    # confirmed defect on every side is essentially never
    # untouched healthy peel; it is filled in here regardless
    # of why the gap formed. This only affects fully enclosed
    # interior holes -- it cannot expand the outer boundary of
    # any blob, so it can't reintroduce the earlier rim/shadow
    # over-detection.
    #
    # calyx_mask is re-subtracted afterward, since a
    # legitimate calyx sitting inside a defect blob is also an
    # enclosed "hole" and should stay excluded.
    # ======================================================

    hole_contours, hole_hierarchy = cv.findContours(
        defect_mask,
        cv.RETR_CCOMP,
        cv.CHAIN_APPROX_SIMPLE
    )

    if hole_hierarchy is not None:

        hole_hierarchy = hole_hierarchy[0]

        for i, h in enumerate(hole_hierarchy):

            parent = h[3]

            # A contour with a parent is an inner hole,
            # not an outer defect boundary.
            if parent == -1:
                continue

            cv.drawContours(
                defect_mask,
                [hole_contours[i]],
                -1,
                255,
                cv.FILLED
            )

    defect_mask = cv.bitwise_and(
        defect_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # ======================================================
    # FINAL SEVERE MOULD RECOVERY
    # ======================================================

    gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    # Gray / white / pale mould
    severe_mold_colour = cv.inRange(
        hsv,
        np.array([0, 0, 45]),
        np.array([180, 125, 240])
    )

    # Rough mould texture
    lap = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    texture = np.abs(
        lap
    )

    texture = np.clip(
        texture,
        0,
        255
    ).astype(np.uint8)

    _, rough_mask = cv.threshold(
        texture,
        10,
        255,
        cv.THRESH_BINARY
    )

    rough_mask = cv.dilate(
        rough_mask,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    severe_mold_mask = cv.bitwise_and(
        severe_mold_colour,
        rough_mask
    )


    # ======================================================
    # SAFE SEVERE-MOULD REGION
    # ======================================================
    #
    # Practical-handbook idea:
    # use erosion to remove uncertain boundary pixels and keep
    # the analysis on sure foreground rather than the RONI.
    #
    # The old 0.46 ellipse still included a white background
    # ring around the fruit. Rough texture at the fruit/background
    # boundary then became a false severe-mould contour.
    #
    # Build expected orange geometry, then ERODE it slightly.
    # Also keep the actual segmented peel contour so real peel
    # near the edge is not lost.

    roi_h, roi_w = severe_mold_mask.shape

    severe_region = np.zeros_like(
        severe_mold_mask
    )

    cv.ellipse(
        severe_region,
        (
            roi_w // 2,
            roi_h // 2
        ),
        (
            int(roi_w * 0.46),
            int(roi_h * 0.46)
        ),
        0,
        0,
        360,
        255,
        cv.FILLED
    )

    safe_margin = max(
        9,
        int(min(roi_h, roi_w) * 0.025)
    )

    safe_kernel_size = (
        safe_margin * 2 + 1
    )

    severe_inner_region = cv.erode(
        severe_region,
        np.ones(
            (
                safe_kernel_size,
                safe_kernel_size
            ),
            np.uint8
        ),
        iterations=1
    )

    # Preserve actual segmented orange peel while excluding
    # the ellipse/background ring.
    severe_safe_region = cv.bitwise_or(
        severe_inner_region,
        raw_contour_mask
    )

    severe_mold_mask = cv.bitwise_and(
        severe_mold_mask,
        severe_safe_region
    )

    severe_mold_mask = cv.morphologyEx(
        severe_mold_mask,
        cv.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )


    # ======================================================
    # SEVERE-MOULD CANDIDATE COVERAGE
    # ======================================================
    #
    # severe_mold_mask is intentionally broad. It must NOT be
    # merged into a normal orange merely because some pale/rough
    # peel exists around the stem or under uneven lighting.
    #
    # Measure its coverage first. The tested truly severe mould
    # case is around 37%, so 30% keeps that case while preventing
    # ordinary oranges from entering the severe path.

    severe_candidate_fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    severe_candidate_pixels = cv.countNonZero(
        severe_mold_mask
    )

    severe_candidate_ratio = (
        severe_candidate_pixels / severe_candidate_fruit_pixels
        if severe_candidate_fruit_pixels > 0
        else 0.0
    )

    severe_candidate_confirmed = (
        severe_candidate_ratio >= 0.30
        and not is_green_orange
    )

    # Keep the normal path clean here.
    # The confirmed severe mask is re-added near the end only
    # if the severe condition remains valid.
    effective_fruit_mask = fruit_mask.copy()

    defect_mask = cv.bitwise_and(
        defect_mask,
        inside_orange_mask
    )


    # ======================================================
    # 16. REMOVE TINY NOISE
    # ======================================================

    cleaned_mask = np.zeros_like(
        defect_mask
    )

    defect_contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in defect_contours:

        area = cv.contourArea(
            contour
        )

        # Keep actual visible defects.
        if area > 40:

            cv.drawContours(
                cleaned_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )


    defect_mask = cleaned_mask


    # Final guarantee:
    # NO defect outside detected fruit boundary.
    defect_mask = cv.bitwise_and(
        defect_mask,
        inside_orange_mask
    )


    # ======================================================
    # 16b. REMOVE FALSE DEFECTS TOUCHING ROI/BBOX BOUNDARY
    # ------------------------------------------------------
    # Corner/edge "defects" caused by background bleeding
    # into the YOLO crop corners. These blobs touch the
    # rectangular crop edge, not just the fruit's own rounded
    # edge, so they must be filtered by the ROI boundary,
    # not the fruit contour.
    #
    # IMPORTANT: use border_overlap_ratio here, NOT the plain
    # bounding-box check. In a tight YOLO crop the orange
    # (and a large real defect like mould covering much of
    # it) can have a bounding box that grazes the crop edge
    # even though almost none of its pixels are actually
    # there. The plain check would then discard the entire
    # real defect. The ratio check only drops a component
    # when most of ITS OWN pixels sit in the border strip --
    # true of small corner-bleed artifacts, false of a large
    # interior mould patch.
    #
    # FIX: that ratio check alone is not enough -- a large
    # background blob (like the one that leaked through the
    # peel_mask fusion bug) extends well past the border
    # strip, so less than border_overlap_ratio of ITS pixels
    # sit in the strip and it used to survive. kill_corner_blocks
    # adds an unconditional pre-check: any large component
    # whose bounding box hugs an actual rectangular ROI
    # corner (touches two adjacent edges) is dropped outright,
    # since real surface defects on a round fruit never form
    # that shape.
    # ======================================================

    defect_mask = remove_border_components(
        defect_mask,
        margin=10,
        min_area=40,
        border_overlap_ratio=0.5,
        kill_corner_blocks=True,
        kill_flush_rectangles=True
    )

    # ======================================================
    # 16b-2. FINAL GREEN / UNRIPE ORANGE SAFETY GATE
    # ======================================================
    #
    # For strongly green oranges, keep ONLY high-confidence
    # surface damage. This removes:
    #   - glossy pale-green highlights
    #   - normal peel colour variation
    #   - bottom curvature/contact shadow
    #
    # This block runs ONLY for green/unripe oranges, so mature,
    # shrivelled and severely mouldy orange logic is unchanged.

    if is_green_orange:

        green_h, green_w = fruit_mask.shape[:2]

        green_inner_distance = max(
            16,
            int(min(green_h, green_w) * 0.08)
        )

        green_inner_mask = np.zeros_like(
            fruit_mask
        )

        green_inner_mask[
            distance >= green_inner_distance
        ] = 255

        # Bright green/yellow/orange transition peel is normal
        # on an unripe orange. Exclude it from brown/mould defects.
        healthy_transition = cv.inRange(
            hsv,
            np.array([5, 50, 145]),
            np.array([95, 255, 255])
        )

        # High-confidence brown damage only.
        green_brown_damage = cv.bitwise_or(
            confirmed_brown_mask,
            connected_brown_rot
        )

        darker_brown_only = np.zeros_like(
            value,
            dtype=np.uint8
        )

        darker_brown_only[
            value <= 150
        ] = 255

        green_brown_damage = cv.bitwise_and(
            green_brown_damage,
            darker_brown_only
        )

        green_brown_damage = cv.bitwise_and(
            green_brown_damage,
            cv.bitwise_not(
                healthy_transition
            )
        )

        green_brown_damage = cv.bitwise_and(
            green_brown_damage,
            green_inner_mask
        )

        # High-confidence very dark / black damage
        strong_green_dark = np.zeros_like(
            value,
            dtype=np.uint8
        )

        strong_green_dark[
            (value <= 70)
            & (dark_difference >= 24)
        ] = 255

        strong_green_dark = cv.bitwise_and(
            strong_green_dark,
            green_inner_mask
        )

        # A healthy green orange (and any leaf entering the YOLO crop)
        # can contain very dark, highly saturated green pixels. Those
        # must not be treated as generic black/dark fruit damage.
        # Genuine dark-green mould is handled separately below by the
        # stricter green_safe_mold_mask path.
        strong_green_dark = cv.bitwise_and(
            strong_green_dark,
            cv.bitwise_not(
                healthy_green
            )
        )

        # High-confidence mould, with glare removed again
        green_mould_damage = cv.bitwise_and(
            green_safe_mold_mask,
            cv.bitwise_not(
                highlight_exclusion
            )
        )

        green_mould_damage = cv.bitwise_and(
            green_mould_damage,
            cv.bitwise_not(
                healthy_transition
            )
        )

        green_mould_damage = cv.bitwise_and(
            green_mould_damage,
            green_inner_mask
        )

        green_final_defect = cv.bitwise_or(
            green_brown_damage,
            strong_green_dark
        )

        green_final_defect = cv.bitwise_or(
            green_final_defect,
            green_mould_damage
        )

        green_final_defect = cv.bitwise_and(
            green_final_defect,
            cv.bitwise_not(
                calyx_mask
            )
        )

        green_final_defect = cv.morphologyEx(
            green_final_defect,
            cv.MORPH_OPEN,
            np.ones(
                (3, 3),
                np.uint8
            ),
            iterations=1
        )

        # Extra geometry guard for green/unripe oranges.
        # The fruit is normally the large central rounded object in the
        # YOLO crop; leaves are commonly attached at the top/side. Restrict
        # green-orange defect candidates to a conservative central ellipse
        # so leaf shadows/leaf tissue cannot become red defect regions.
        green_central_guard = np.zeros_like(
            green_final_defect
        )

        cv.ellipse(
            green_central_guard,
            (
                green_w // 2,
                green_h // 2
            ),
            (
                max(5, int(green_w * 0.43)),
                max(5, int(green_h * 0.43))
            ),
            0,
            0,
            360,
            255,
            cv.FILLED
        )

        green_final_defect = cv.bitwise_and(
            green_final_defect,
            green_central_guard
        )

        green_component_mask = np.zeros_like(
            green_final_defect
        )

        green_components, _ = cv.findContours(
            green_final_defect,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        green_fruit_pixels_final = max(
            1,
            cv.countNonZero(
                fruit_mask
            )
        )

        green_component_min_area = max(
            80,
            int(
                green_fruit_pixels_final
                * 0.004
            )
        )

        for green_component in green_components:

            green_area = cv.contourArea(
                green_component
            )

            if green_area < green_component_min_area:
                continue

            gx, gy, gw, gh = cv.boundingRect(
                green_component
            )

            green_bbox_area = max(
                1,
                gw * gh
            )

            green_extent = (
                green_area / green_bbox_area
            )

            green_aspect = max(
                gw / max(1, gh),
                gh / max(1, gw)
            )

            if green_extent < 0.28:
                continue

            if green_aspect > 5.0:
                continue

            cv.drawContours(
                green_component_mask,
                [green_component],
                -1,
                255,
                cv.FILLED
            )

        defect_mask = green_component_mask

# ======================================================
# 16c. ORANGE WRINKLE / SHRIVEL / DRY PEEL DETECTION
# ======================================================

    if is_green_orange:

        # Wrinkle/dry-peel analysis is an overripe-texture cue.
        # Do not use it on a strongly green/unripe orange.
        wrinkle_mask = np.zeros_like(
            fruit_mask
        )

    else:

        wrinkle_mask = detect_orange_wrinkles(
            roi,
            fruit_mask,
            calyx_mask
        )

    # Use stronger inner surface mask for texture defects
    # so fruit-edge/background artifacts are excluded.
    wrinkle_mask = cv.bitwise_and(
        wrinkle_mask,
        surface_mask
    )

    # Bright smooth reflection is not wrinkle/dry peel.
    # This only has an effect on strongly green oranges because
    # highlight_exclusion is otherwise an all-zero mask.
    wrinkle_mask = cv.bitwise_and(
        wrinkle_mask,
        cv.bitwise_not(
            highlight_exclusion
        )
    )

    # Combine colour/mould + texture defects
    defect_mask = cv.bitwise_or(
        defect_mask,
        wrinkle_mask
    )

    # Keep final result inside actual orange
    defect_mask = cv.bitwise_and(
        defect_mask,
        inside_orange_mask
    )

    # Clean again after wrinkles are added
    defect_mask = remove_border_components(
        defect_mask,
        margin=10,
        min_area=40,
        border_overlap_ratio=0.5,
        kill_corner_blocks=True,
        kill_flush_rectangles=True
    )

    # ======================================================
    # FINAL STRICT ORANGE SURFACE CLIP
    # ======================================================

    # Measure how far every fruit pixel is
    # from the actual fruit boundary.
    final_distance = cv.distanceTransform(
        fruit_mask,
        cv.DIST_L2,
        5
    )

    h, w = fruit_mask.shape

    # Ignore only the outer ~4% of fruit.
    # Strong enough to remove bottom shadow/edge artifacts,
    # but less aggressive than the 6% surface_mask.
    final_margin = max(
        12,
        int(min(h, w) * 0.04)
    )

    final_surface_mask = np.zeros_like(
        fruit_mask
    )

    final_surface_mask[
        final_distance >= final_margin
    ] = 255


    # Final defect must be safely inside orange
    defect_mask = cv.bitwise_and(
        defect_mask,
        final_surface_mask
    )


    # Remove actual calyx/stem
    defect_mask = cv.bitwise_and(
        defect_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # ======================================================
    # 16e. SMOOTH DEFECT BOUNDARY (cosmetic only)
    # ------------------------------------------------------
    # This must run AFTER every hard clipping stage above --
    # including FINAL STRICT ORANGE SURFACE CLIP -- not
    # before it. That clip intersects the mask with
    # final_surface_mask, a crisp distance-transform boundary
    # that runs close to the actual fruit edge; if smoothing
    # happens earlier, this later intersection re-imposes the
    # same hard edge and erases the smoothing right where the
    # defect meets the fruit boundary (e.g. the bottom of a
    # patch that reaches near the crop edge). Running it last
    # means every remaining edge -- both the organic
    # colour-based edge and the distance-clip edge -- gets
    # rounded off together.
    #
    # Still purely cosmetic: it only reshapes the boundary of
    # regions already present, so it can't materially change
    # defect_percentage. Re-clip against inside_orange_mask
    # and the calyx afterward (not final_surface_mask, or the
    # smoothing would be undone again) so blurring can't leak
    # defect pixels outside the fruit or into the calyx notch.
    # ======================================================

    defect_mask = smooth_mask_boundary(
        defect_mask,
        blur_ksize=9,
        close_ksize=5
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        inside_orange_mask
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # ======================================================
    # FINAL: PUT SEVERE MOULD BACK AFTER ALL STRICT CLIPPING
    # ======================================================
    #
    # Important:
    # Earlier stages intentionally clip normal defects to
    # inside_orange_mask / final_surface_mask to remove
    # background and edge shadows.
    #
    # A severely mouldy orange can lose its normal orange
    # colour, so part of the mould may lie outside those
    # saturation-based masks. Therefore the severe mould mask
    # must be re-added HERE, after every strict clipping step.

    final_severe_mold = severe_mold_mask.copy()

    # Keep only expected central fruit geometry.
    final_severe_mold = cv.bitwise_and(
        final_severe_mold,
        severe_safe_region
    )

    # Never count the calyx/stem as mould.
    final_severe_mold = cv.bitwise_and(
        final_severe_mold,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # Join nearby rough mould pixels into larger visible areas.
    final_severe_mold = cv.morphologyEx(
        final_severe_mold,
        cv.MORPH_CLOSE,
        np.ones(
            (9, 9),
            np.uint8
        ),
        iterations=2
    )

    # Slightly expand confirmed mould so the whole fuzzy patch
    # is counted instead of only its strongest texture pixels.
    final_severe_mold = cv.dilate(
        final_severe_mold,
        np.ones(
            (5, 5),
            np.uint8
        ),
        iterations=1
    )

    # Remove tiny isolated noise.
    severe_clean = np.zeros_like(
        final_severe_mold
    )

    severe_contours, _ = cv.findContours(
        final_severe_mold,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    severe_min_area = max(
        60,
        int(final_severe_mold.size * 0.001)
    )

    for contour in severe_contours:

        if cv.contourArea(contour) >= severe_min_area:

            cv.drawContours(
                severe_clean,
                [contour],
                -1,
                255,
                cv.FILLED
            )

    final_severe_mold = severe_clean

    # Broad severe-mould recovery is for mature/severely rotten
    # oranges. On a strongly green/unripe orange, keep only the
    # conservative mould mask built above.
    if is_green_orange:

        final_severe_mold = cv.bitwise_and(
            final_severe_mold,
            green_safe_mold_mask
        )

    # Do not let smooth green-fruit reflections inflate the
    # confirmed severe-mould ratio or get re-added later.
    final_severe_mold = cv.bitwise_and(
        final_severe_mold,
        cv.bitwise_not(
            highlight_exclusion
        )
    )

    # ======================================================
    # SPECIAL SEVERE-MOULD COMPLETION
    # ======================================================
    #
    # IMPORTANT:
    # This branch is deliberately isolated from the normal
    # Orange algorithm.
    #
    # It activates ONLY when at least 30% of the fruit has
    # already been confirmed as mould by the existing detector.
    #
    # Therefore:
    # - healthy orange: unchanged
    # - green / unripe orange: unchanged
    # - minor defect orange: unchanged
    # - moderate damaged orange: unchanged
    # - only a severely mouldy orange enters this block
    #
    # Once activated, connected low-saturation white/gray
    # pixels are recovered by geodesic growth from the already
    # confirmed mould. The growth cannot jump across unrelated
    # healthy peel because every new pixel must belong to the
    # pale mould candidate AND remain connected to the seed.

    confirmed_mold_seed = cv.bitwise_or(
        mold_mask,
        final_severe_mold
    )

    trigger_fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    confirmed_mold_pixels = cv.countNonZero(
        confirmed_mold_seed
    )

    confirmed_mold_ratio = (
        confirmed_mold_pixels / trigger_fruit_pixels
        if trigger_fruit_pixels > 0
        else 0.0
    )

    # Much stricter than the previous 12% trigger.
    # The tested severe mould image is ~37%, so it enters;
    # ordinary orange conditions stay on the original path.
    is_severe_orange = (
        severe_candidate_confirmed
        and confirmed_mold_ratio >= 0.30
        and not is_green_orange
    )

    if is_severe_orange:

        # --------------------------------------------------
        # SAFE SEVERE-MOULD REGION
        # --------------------------------------------------
        #
        # ellipse_mask is only expected fruit geometry and can
        # still contain a white background band around the fruit.
        #
        # For severe cases only, allow recovery just a small
        # distance outside the already segmented fruit surface.
        # This helps recover fuzzy mould without reaching the
        # outer background ring.

        # ==================================================
        # STRICT SURE-FOREGROUND REGION FOR SEVERE CASE ONLY
        # ==================================================
        #
        # The previous severe_safe_region still preserved the
        # raw fruit contour, so rough pixels exactly on the
        # fruit/background boundary could survive as a large
        # red ring.
        #
        # Use the already-computed inside_orange_mask instead.
        # It comes from distance transform, so uncertain outer
        # boundary pixels are removed. Dilate it slightly to
        # recover genuine mould close to the peel edge.
        #
        # IMPORTANT:
        # This runs ONLY when confirmed_mold_ratio >= 0.30,
        # so healthy / green / minor / moderate orange cases
        # keep their previous behaviour.

        severe_allowed_region = cv.dilate(
            inside_orange_mask,
            np.ones(
                (11, 11),
                np.uint8
            ),
            iterations=1
        )

        severe_allowed_region = cv.bitwise_and(
            severe_allowed_region,
            ellipse_mask
        )

        # Remove severe-mask leakage near the expected ellipse edge.
        final_severe_mold = cv.bitwise_and(
            final_severe_mold,
            severe_allowed_region
        )

        # White / gray / pale fuzzy mould candidate.
        severe_pale_candidate = cv.inRange(
            hsv,
            np.array([0, 0, 45]),
            np.array([180, 130, 255])
        )

        severe_pale_candidate = cv.bitwise_and(
            severe_pale_candidate,
            severe_allowed_region
        )

        severe_pale_candidate = cv.bitwise_and(
            severe_pale_candidate,
            cv.bitwise_not(
                highlight_exclusion
            )
        )

        # Never recover the detected calyx / stem.
        severe_pale_candidate = cv.bitwise_and(
            severe_pale_candidate,
            cv.bitwise_not(
                calyx_mask
            )
        )

        # --------------------------------------------------
        # Geodesic / connected-region growth
        # --------------------------------------------------
        #
        # Start from confirmed mould and grow only through
        # connected pale mould candidate pixels inside the
        # safe fruit neighbourhood.

        growth = cv.bitwise_and(
            confirmed_mold_seed,
            severe_allowed_region
        )

        growth_kernel = np.ones(
            (5, 5),
            np.uint8
        )

        for _ in range(25):

            expanded = cv.dilate(
                growth,
                growth_kernel,
                iterations=1
            )

            expanded = cv.bitwise_and(
                expanded,
                severe_pale_candidate
            )

            updated = cv.bitwise_or(
                growth,
                expanded
            )

            if np.array_equal(
                updated,
                growth
            ):
                break

            growth = updated

        connected_severe_pale = cv.bitwise_and(
            growth,
            severe_pale_candidate
        )

        connected_severe_pale = cv.morphologyEx(
            connected_severe_pale,
            cv.MORPH_CLOSE,
            np.ones(
                (5, 5),
                np.uint8
            ),
            iterations=1
        )

        connected_severe_pale = cv.bitwise_and(
            connected_severe_pale,
            severe_allowed_region
        )

        connected_severe_pale = cv.bitwise_and(
            connected_severe_pale,
            cv.bitwise_not(
                calyx_mask
            )
        )

        # Add only connected pale mould completion.
        final_severe_mold = cv.bitwise_or(
            final_severe_mold,
            connected_severe_pale
        )

    # Re-add severe mould ONLY for a genuinely severe case.
    #
    # This is the key isolation fix:
    # normal ripe / minor-defect / shrivelled oranges keep the
    # normal detector result and are not contaminated by the
    # broad severe-mould thresholds.
    if not is_severe_orange:

        final_severe_mold = np.zeros_like(
            final_severe_mold
        )

    defect_mask = cv.bitwise_or(
        defect_mask,
        final_severe_mold
    )

    # The recovered mould is still part of the fruit surface,
    # so include it in the denominator too.
    effective_fruit_mask = cv.bitwise_or(
        fruit_mask,
        final_severe_mold
    )

    # ======================================================
    # 17. CALCULATE DEFECT PERCENTAGE
    # ======================================================

    fruit_pixels = cv.countNonZero(
        effective_fruit_mask
    )

    defect_pixels = cv.countNonZero(
        defect_mask
    )

    if fruit_pixels == 0:

        defect_percentage = 0.0

    else:

        defect_percentage = (
            defect_pixels
            / fruit_pixels
        ) * 100

        defect_percentage = min(
            defect_percentage,
            100.0
        )


    # ======================================================
    # 18. DRAW DEFECT CONTOURS ONLY
    # ======================================================
    #
    # DISPLAY ONLY:
    # Remove false red contours that follow the outer orange
    # boundary / background. The real defect percentage and
    # ripeness are NOT changed here.

    output = roi.copy()

    # ------------------------------------------------------
    # Build a boundary band around the outer fruit edge
    # ------------------------------------------------------

    display_distance = cv.distanceTransform(
        fruit_mask,
        cv.DIST_L2,
        5
    )

    display_margin = max(
        18,
        int(min(h, w) * 0.07)
    )

    display_inner_mask = np.zeros_like(
        fruit_mask
    )

    display_inner_mask[
        display_distance >= display_margin
    ] = 255

    # Start from the real defect mask, but only for display.
    #
    # Directly remove the outer fruit-boundary band.
    # This guarantees red contours cannot follow the
    # orange outline/background edge, even when that edge
    # is connected to a real mould component.
    #
    # DISPLAY ONLY:
    # defect_percentage and ripeness remain unchanged.
    display_defect_mask = cv.bitwise_and(
        defect_mask,
        display_inner_mask
    )

    display_defect_mask = cv.bitwise_and(
        display_defect_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # ------------------------------------------------------
    # Remove connected components that mainly sit on the
    # outer fruit boundary. These are the large red arcs seen
    # around the orange, not true internal defect regions.
    # ------------------------------------------------------

    boundary_band = cv.subtract(
        fruit_mask,
        display_inner_mask
    )

    cleaned_display_mask = np.zeros_like(
        display_defect_mask
    )

    display_contours, _ = cv.findContours(
        display_defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    # Bounding box of the detected fruit surface.
    fruit_x, fruit_y, fruit_w, fruit_h = cv.boundingRect(
        fruit_mask
    )

    for contour in display_contours:

        area = cv.contourArea(
            contour
        )

        if area <= 40:
            continue

        component_mask = np.zeros_like(
            display_defect_mask
        )

        cv.drawContours(
            component_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )

        component_pixels = cv.countNonZero(
            component_mask
        )

        boundary_pixels = cv.countNonZero(
            cv.bitwise_and(
                component_mask,
                boundary_band
            )
        )

        boundary_ratio = (
            boundary_pixels / component_pixels
            if component_pixels > 0
            else 1.0
        )

        # --------------------------------------------------
        # Reject OUTER-RING / FRUIT-BOUNDARY artifacts
        # --------------------------------------------------
        #
        # The false orange boundary appears as a very large,
        # thin component spanning most of the fruit. A true
        # defect is normally more compact.
        #
        # This filter is applied ONLY to Orange's final mask.

        cx, cy, cw, ch = cv.boundingRect(
            contour
        )

        span_w = (
            cw / fruit_w
            if fruit_w > 0
            else 1.0
        )

        span_h = (
            ch / fruit_h
            if fruit_h > 0
            else 1.0
        )

        bbox_area = max(
            1,
            cw * ch
        )

        fill_ratio = (
            component_pixels
            / bbox_area
        )

        huge_thin_ring = (
            span_w >= 0.72
            and
            span_h >= 0.72
            and
            fill_ratio <= 0.35
        )

        wide_edge_arc = (
            span_w >= 0.70
            and
            boundary_ratio >= 0.25
            and
            fill_ratio <= 0.40
        )

        tall_edge_arc = (
            span_h >= 0.70
            and
            boundary_ratio >= 0.25
            and
            fill_ratio <= 0.40
        )

        # Existing boundary test, plus explicit ring/arc tests.
        if (
            boundary_ratio >= 0.45
            or huge_thin_ring
            or wide_edge_arc
            or tall_edge_arc
        ):
            continue

        cv.drawContours(
            cleaned_display_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )

    # Final display mask must stay on the fruit.
    cleaned_display_mask = cv.bitwise_and(
        cleaned_display_mask,
        fruit_mask
    )

    cleaned_display_mask = cv.bitwise_and(
        cleaned_display_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # ======================================================
    # FINAL ROI-EDGE ARTIFACT REMOVAL
    # ======================================================
    #
    # Remove small false red fragments that touch the outer
    # ROI/crop edge. These are background or boundary artifacts,
    # not actual defects on the orange.
    #
    # This affects Orange only.

    edge_margin = max(
        8,
        int(min(h, w) * 0.025)
    )

    edge_clean_mask = np.zeros_like(
        cleaned_display_mask
    )

    edge_contours, _ = cv.findContours(
        cleaned_display_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for edge_contour in edge_contours:

        area = cv.contourArea(
            edge_contour
        )

        if area <= 40:
            continue

        x_ec, y_ec, w_ec, h_ec = cv.boundingRect(
            edge_contour
        )

        touches_roi_edge = (
            x_ec <= edge_margin
            or
            y_ec <= edge_margin
            or
            x_ec + w_ec >= w - edge_margin
            or
            y_ec + h_ec >= h - edge_margin
        )

        if touches_roi_edge:
            continue

        cv.drawContours(
            edge_clean_mask,
            [edge_contour],
            -1,
            255,
            cv.FILLED
        )

    cleaned_display_mask = edge_clean_mask

    # ======================================================
    # FINAL ORANGE DEFECT MASK + PERCENTAGE
    # ======================================================
    #
    # Keep the cleaned colour/mould mask so false outer-orange
    # boundary artifacts stay removed.
    #
    # BUT add back the wrinkle/shrivel mask separately.
    # wrinkle_mask was already restricted to surface_mask earlier,
    # so it does not contain the large background ring.
    #
    # This is important for shrivelled oranges: the previous
    # cleanup removed most wrinkle evidence and caused a badly
    # shrivelled orange to become only ~1% defect / Ripe.

    if is_green_orange:

        safe_wrinkle_mask = np.zeros_like(
            fruit_mask
        )

    else:

        wrinkle_candidate = cv.bitwise_and(
            wrinkle_mask,
            surface_mask
        )

        wrinkle_candidate = cv.bitwise_and(
            wrinkle_candidate,
            cv.bitwise_not(
                calyx_mask
            )
        )

        # --------------------------------------------------
        # SHRIVEL CONFIRMATION
        # --------------------------------------------------
        #
        # Normal ripe oranges can have rough/pale texture near
        # the stem and top shoulder. A genuinely shrivelled fruit
        # should show wrinkle evidence across the main body too.
        #
        # Therefore require enough wrinkle area OUTSIDE the upper
        # stem zone before allowing wrinkle/dry-peel defects into
        # the final result.

        fx, fy, fw, fh = cv.boundingRect(
            fruit_mask
        )

        top_exclusion = np.zeros_like(
            fruit_mask
        )

        top_limit = min(
            fruit_mask.shape[0],
            fy + int(fh * 0.30)
        )

        top_exclusion[
            fy:top_limit,
            fx:fx + fw
        ] = 255

        body_region = cv.bitwise_and(
            surface_mask,
            cv.bitwise_not(
                top_exclusion
            )
        )

        body_wrinkle = cv.bitwise_and(
            wrinkle_candidate,
            body_region
        )

        body_pixels = cv.countNonZero(
            body_region
        )

        body_wrinkle_pixels = cv.countNonZero(
            body_wrinkle
        )

        body_wrinkle_ratio = (
            body_wrinkle_pixels / body_pixels
            if body_pixels > 0
            else 0.0
        )

        # Real shrivel/dried peel usually loses saturation across
        # the fruit surface. A healthy ripe orange can still have
        # strong normal peel texture, especially around the stem,
        # so texture alone is not enough.
        fruit_saturation_values = hsv[:, :, 1][
            fruit_mask > 0
        ]

        median_fruit_saturation = (
            float(np.median(fruit_saturation_values))
            if fruit_saturation_values.size > 0
            else 255.0
        )

        allow_wrinkle_defect = (
            body_wrinkle_ratio >= 0.035
            and median_fruit_saturation <= 175.0
        )

        if allow_wrinkle_defect:

            safe_wrinkle_mask = wrinkle_candidate

        else:

            safe_wrinkle_mask = np.zeros_like(
                fruit_mask
            )

    final_defect_mask = cv.bitwise_or(
        cleaned_display_mask,
        safe_wrinkle_mask
    )

    # Final result must stay on the actual orange.
    final_defect_mask = cv.bitwise_and(
        final_defect_mask,
        fruit_mask
    )

    final_defect_mask = cv.bitwise_and(
        final_defect_mask,
        cv.bitwise_not(
            calyx_mask
        )
    )

    # Use the effective fruit surface for percentage.
    # For severe mould this includes recovered mould surface;
    # for normal/shrivelled oranges it is essentially fruit_mask.
    final_fruit_pixels = cv.countNonZero(
        effective_fruit_mask
    )

    final_defect_pixels = cv.countNonZero(
        final_defect_mask
    )

    defect_percentage = (
        (
            final_defect_pixels
            / final_fruit_pixels
        ) * 100.0
        if final_fruit_pixels > 0
        else 0.0
    )

    defect_percentage = min(
        defect_percentage,
        100.0
    )

    # ======================================================
    # DISPLAY-ONLY ROI EDGE CLEANUP
    # ======================================================
    #
    # Small red fragments that touch the YOLO crop border are
    # display artifacts, not useful defect contours.
    #
    # Keep the REAL final_defect_mask and percentage unchanged.
    # Only remove a thin strip from the mask used for drawing.

    display_mask = final_defect_mask.copy()

    edge_strip = max(
        6,
        int(min(h, w) * 0.02)
    )

    display_mask[:edge_strip, :] = 0
    display_mask[h - edge_strip:, :] = 0
    display_mask[:, :edge_strip] = 0
    display_mask[:, w - edge_strip:] = 0

    contours, _ = cv.findContours(
        display_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv.contourArea(contour) > 40:

            cv.drawContours(
                output,
                [contour],
                -1,
                (0, 0, 255),
                2
            )


    return (
        output,
        final_defect_mask,
        defect_percentage
    )



# ==========================================================
# Mango Defect Detection
# ==========================================================

def detect_mango_defect(roi, fruit_mask_hint=None):
    """
    Detect visible mango surface defects while reducing false positives
    from normal green / yellow / orange / red mango skin.

    Handles, conservatively:
    - brown / black rot
    - dark bruising
    - clustered black spots / anthracnose-like damage
    - gray / white / pale mould
    - olive / dark-green damaged patches
    - dry / rough shrivelled regions

    Natural colour transitions, red blush, small lenticels, stem contact,
    outer shadows and tight YOLO crop boundaries are suppressed where
    possible.
    """

    if roi is None or roi.size == 0:
        return (
            roi.copy() if roi is not None else roi,
            np.zeros((1, 1), dtype=np.uint8),
            0.0
        )

    # ======================================================
    # 1. PREPROCESSING
    # ======================================================

    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    h, w = value.shape

    # ======================================================
    # 2. MANGO-SPECIFIC FRUIT MASK
    # ======================================================
    # Mango peel may be green, yellow, orange or red. Build the
    # fruit boundary mainly from chromatic peel pixels, then fill
    # interior dark defects geometrically. This is safer than using
    # one generic saturation/value rule for all fruit types.

    chromatic_seed = np.zeros_like(value, dtype=np.uint8)

    plausible_mango_hue = (
        (hue <= 100)
        | (hue >= 170)
    )

    chromatic_seed[
        plausible_mango_hue
        & (saturation >= 28)
        & (value >= 28)
    ] = 255

    # Strong blue/cyan is unlikely to be mango peel.
    blue_cyan = (
        (hue >= 90)
        & (hue <= 135)
        & (saturation >= 55)
    )

    chromatic_seed[
        blue_cyan
    ] = 0

    # Keep the literal ROI corners conservative. A mango is rounded,
    # so rectangular corner blobs are usually background leakage.
    chromatic_seed = suppress_bbox_corners(
        chromatic_seed,
        corner_ratio=0.08,
        min_corner_px=4
    )

    chromatic_seed = cv.morphologyEx(
        chromatic_seed,
        cv.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )

    chromatic_seed = cv.morphologyEx(
        chromatic_seed,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    contours, _ = cv.findContours(
        chromatic_seed,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    fruit_mask = np.zeros_like(
        value,
        dtype=np.uint8
    )

    if not contours:
        # Conservative fallback to the existing generic fruit mask.
        fruit_mask = get_fruit_mask(roi)
    else:
        # Prefer a large component that is also near the ROI centre.
        # This reduces the chance of selecting an isolated background
        # blob when a YOLO crop has extra space around the mango.
        cx0 = w / 2.0
        cy0 = h / 2.0
        diag = max(1.0, (w * w + h * h) ** 0.5)

        def contour_score(c):
            area = cv.contourArea(c)
            m = cv.moments(c)

            if m["m00"] != 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
            else:
                x, y, cw, ch = cv.boundingRect(c)
                cx = x + cw / 2.0
                cy = y + ch / 2.0

            centre_distance = (
                ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
                / diag
            )

            centre_weight = max(
                0.25,
                1.0 - centre_distance
            )

            return area * centre_weight

        largest = max(
            contours,
            key=contour_score
        )

        raw_mask = np.zeros_like(
            value,
            dtype=np.uint8
        )

        cv.drawContours(
            raw_mask,
            [largest],
            -1,
            255,
            cv.FILLED
        )

        # Mango is usually approximately convex / elliptical. The hull
        # recovers internal brown/black/mould areas that were missing from
        # the colour seed without changing any other fruit detector.
        hull = cv.convexHull(largest)

        hull_mask = np.zeros_like(
            value,
            dtype=np.uint8
        )

        cv.drawContours(
            hull_mask,
            [hull],
            -1,
            255,
            cv.FILLED
        )

        # Restrict hull filling using a fitted ellipse where possible so
        # a hand, table shadow or rectangular crop corner is less likely
        # to become fruit.
        if len(largest) >= 5:
            ellipse = cv.fitEllipse(largest)
            center, axes, angle = ellipse

            expanded_ellipse = (
                center,
                (
                    axes[0] * 1.04,
                    axes[1] * 1.04
                ),
                angle
            )

            ellipse_mask = np.zeros_like(
                value,
                dtype=np.uint8
            )

            cv.ellipse(
                ellipse_mask,
                expanded_ellipse,
                255,
                cv.FILLED
            )

            hull_mask = cv.bitwise_and(
                hull_mask,
                ellipse_mask
            )

        # Preserve the measured fruit. Do NOT blindly union the whole
        # convex hull: on a tight YOLO crop that can fill rectangular
        # corner/background wedges and later make them look like dark rot.
        #
        # Only recover hull-added regions that are fully interior to the ROI.
        hull_gap = cv.bitwise_and(
            hull_mask,
            cv.bitwise_not(raw_mask)
        )

        hull_gap = remove_border_components(
            hull_gap,
            margin=max(2, int(min(h, w) * 0.015)),
            min_area=12
        )

        fruit_mask = cv.bitwise_or(
            raw_mask,
            hull_gap
        )

        # Keep the recovered mask inside the fitted mango geometry when
        # ellipse fitting was available.
        if len(largest) >= 5:
            fruit_mask = cv.bitwise_and(
                fruit_mask,
                ellipse_mask
            )

    # ------------------------------------------------------
    # Optional external segmentation hint
    # ------------------------------------------------------
    # The overall pipeline already segments each fruit before calling the
    # defect module. For Mango only, accept that mask as a stronger boundary
    # hint when it exactly matches this raw ROI. This prevents background,
    # leaves, fingers and crop-corner wedges from being reintroduced by the
    # mango-specific colour mask. The standalone detector still works because
    # this argument is optional.
    if (
        isinstance(fruit_mask_hint, np.ndarray)
        and fruit_mask_hint.shape[:2] == fruit_mask.shape[:2]
    ):
        hint = np.where(
            fruit_mask_hint > 0,
            255,
            0
        ).astype(np.uint8)

        hint = cv.morphologyEx(
            hint,
            cv.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=1
        )

        hint_contours, _ = cv.findContours(
            hint,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        if hint_contours:
            hint_largest = max(
                hint_contours,
                key=cv.contourArea
            )

            hint_clean = np.zeros_like(
                hint
            )

            cv.drawContours(
                hint_clean,
                [hint_largest],
                -1,
                255,
                cv.FILLED
            )

            hint_pixels = cv.countNonZero(
                hint_clean
            )

            roi_pixels = h * w

            # Ignore obviously collapsed/full-rectangle hints.
            if (
                hint_pixels >= roi_pixels * 0.18
                and hint_pixels <= roi_pixels * 0.94
            ):
                fruit_mask = hint_clean

    # ======================================================
    # 2b. RECOVER LARGE GRAY / BLACK EDGE ROT
    # ======================================================
    # A badly rotten mango can lose almost all normal green/yellow/orange
    # chroma on one side. In that case the colour-built fruit mask may stop
    # at the healthy/rotten transition and incorrectly treat the dark rotten
    # side as background. Recover ONLY large, low-saturation gray/black
    # regions that are attached to the already-detected mango and penetrate
    # well inside a mango-shaped ROI envelope.
    #
    # This is intentionally stricter than the normal defect threshold so
    # dark leaves, hands and crop-corner shadows do not come back as fruit.

    provisional_fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    severe_edge_recovery_mask = np.zeros_like(
        fruit_mask
    )

    if provisional_fruit_pixels > 0:
        provisional_values = value[
            fruit_mask > 0
        ]

        provisional_median_v = float(
            np.median(provisional_values)
        )

        recovery_v_limit = int(
            max(
                92,
                min(155, provisional_median_v * 0.72)
            )
        )

        gray_black_candidate = np.zeros_like(
            value,
            dtype=np.uint8
        )

        gray_black_candidate[
            (value <= recovery_v_limit)
            & (saturation <= 118)
        ] = 255

        gray_black_candidate = cv.morphologyEx(
            gray_black_candidate,
            cv.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=2
        )

        gray_black_candidate = cv.morphologyEx(
            gray_black_candidate,
            cv.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        # Approximate a tight YOLO mango crop with a centred ellipse. This
        # excludes the rectangular corners where background leakage is most
        # common, while still allowing a genuine rotten side to reach the
        # left/right crop boundary.
        roi_mango_envelope = np.zeros_like(
            fruit_mask
        )

        cv.ellipse(
            roi_mango_envelope,
            (w // 2, h // 2),
            (
                max(1, int(w * 0.52)),
                max(1, int(h * 0.50))
            ),
            0,
            0,
            360,
            255,
            cv.FILLED
        )

        gray_black_candidate = cv.bitwise_and(
            gray_black_candidate,
            roi_mango_envelope
        )

        # Candidate must physically attach to the existing mango body.
        attach_px = max(
            5,
            int(min(h, w) * 0.035)
        )

        attach_kernel_size = attach_px * 2 + 1

        fruit_attachment_zone = cv.dilate(
            fruit_mask,
            np.ones(
                (attach_kernel_size, attach_kernel_size),
                np.uint8
            ),
            iterations=1
        )

        # Deep penetration inside the ROI ellipse distinguishes a real rotten
        # side from a shallow black crescent sitting only on the crop edge.
        padded_envelope = cv.copyMakeBorder(
            roi_mango_envelope,
            1, 1, 1, 1,
            cv.BORDER_CONSTANT,
            value=0
        )

        envelope_distance = cv.distanceTransform(
            padded_envelope,
            cv.DIST_L2,
            5
        )[1:h + 1, 1:w + 1]

        envelope_deep_margin = max(
            8,
            int(min(h, w) * 0.08)
        )

        envelope_deep_zone = np.zeros_like(
            fruit_mask
        )

        envelope_deep_zone[
            envelope_distance >= envelope_deep_margin
        ] = 255

        recovery_contours, _ = cv.findContours(
            gray_black_candidate,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        recovery_min_area = max(
            100,
            int(provisional_fruit_pixels * 0.008)
        )

        for recovery_contour in recovery_contours:
            recovery_area = cv.contourArea(
                recovery_contour
            )

            if recovery_area < recovery_min_area:
                continue

            recovery_component = np.zeros_like(
                fruit_mask
            )

            cv.drawContours(
                recovery_component,
                [recovery_contour],
                -1,
                255,
                cv.FILLED
            )

            recovery_pixels = max(
                1,
                cv.countNonZero(recovery_component)
            )

            attached_pixels = cv.countNonZero(
                cv.bitwise_and(
                    recovery_component,
                    fruit_attachment_zone
                )
            )

            deep_pixels = cv.countNonZero(
                cv.bitwise_and(
                    recovery_component,
                    envelope_deep_zone
                )
            )

            attached_ratio = (
                attached_pixels / recovery_pixels
            )

            deep_ratio = (
                deep_pixels / recovery_pixels
            )

            recovery_values = value[
                recovery_component > 0
            ]

            recovery_saturation = saturation[
                recovery_component > 0
            ]

            recovery_median_v = (
                float(np.median(recovery_values))
                if recovery_values.size
                else 255.0
            )

            recovery_median_s = (
                float(np.median(recovery_saturation))
                if recovery_saturation.size
                else 255.0
            )

            # Require a meaningful attachment, true inward penetration, and
            # genuinely gray/black appearance.
            if (
                attached_ratio >= 0.025
                and deep_ratio >= 0.12
                and recovery_median_v <= 138
                and recovery_median_s <= 105
            ):
                cv.drawContours(
                    severe_edge_recovery_mask,
                    [recovery_contour],
                    -1,
                    255,
                    cv.FILLED
                )

        # A recovered severe-rotten side is part of the mango, so include it
        # in the denominator as well as the defect mask later.
        fruit_mask = cv.bitwise_or(
            fruit_mask,
            severe_edge_recovery_mask
        )

    fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    if fruit_pixels == 0:
        return (
            roi.copy(),
            np.zeros(roi.shape[:2], dtype=np.uint8),
            0.0
        )

    # ======================================================
    # 3. PADDED DISTANCE TO FRUIT BOUNDARY
    # ======================================================
    # Padding is important when the mango touches a tight YOLO crop edge.
    # Without it, distanceTransform may treat that edge incorrectly.

    padded_fruit = cv.copyMakeBorder(
        fruit_mask,
        1, 1, 1, 1,
        cv.BORDER_CONSTANT,
        value=0
    )

    padded_distance = cv.distanceTransform(
        padded_fruit,
        cv.DIST_L2,
        5
    )

    distance = padded_distance[
        1:h + 1,
        1:w + 1
    ]

    analysis_margin = max(
        3,
        int(min(h, w) * 0.018)
    )

    analysis_mask = np.zeros_like(
        fruit_mask
    )

    analysis_mask[
        distance >= analysis_margin
    ] = 255

    # If the ROI is tiny, avoid eroding away nearly all the mango.
    if cv.countNonZero(analysis_mask) < fruit_pixels * 0.45:
        analysis_mask = fruit_mask.copy()

    # ======================================================
    # 4. ADAPTIVE BRIGHTNESS STATISTICS
    # ======================================================

    fruit_values = value[
        fruit_mask > 0
    ]

    median_v = float(np.median(fruit_values))
    q15_v = float(np.percentile(fruit_values, 15))

    # Local brightness reference. Kernel scales with ROI size and stays odd.
    local_k = max(
        15,
        int(min(h, w) * 0.18)
    )

    if local_k % 2 == 0:
        local_k += 1

    local_k = min(
        local_k,
        51
    )

    local_brightness = cv.GaussianBlur(
        value,
        (local_k, local_k),
        0
    )

    dark_difference = cv.subtract(
        local_brightness,
        value
    )

    # ======================================================
    # 5. STRONG BLACK / VERY DARK DAMAGE
    # ======================================================

    black_limit = int(
        max(
            42,
            min(75, q15_v - 18)
        )
    )

    severe_dark_mask = np.zeros_like(
        value,
        dtype=np.uint8
    )

    severe_dark_mask[
        (value <= black_limit)
        & (dark_difference >= 8)
    ] = 255

    # Large rotten mango patches can be uniformly dark, so their centre may
    # have little local contrast and the dark_difference test above only finds
    # the rim. Recover globally-dark rotten tissue conservatively.
    #
    # Low-saturation gray/black rot is the safest cue. For colourful dark
    # brown/black rot, require a brown/red hue as well. On generally dark
    # green/unripe mangoes the thresholds tighten automatically.
    global_dark_limit = int(
        max(
            72,
            min(145, median_v * (0.68 if median_v >= 155 else 0.58))
        )
    )

    desaturated_dark_rot = np.zeros_like(
        value,
        dtype=np.uint8
    )

    desaturated_dark_rot[
        (value <= global_dark_limit)
        & (saturation <= 115)
    ] = 255

    coloured_dark_limit = int(
        max(
            68,
            min(135, median_v * (0.64 if median_v >= 155 else 0.54))
        )
    )

    coloured_dark_rot = np.zeros_like(
        value,
        dtype=np.uint8
    )

    coloured_dark_rot[
        (
            (hue <= 24)
            | (hue >= 170)
        )
        & (value <= coloured_dark_limit)
        & (saturation >= 35)
    ] = 255

    severe_dark_mask = cv.bitwise_or(
        severe_dark_mask,
        desaturated_dark_rot
    )

    severe_dark_mask = cv.bitwise_or(
        severe_dark_mask,
        coloured_dark_rot
    )

    severe_dark_mask = cv.bitwise_and(
        severe_dark_mask,
        fruit_mask
    )

    # Any region that passed the strict gray/black edge-rot recovery above is
    # explicit severe-dark defect evidence even when its centre is uniformly
    # dark and therefore has little local contrast.
    severe_dark_mask = cv.bitwise_or(
        severe_dark_mask,
        severe_edge_recovery_mask
    )

    # ======================================================
    # 6. BROWN / DARK BRUISE
    # ======================================================
    # Red/orange blush is healthy on many mango cultivars. Brown candidates
    # therefore need either clearly low brightness or strong LOCAL darkness.

    brown_candidate = cv.inRange(
        hsv,
        np.array([0, 45, 18]),
        np.array([30, 255, 170])
    )

    brown_evidence = np.zeros_like(
        value,
        dtype=np.uint8
    )

    brown_evidence[
        (
            (dark_difference >= 24)
            & (value <= 165)
        )
        | (
            (dark_difference >= 14)
            & (value <= 105)
        )
    ] = 255

    brown_mask = cv.bitwise_and(
        brown_candidate,
        brown_evidence
    )

    # Relative dark bruise regardless of exact hue, but require a strong
    # local contrast so normal red-to-yellow gradients are not counted.
    bruise_mask = np.zeros_like(
        value,
        dtype=np.uint8
    )

    bruise_v_limit = int(
        min(
            175,
            max(110, median_v * 0.82)
        )
    )

    bruise_mask[
        (dark_difference >= 30)
        & (value <= bruise_v_limit)
    ] = 255

    # ======================================================
    # 7. OLIVE / DARK-GREEN DAMAGE
    # ======================================================
    # Healthy unripe mango can be green, so green colour by itself is never
    # treated as a defect. It must also be unusually dark locally.

    olive_candidate = cv.inRange(
        hsv,
        np.array([25, 35, 20]),
        np.array([95, 255, 125])
    )

    olive_evidence = np.zeros_like(
        value,
        dtype=np.uint8
    )

    olive_evidence[
        (dark_difference >= 22)
        & (value <= 115)
    ] = 255

    olive_mask = cv.bitwise_and(
        olive_candidate,
        olive_evidence
    )

    # ======================================================
    # 8. CLUSTERED BLACK SPOTS / ANTHRACNOSE-LIKE DAMAGE
    # ======================================================
    # Isolated mango lenticels are common and should be ignored. Dense dark
    # spot groups are more meaningful, so evaluate local density.

    spot_seed = np.zeros_like(
        value,
        dtype=np.uint8
    )

    spot_v_limit = int(
        min(
            115,
            max(72, median_v * 0.62)
        )
    )

    spot_seed[
        (value <= spot_v_limit)
        & (dark_difference >= 14)
    ] = 255

    spot_seed = cv.bitwise_and(
        spot_seed,
        analysis_mask
    )

    density_k = max(
        15,
        int(min(h, w) * 0.12)
    )

    if density_k % 2 == 0:
        density_k += 1

    density_k = min(
        density_k,
        35
    )

    spot_density = cv.blur(
        spot_seed,
        (density_k, density_k)
    )

    dense_spot_region = np.zeros_like(
        spot_seed
    )

    dense_spot_region[
        spot_density >= 28
    ] = 255

    dense_spot_region = cv.morphologyEx(
        dense_spot_region,
        cv.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    dense_spot_region = cv.bitwise_and(
        dense_spot_region,
        analysis_mask
    )

    # ======================================================
    # 9. PALE / GRAY / WHITE MOULD + DRY ROUGH PEEL
    # ======================================================

    laplacian = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    texture_strength = np.abs(
        laplacian
    )

    texture_strength = np.clip(
        texture_strength,
        0,
        255
    ).astype(np.uint8)

    _, strong_texture = cv.threshold(
        texture_strength,
        18,
        255,
        cv.THRESH_BINARY
    )

    # Use local texture density rather than a single edge pixel.
    texture_density = cv.blur(
        strong_texture,
        (11, 11)
    )

    rough_region = np.zeros_like(
        value,
        dtype=np.uint8
    )

    rough_region[
        texture_density >= 24
    ] = 255

    # Mould should usually be desaturated RELATIVE to nearby peel, not
    # merely a naturally pale green/yellow patch. Use local saturation
    # loss + rough texture to distinguish the two.
    local_saturation = cv.GaussianBlur(
        saturation,
        (local_k, local_k),
        0
    )

    saturation_drop = cv.subtract(
        local_saturation,
        saturation
    )

    pale_candidate = np.zeros_like(
        value,
        dtype=np.uint8
    )

    pale_candidate[
        (
            (saturation <= 70)
            & (value >= 55)
            & (value <= 235)
        )
        | (
            (saturation <= 105)
            & (saturation_drop >= 20)
            & (value >= 55)
            & (value <= 225)
        )
    ] = 255

    mould_texture = np.zeros_like(
        value,
        dtype=np.uint8
    )

    mould_texture[
        texture_density >= 32
    ] = 255

    mold_mask = cv.bitwise_and(
        pale_candidate,
        mould_texture
    )

    # Fruit/background contrast creates very strong texture right on the
    # mango outline. Do not let that outer edge masquerade as mould.
    # Genuine mould that reaches the boundary is still retained when the
    # patch also extends into this deeper interior region.
    mould_margin = max(
        7,
        int(min(h, w) * 0.09)
    )

    mould_inner_mask = np.zeros_like(
        fruit_mask
    )

    mould_inner_mask[
        distance >= mould_margin
    ] = 255

    mold_mask = cv.bitwise_and(
        mold_mask,
        mould_inner_mask
    )

    # Dry/shrivelled peel tends to become both rough and less saturated.
    # Requiring desaturation avoids flagging normal colourful red/yellow
    # mango gradients as dry defects.
    dry_mask = np.zeros_like(
        value,
        dtype=np.uint8
    )

    dry_mask[
        (texture_density >= 50)
        & (dark_difference >= 18)
        & (value <= 170)
        & (saturation <= 115)
    ] = 255

    dry_mask = cv.bitwise_and(
        dry_mask,
        mould_inner_mask
    )

    # ======================================================
    # 10. COMBINE DEFECT EVIDENCE
    # ======================================================

    # Normal colour/texture candidates stay away from the fruit boundary,
    # because shadows and crop edges are common false positives there.
    interior_defect_mask = cv.bitwise_or(
        brown_mask,
        bruise_mask
    )

    interior_defect_mask = cv.bitwise_or(
        interior_defect_mask,
        olive_mask
    )

    interior_defect_mask = cv.bitwise_or(
        interior_defect_mask,
        dense_spot_region
    )

    interior_defect_mask = cv.bitwise_or(
        interior_defect_mask,
        mold_mask
    )

    interior_defect_mask = cv.bitwise_or(
        interior_defect_mask,
        dry_mask
    )

    interior_defect_mask = cv.bitwise_and(
        interior_defect_mask,
        analysis_mask
    )

    # Very dark evidence is allowed to reach the fruit boundary. This is
    # important for real rot that starts at one side of a mango. The
    # component-level filter below still rejects shallow edge-only shadows.
    defect_mask = cv.bitwise_or(
        interior_defect_mask,
        severe_dark_mask
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        fruit_mask
    )

    # Small morphology only. Aggressive closing can merge normal mango
    # colour transitions into one huge false defect region.
    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Normal candidates stay inside analysis_mask, but the explicitly
    # recovered severe gray/black side is allowed to reach the true mango
    # boundary. Without this OR-back step a valid edge rot gets clipped away
    # immediately after it was recovered.
    defect_mask = cv.bitwise_and(
        defect_mask,
        analysis_mask
    )

    defect_mask = cv.bitwise_or(
        defect_mask,
        severe_edge_recovery_mask
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        fruit_mask
    )

    # ======================================================
    # 11. COMPONENT-LEVEL FALSE-POSITIVE FILTERING
    # ======================================================
    # Reject:
    # - tiny isolated lenticels
    # - long thin components hugging only the mango edge
    # - stem/contact-shadow style slivers
    # Keep:
    # - compact internal rot
    # - dense black-spot groups
    # - components that extend meaningfully into the fruit interior

    cleaned_mask = np.zeros_like(
        defect_mask
    )

    # Only clustered black spots / mould get the very small-component
    # allowance. A single tiny very-dark point is usually a normal mango
    # lenticel or image noise, so severe darkness alone does not lower the
    # minimum component area.
    # Tiny standalone mould-like patches are too easy to confuse with normal
    # mango lenticels/highlights. Only dense dark-spot evidence gets the
    # reduced minimum-area allowance; mould still has to clear the normal
    # component-size threshold.
    small_component_evidence = dense_spot_region.copy()

    edge_band = max(
        4,
        int(min(h, w) * 0.035)
    )

    deep_margin = max(
        edge_band + 2,
        int(min(h, w) * 0.07)
    )

    edge_zone = np.zeros_like(
        fruit_mask
    )

    edge_zone[
        (distance > 0)
        & (distance < edge_band)
    ] = 255

    deep_zone = np.zeros_like(
        fruit_mask
    )

    deep_zone[
        distance >= deep_margin
    ] = 255

    normal_min_area = max(
        18,
        int(fruit_pixels * 0.0012)
    )

    strong_min_area = max(
        8,
        int(fruit_pixels * 0.00035)
    )

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv.contourArea(contour)

        if area <= 0:
            continue

        component = np.zeros_like(
            defect_mask
        )

        cv.drawContours(
            component,
            [contour],
            -1,
            255,
            cv.FILLED
        )

        component_pixels = max(
            1,
            cv.countNonZero(component)
        )

        small_evidence_pixels = cv.countNonZero(
            cv.bitwise_and(
                component,
                small_component_evidence
            )
        )

        small_evidence_ratio = (
            small_evidence_pixels
            / component_pixels
        )

        min_area = (
            strong_min_area
            if small_evidence_ratio >= 0.18
            else normal_min_area
        )

        if area < min_area:
            continue

        edge_pixels = cv.countNonZero(
            cv.bitwise_and(
                component,
                edge_zone
            )
        )

        deep_pixels = cv.countNonZero(
            cv.bitwise_and(
                component,
                deep_zone
            )
        )

        edge_ratio = (
            edge_pixels
            / component_pixels
        )

        deep_ratio = (
            deep_pixels
            / component_pixels
        )

        recovery_overlap_pixels = cv.countNonZero(
            cv.bitwise_and(
                component,
                severe_edge_recovery_mask
            )
        )

        recovery_overlap_ratio = (
            recovery_overlap_pixels / component_pixels
        )

        confirmed_severe_edge_rot = (
            recovery_overlap_ratio >= 0.20
        )

        x, y, cw, ch = cv.boundingRect(
            contour
        )

        short_side = max(
            1,
            min(cw, ch)
        )

        long_side = max(
            cw,
            ch
        )

        elongation = (
            long_side
            / short_side
        )

        bbox_area = max(
            1,
            cw * ch
        )

        extent = (
            area
            / bbox_area
        )

        # Boundary-only regions are unreliable because normal mango
        # shading, stem contact and the YOLO crop edge can all become dark.
        # A genuine edge defect is kept when it extends meaningfully into
        # the fruit interior; a shallow edge-only component is rejected.
        if (
            edge_ratio >= 0.55
            and deep_ratio < 0.18
            and not confirmed_severe_edge_rot
        ):
            continue

        # Additional protection for compact edge blobs that are not long
        # enough to trigger the rule above. This mainly removes small dark
        # corner/shadow patches while preserving larger defects that reach
        # deeper into the mango.
        boundary_blob_limit = max(
            60,
            int(fruit_pixels * 0.012)
        )

        if (
            edge_ratio >= 0.35
            and deep_ratio < 0.10
            and area < boundary_blob_limit
            and not confirmed_severe_edge_rot
        ):
            continue

        # --------------------------------------------------
        # ROI-corner / crop-edge artifact suppression
        # --------------------------------------------------
        # The final defect mask has already been clipped a few pixels inside
        # analysis_mask, so a crop-edge artifact often starts at x/y~=2-4
        # instead of literal 0. Detect that geometry using a tolerant margin.
        crop_edge_margin = max(
            analysis_margin + 2,
            int(min(h, w) * 0.025)
        )

        touches_left = x <= crop_edge_margin
        touches_right = (x + cw) >= (w - crop_edge_margin)
        touches_top = y <= crop_edge_margin
        touches_bottom = (y + ch) >= (h - crop_edge_margin)

        touches_corner = (
            (touches_left or touches_right)
            and (touches_top or touches_bottom)
        )

        touches_any_crop_edge = (
            touches_left
            or touches_right
            or touches_top
            or touches_bottom
        )

        component_area_ratio = (
            component_pixels / max(1, fruit_pixels)
        )

        component_values = value[
            component > 0
        ]

        component_median_v = (
            float(np.median(component_values))
            if component_values.size
            else 255.0
        )

        # Small/medium components occupying a rectangular ROI corner are
        # overwhelmingly likely to be background, leaf or contact-shadow
        # leakage. Preserve only a genuinely very-dark patch that also reaches
        # well into the mango body (real black rot can start at the edge).
        if touches_corner and component_area_ratio < 0.12:
            very_dark_deep_rot = (
                component_median_v <= 62
                and deep_ratio >= 0.25
            )

            if not (
                very_dark_deep_rot
                or confirmed_severe_edge_rot
            ):
                continue

        # Reject small single-edge slivers. Keep a genuinely black, inward-
        # extending lesion so real edge rot is not thrown away.
        if (
            touches_any_crop_edge
            and not touches_corner
            and component_area_ratio < 0.035
        ):
            strong_edge_rot = (
                component_median_v <= 58
                and deep_ratio >= 0.14
            )

            if not (
                strong_edge_rot
                or confirmed_severe_edge_rot
            ):
                continue

        # Long thin crop-edge bands are typical of leaf/background/shadow
        # leakage rather than organic mango lesions.
        if (
            touches_any_crop_edge
            and elongation >= 2.2
            and extent < 0.62
            and deep_ratio < 0.30
            and not confirmed_severe_edge_rot
        ):
            continue

        cv.drawContours(
            cleaned_mask,
            [contour],
            -1,
            255,
            cv.FILLED
        )

    defect_mask = cv.bitwise_and(
        cleaned_mask,
        fruit_mask
    )

    # ======================================================
    # 12. DEFECT PERCENTAGE
    # ======================================================

    defect_pixels = cv.countNonZero(
        defect_mask
    )

    defect_percentage = (
        (defect_pixels / fruit_pixels) * 100
        if fruit_pixels > 0
        else 0.0
    )

    defect_percentage = min(
        defect_percentage,
        100.0
    )

    # ======================================================
    # 13. DRAW DEFECT CONTOURS
    # ======================================================
    # Percentage uses the real cleaned mask.
    #
    # Normal mango defects still hide line pixels that sit directly on the
    # outer fruit boundary. This prevents a tight crop, leaf shadow or normal
    # peel edge from creating a false red outline around the whole mango.
    #
    # IMPORTANT exception: a component that overlaps the strict
    # severe_edge_recovery_mask is confirmed large gray/black edge rot.
    # For that case the OUTER boundary is part of the real defect and should
    # be visible too. Draw that contour separately without the display-edge
    # clipping. This changes VISUALISATION ONLY -- defect_mask and percentage
    # are unchanged.

    output = roi.copy()

    normal_line_mask = np.zeros_like(
        defect_mask
    )

    severe_edge_line_mask = np.zeros_like(
        defect_mask
    )

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv.contourArea(contour) < strong_min_area:
            continue

        component = np.zeros_like(
            defect_mask
        )

        cv.drawContours(
            component,
            [contour],
            -1,
            255,
            cv.FILLED
        )

        component_pixels = max(
            1,
            cv.countNonZero(component)
        )

        recovery_pixels = cv.countNonZero(
            cv.bitwise_and(
                component,
                severe_edge_recovery_mask
            )
        )

        recovery_ratio = (
            recovery_pixels / component_pixels
        )

        # Only confirmed severe edge rot is allowed to show its complete
        # outer contour. Ordinary defects keep the safer clipped display.
        if recovery_ratio >= 0.20:
            cv.drawContours(
                severe_edge_line_mask,
                [contour],
                -1,
                255,
                2
            )
        else:
            cv.drawContours(
                normal_line_mask,
                [contour],
                -1,
                255,
                2
            )

    display_edge_margin = max(
        2,
        int(min(h, w) * 0.012)
    )

    display_safe = np.zeros_like(
        fruit_mask
    )

    display_safe[
        distance >= display_edge_margin
    ] = 255

    # Hide only NORMAL contour-line pixels close to the mango boundary.
    normal_line_mask = cv.bitwise_and(
        normal_line_mask,
        display_safe
    )

    contour_line_mask = cv.bitwise_or(
        normal_line_mask,
        severe_edge_line_mask
    )

    output[
        contour_line_mask > 0
    ] = (0, 0, 255)

    return (
        output,
        defect_mask,
        defect_percentage
    )


# ==========================================================
# Strawberry Defect Detection
# ==========================================================

def detect_strawberry_defect(roi):
    """
    Detects visible strawberry defects:
    - dark brown / black rot
    - bruised dark regions
    - pale / gray mould

    Natural strawberry seeds are reduced by removing
    small isolated dark components.
    """

    blurred = cv.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    fruit_mask = get_fruit_mask(
        roi
    )

    if cv.countNonZero(fruit_mask) == 0:
        return (
            roi.copy(),
            np.zeros(roi.shape[:2], dtype=np.uint8),
            0.0
        )

    # ------------------------------------------------------
    # Safe inner fruit region
    # ------------------------------------------------------

    distance = cv.distanceTransform(
        fruit_mask,
        cv.DIST_L2,
        5
    )

    h, w = fruit_mask.shape

    margin = max(
        6,
        int(min(h, w) * 0.025)
    )

    inner_mask = np.zeros_like(
        fruit_mask
    )

    inner_mask[
        distance >= margin
    ] = 255

    # ------------------------------------------------------
    # Local dark / bruised regions
    # ------------------------------------------------------

    value = hsv[:, :, 2]

    local_brightness = cv.GaussianBlur(
        value,
        (25, 25),
        0
    )

    dark_difference = cv.subtract(
        local_brightness,
        value
    )

    relative_dark = np.zeros_like(
        value,
        dtype=np.uint8
    )

    relative_dark[
        (dark_difference > 28)
        & (value < 170)
    ] = 255

    # ------------------------------------------------------
    # Brown / rotten surface
    # Red healthy skin is common, therefore brown pixels
    # must also be locally darker than their surroundings.
    # ------------------------------------------------------

    brown_candidate = cv.inRange(
        hsv,
        np.array([0, 45, 15]),
        np.array([25, 255, 145])
    )

    brown_confirm = np.zeros_like(
        value,
        dtype=np.uint8
    )

    brown_confirm[
        dark_difference > 14
    ] = 255

    brown_mask = cv.bitwise_and(
        brown_candidate,
        brown_confirm
    )

    # ------------------------------------------------------
    # Large / severe brown rot
    #
    # A large rotten patch can be uniformly brown, so it may
    # NOT be much darker than its own local neighbourhood.
    # The previous local-darkness rule can therefore miss a
    # big rotten area even when it is visually obvious.
    #
    # This Strawberry-only mask detects genuinely dark/dull
    # brown tissue directly by HSV colour.
    # ------------------------------------------------------

    severe_brown_1 = cv.inRange(
        hsv,
        np.array([4, 45, 20]),
        np.array([28, 255, 150])
    )

    severe_brown_2 = cv.inRange(
        hsv,
        np.array([0, 20, 20]),
        np.array([180, 170, 135])
    )

    severe_brown_mask = cv.bitwise_or(
        severe_brown_1,
        severe_brown_2
    )

    # Keep it on the actual strawberry surface only.
    severe_brown_mask = cv.bitwise_and(
        severe_brown_mask,
        inner_mask
    )

    # Join nearby rotten pixels into one meaningful patch.
    severe_brown_mask = cv.morphologyEx(
        severe_brown_mask,
        cv.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    # Remove tiny seed-like dark dots. Only keep meaningful
    # severe-brown components.
    strawberry_pixels_for_rot = cv.countNonZero(
        fruit_mask
    )

    severe_min_area = max(
        90,
        int(strawberry_pixels_for_rot * 0.004)
    )

    severe_clean = np.zeros_like(
        severe_brown_mask
    )

    severe_contours, _ = cv.findContours(
        severe_brown_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for severe_contour in severe_contours:

        if cv.contourArea(severe_contour) >= severe_min_area:

            cv.drawContours(
                severe_clean,
                [severe_contour],
                -1,
                255,
                cv.FILLED
            )

    severe_brown_mask = severe_clean

    # ------------------------------------------------------
    # Black rot
    # ------------------------------------------------------

    black_mask = cv.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 60])
    )

    # ------------------------------------------------------
    # Pale / white / gray mould
    # Require rough texture to reduce normal glare.
    # ------------------------------------------------------

    pale_candidate = cv.inRange(
        hsv,
        np.array([0, 0, 75]),
        np.array([180, 95, 240])
    )

    gray = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2GRAY
    )

    laplacian = cv.Laplacian(
        gray,
        cv.CV_32F,
        ksize=3
    )

    texture = np.abs(
        laplacian
    )

    texture = np.clip(
        texture,
        0,
        255
    ).astype(np.uint8)

    _, texture_mask = cv.threshold(
        texture,
        16,
        255,
        cv.THRESH_BINARY
    )

    texture_mask = cv.dilate(
        texture_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    mold_mask = cv.bitwise_and(
        pale_candidate,
        texture_mask
    )

    # ------------------------------------------------------
    # Combine candidates
    # ------------------------------------------------------

    defect_mask = cv.bitwise_or(
        relative_dark,
        brown_mask
    )

    # IMPORTANT:
    # Include the large / severe brown rot mask.
    # It was created above but was not previously combined into
    # the final Strawberry defect mask, so large brown rotten
    # areas had no effect on the final percentage.
    defect_mask = cv.bitwise_or(
        defect_mask,
        severe_brown_mask
    )

    defect_mask = cv.bitwise_or(
        defect_mask,
        black_mask
    )

    defect_mask = cv.bitwise_or(
        defect_mask,
        mold_mask
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        inner_mask
    )

    # ------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    defect_mask = cv.bitwise_and(
        defect_mask,
        inner_mask
    )

    defect_mask = remove_border_components(
        defect_mask,
        margin=5,
        min_area=40,
        border_overlap_ratio=0.6
    )

    # ------------------------------------------------------
    # Remove natural seeds and tiny isolated dots
    # ------------------------------------------------------

    fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    min_area = max(
        70,
        int(fruit_pixels * 0.0025)
    )

    cleaned_mask = np.zeros_like(
        defect_mask
    )

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv.contourArea(contour) >= min_area:

            cv.drawContours(
                cleaned_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )

    defect_mask = cleaned_mask

    # ------------------------------------------------------
    # Calculate percentage
    # ------------------------------------------------------

    defect_pixels = cv.countNonZero(
        defect_mask
    )

    if fruit_pixels == 0:

        defect_percentage = 0.0

    else:

        defect_percentage = (
            defect_pixels
            / fruit_pixels
        ) * 100

        defect_percentage = min(
            defect_percentage,
            100.0
        )

    # ------------------------------------------------------
    # Draw defect contours
    # ------------------------------------------------------

    output = roi.copy()

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

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
        defect_mask,
        defect_percentage
    )


# ==========================================================
# Main Defect Detector
# ==========================================================

def detect_defect(
    roi,
    fruit_type,
    fruit_mask_hint=None
):

    fruit_type = fruit_type.lower()


    if fruit_type == "banana":

        return detect_banana_defect(
            roi
        )


    elif fruit_type == "apple":

        return detect_apple_defect(
            roi
        )


    elif fruit_type == "orange":

        return detect_orange_defect(
            roi
        )


    elif fruit_type == "mango":

        return detect_mango_defect(
            roi,
            fruit_mask_hint=fruit_mask_hint
        )


    elif fruit_type == "strawberry":

        return detect_strawberry_defect(
            roi
        )


    return (
        roi.copy(),

        np.zeros(
            roi.shape[:2],
            dtype=np.uint8
        ),

        0.0
    )