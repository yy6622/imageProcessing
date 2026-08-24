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
    Detects banana defects:
    - brown rot
    - dark brown rot
    - black damaged peel
    - gray / white mould

    Healthy yellow / green peel and common
    shadows are reduced.
    """

    # ------------------------------------------------------
    # Blur
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

    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    # ======================================================
    # 1. Banana Fruit Mask
    # ======================================================

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    rough_mask = np.zeros(
        saturation.shape,
        dtype=np.uint8
    )

    # Banana coloured / dark skin
    rough_mask[
        (saturation > 20)
        & (value > 10)
    ] = 255


    # ------------------------------------------------------
    # Remove blue / cyan background
    # ------------------------------------------------------

    blue_background = (
        (hue >= 80)
        & (hue <= 130)
        & (saturation > 40)
    )

    rough_mask[
        blue_background
    ] = 0


    # ------------------------------------------------------
    # Clean mask
    # ------------------------------------------------------

    rough_mask = cv.morphologyEx(
        rough_mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    rough_mask = cv.morphologyEx(
        rough_mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )


    contours, _ = cv.findContours(
        rough_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    fruit_mask = np.zeros(
        roi.shape[:2],
        dtype=np.uint8
    )

    if contours:

        largest = max(
            contours,
            key=cv.contourArea
        )

        cv.drawContours(
            fruit_mask,
            [largest],
            -1,
            255,
            cv.FILLED
        )


    # ------------------------------------------------------
    # Final blue background removal
    # ------------------------------------------------------

    blue_mask = cv.inRange(
        hsv,
        np.array([80, 40, 0]),
        np.array([130, 255, 255])
    )

    fruit_mask[
        blue_mask > 0
    ] = 0


    # Remove a small outer edge
    fruit_mask = cv.erode(
        fruit_mask,
        kernel,
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


    # ======================================================
    # 11. Morphological Processing
    # ======================================================

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
        inner_fruit_mask
    )


    # ======================================================
    # 12. Remove Tiny Noise
    # ======================================================

    cleaned_mask = np.zeros_like(
        defect_mask
    )

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area > 60:

            cv.drawContours(
                cleaned_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )


    defect_mask = cleaned_mask

    defect_mask = cv.bitwise_and(
        defect_mask,
        inner_fruit_mask
    )


    # ======================================================
    # 13. Calculate Defect Percentage
    # ======================================================

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


    # ======================================================
    # 14. Draw Defects
    # ======================================================

    output = roi.copy()

    contours, _ = cv.findContours(
        defect_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv.contourArea(contour) > 60:

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
    Detects wrinkle / shrivel texture on apple skin.

    Healthy smooth apple:
        low wrinkle percentage

    Old / shrivelled apple:
        higher wrinkle percentage
    """

    # ------------------------------------------------------
    # Convert to grayscale
    # ------------------------------------------------------

    gray = cv.cvtColor(
        roi,
        cv.COLOR_BGR2GRAY
    )


    # ------------------------------------------------------
    # Blur small noise
    # ------------------------------------------------------

    blurred = cv.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # ------------------------------------------------------
    # Canny edge detection
    # Wrinkles create many small edges
    # ------------------------------------------------------

    edges = cv.Canny(
        blurred,
        70,
        150
    )


    # ------------------------------------------------------
    # Remove outside edge of apple
    # ------------------------------------------------------

    edge_kernel = np.ones(
        (7, 7),
        np.uint8
    )

    inner_mask = cv.erode(
        fruit_mask,
        edge_kernel,
        iterations=2
    )

    edges = cv.bitwise_and(
        edges,
        inner_mask
    )


    # ------------------------------------------------------
    # Connect nearby wrinkle lines
    # ------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    wrinkle_mask = cv.morphologyEx(
        edges,
        cv.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    # ------------------------------------------------------
    # Slightly thicken wrinkle lines
    # ------------------------------------------------------

    wrinkle_mask = cv.dilate(
        wrinkle_mask,
        kernel,
        iterations=1
    )


    # Keep wrinkles inside apple
    wrinkle_mask = cv.bitwise_and(
        wrinkle_mask,
        inner_mask
    )


    # ------------------------------------------------------
    # Remove very tiny wrinkle/noise regions
    # ------------------------------------------------------

    cleaned_mask = np.zeros_like(
        wrinkle_mask
    )

    contours, _ = cv.findContours(
        wrinkle_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv.contourArea(
            contour
        )

        if area > 25:

            cv.drawContours(
                cleaned_mask,
                [contour],
                -1,
                255,
                cv.FILLED
            )


    wrinkle_mask = cleaned_mask


    # ------------------------------------------------------
    # Calculate wrinkle percentage
    # ------------------------------------------------------

    fruit_pixels = cv.countNonZero(
        inner_mask
    )

    wrinkle_pixels = cv.countNonZero(
        wrinkle_mask
    )

    if fruit_pixels == 0:

        wrinkle_percentage = 0.0

    else:

        wrinkle_percentage = (
            wrinkle_pixels
            / fruit_pixels
        ) * 100


    return (
        wrinkle_mask,
        wrinkle_percentage
    )


# ==========================================================
# Apple Defect Detection
# ==========================================================

def detect_apple_defect(roi):
    """
    Detects apple defects using:

    1. Brown / dark damaged areas
    2. Wrinkle / shrivel texture
    """

    # ======================================================
    # 1. Colour-Based Defects
    # ======================================================

    (
        colour_output,
        colour_mask,
        colour_percentage
    ) = calculate_defect(
        roi,

        lower_brown=np.array(
            [5, 50, 20]
        ),

        upper_brown=np.array(
            [25, 255, 170]
        ),

        dark_threshold=90
    )


    # ======================================================
    # 2. Fruit Mask
    # ======================================================

    fruit_mask = get_fruit_mask(
        roi
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    fruit_mask = cv.erode(
        fruit_mask,
        kernel,
        iterations=1
    )


    # ======================================================
    # 3. Wrinkle Detection
    # ======================================================

    (
        wrinkle_mask,
        wrinkle_percentage
    ) = detect_apple_wrinkles(
        roi,
        fruit_mask
    )


    # ======================================================
    # 4. Combine Colour + Wrinkle Defects
    # ======================================================

    defect_mask = cv.bitwise_or(
        colour_mask,
        wrinkle_mask
    )


    # Keep defects inside apple
    defect_mask = cv.bitwise_and(
        defect_mask,
        fruit_mask
    )


    # ======================================================
    # 5. Clean Combined Mask
    # ======================================================

    small_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    defect_mask = cv.morphologyEx(
        defect_mask,
        cv.MORPH_CLOSE,
        small_kernel,
        iterations=1
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
# Use ellipse only as a clipping constraint so that
# table shadows / drooping tails cannot become fruit.

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

    # A blemish should be locally darker,
    # not merely darker because it is near
    # the curved edge of the orange.
    relative_dark_mask[
        (inside_orange_mask > 0)
        & (dark_difference > 18)
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

    brown_local_dark[
        dark_difference > 10
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
    # EXPAND CONFIRMED MOULD INTO ADJACENT GREEN MOULD
    # ======================================================

    # Green / olive mould candidate
    green_mold_candidate = cv.inRange(
        hsv,
        np.array([25, 20, 25]),
        np.array([100, 200, 210])
    )

    # Existing confirmed mould acts as the seed
    mold_seed = cv.bitwise_or(
        white_gray_mold,
        pale_green_mold
    )

    mold_seed = cv.bitwise_or(
        mold_seed,
        dark_green_mold
    )

    # Expand seed slightly so neighbouring mould pixels
    # can be included
    grow_kernel = np.ones(
        (9, 9),
        np.uint8
    )

    expanded_seed = cv.dilate(
        mold_seed,
        grow_kernel,
        iterations=2
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

    mold_mask = cv.bitwise_or(
        white_gray_mold,
        pale_green_mold
    )

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
    # 13. NON-MOULD DEFECTS
    # ======================================================

    # confirmed_brown_mask was already created above
    # using brown colour + dark_difference > 10


    # ------------------------------------------------------
    # Expand confirmed brown rot into nearby brown areas
    # ------------------------------------------------------

    brown_grow_kernel = np.ones(
        (9, 9),
        np.uint8
    )

    expanded_brown_seed = cv.dilate(
        confirmed_brown_mask,
        brown_grow_kernel,
        iterations=2
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
# 16c. ORANGE WRINKLE / SHRIVEL / DRY PEEL DETECTION
# ======================================================

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
    # 17. CALCULATE DEFECT PERCENTAGE
    # ======================================================

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


    # ======================================================
    # 18. DRAW DEFECT CONTOURS
    # ======================================================

    output = roi.copy()

    contours, _ = cv.findContours(
        defect_mask,
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
        defect_mask,
        defect_percentage
    )


# ==========================================================
# Main Defect Detector
# ==========================================================

def detect_defect(
    roi,
    fruit_type
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


    return (
        roi.copy(),

        np.zeros(
            roi.shape[:2],
            dtype=np.uint8
        ),

        0.0
    )