import cv2 as cv
import numpy as np


# ==========================================================
# Colour Percentage
# ==========================================================

def colour_percentage(
    hsv,
    lower,
    upper,
    fruit_mask
):

    mask = cv.inRange(
        hsv,
        np.array(
            lower,
            dtype=np.uint8
        ),
        np.array(
            upper,
            dtype=np.uint8
        )
    )

    mask = cv.bitwise_and(
        mask,
        fruit_mask
    )

    fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    if fruit_pixels == 0:
        return 0.0

    colour_pixels = cv.countNonZero(
        mask
    )

    return (
        colour_pixels
        / fruit_pixels
    ) * 100


# ==========================================================
# Fruit Mask
# ==========================================================

def get_fruit_mask(roi):
    """
    Creates a fruit mask from the YOLO ROI.

    The mask includes:
    - coloured fruit skin
    - dark / black damaged fruit skin

    White = fruit
    Black = background
    """

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # ------------------------------------------------------
    # Normal coloured fruit
    # ------------------------------------------------------

    colour_mask = np.zeros(
        saturation.shape,
        dtype=np.uint8
    )

    colour_mask[
        (saturation > 25)
        & (value > 20)
    ] = 255


    # ------------------------------------------------------
    # Very dark fruit skin
    #
    # Important for heavily overripe banana,
    # because black / grey peel may have low saturation.
    # ------------------------------------------------------

    dark_mask = cv.inRange(
        hsv,
        np.array(
            [0, 0, 10]
        ),
        np.array(
            [180, 255, 110]
        )
    )


    # ------------------------------------------------------
    # Combine
    # ------------------------------------------------------

    mask = cv.bitwise_or(
        colour_mask,
        dark_mask
    )


    # ------------------------------------------------------
    # Morphology
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # Largest connected region assumed to be fruit
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # Slightly remove ROI edge
    # ------------------------------------------------------

    fruit_mask = cv.erode(
        fruit_mask,
        kernel,
        iterations=1
    )

    return fruit_mask


# ==========================================================
# Banana Ripeness
# ==========================================================

def classify_banana(
    roi,
    defect_percentage
):

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    fruit_mask = get_fruit_mask(
        roi
    )


    # ------------------------------------------------------
    # Green = Unripe
    # ------------------------------------------------------

    green = colour_percentage(
        hsv,
        [35, 40, 30],
        [90, 255, 255],
        fruit_mask
    )


    # ------------------------------------------------------
    # Yellow = Ripe
    # ------------------------------------------------------

    yellow = colour_percentage(
        hsv,
        [20, 60, 80],
        [35, 255, 255],
        fruit_mask
    )


    # ------------------------------------------------------
    # Dark / Black Peel
    #
    # Used to recognise severely overripe bananas
    # even when the defect detector is intentionally
    # strict to avoid shadows.
    # ------------------------------------------------------

    dark_mask = cv.inRange(
        hsv,
        np.array(
            [0, 0, 0]
        ),
        np.array(
            [180, 255, 90]
        )
    )

    dark_mask = cv.bitwise_and(
        dark_mask,
        fruit_mask
    )


    # ------------------------------------------------------
    # Calculate dark percentage
    # ------------------------------------------------------

    fruit_pixels = cv.countNonZero(
        fruit_mask
    )

    dark_pixels = cv.countNonZero(
        dark_mask
    )

    if fruit_pixels == 0:

        dark_percentage = 0.0

    else:

        dark_percentage = (
            dark_pixels
            / fruit_pixels
        ) * 100


    # ======================================================
    # Banana Ripeness Decision
    # ======================================================

    # Very damaged OR mostly dark/black
    if (
        defect_percentage >= 20
        or dark_percentage >= 30
    ):

        ripeness = "Overripe"


    # Mostly green
    elif green >= 35:

        ripeness = "Unripe"


    # Otherwise normal yellow banana
    else:

        ripeness = "Ripe"


    return (
        ripeness,
        green,
        yellow
    )


# ==========================================================
# Apple Ripeness
# ==========================================================

def classify_apple(
    roi,
    defect_percentage
):

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    fruit_mask = get_fruit_mask(
        roi
    )


    # ------------------------------------------------------
    # Green / yellow-green apple
    # ------------------------------------------------------

    green = colour_percentage(
        hsv,
        [25, 25, 30],
        [90, 255, 255],
        fruit_mask
    )


    # ------------------------------------------------------
    # Red Apple Range 1
    # ------------------------------------------------------

    red1 = colour_percentage(
        hsv,
        [0, 50, 40],
        [10, 255, 255],
        fruit_mask
    )


    # ------------------------------------------------------
    # Red Apple Range 2
    # ------------------------------------------------------

    red2 = colour_percentage(
        hsv,
        [170, 50, 40],
        [180, 255, 255],
        fruit_mask
    )

    red = (
        red1
        + red2
    )


    # ======================================================
    # Apple Ripeness Decision
    # ======================================================

    # Rotten / heavily damaged
    if defect_percentage >= 8:

        ripeness = "Overripe"


    # Green-dominant apple
    elif (
        green >= 30
        and green > red
    ):

        ripeness = "Unripe"


    # Normal mature apple
    else:

        ripeness = "Ripe"


    return (
        ripeness,
        green,
        red
    )


# ==========================================================
# Orange Ripeness
# ==========================================================

def classify_orange(
    roi,
    defect_percentage
):

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    fruit_mask = get_fruit_mask(
        roi
    )


    # ------------------------------------------------------
    # Green / yellow-green orange
    # ------------------------------------------------------

    green = colour_percentage(
        hsv,
        [25, 35, 30],
        [90, 255, 255],
        fruit_mask
    )


    # ------------------------------------------------------
    # Mature orange colour
    # ------------------------------------------------------

    orange = colour_percentage(
        hsv,
        [5, 70, 70],
        [24, 255, 255],
        fruit_mask
    )


    # ======================================================
    # Orange Ripeness Decision
    # ======================================================

    # Rotten / mouldy
    if defect_percentage >= 8.0:

        ripeness = "Overripe"


    # Significant green peel
    elif green >= 25:

        ripeness = "Unripe"


    # Clearly mature orange
    elif (
        orange >= 50
        and green < 20
    ):

        ripeness = "Ripe"


    # Still not sufficiently orange
    else:

        ripeness = "Unripe"


    return (
        ripeness,
        green,
        orange
    )


# ==========================================================
# Main Ripeness Classifier
# ==========================================================

def classify_ripeness(
    roi,
    fruit_type,
    defect_percentage
):

    fruit_type = fruit_type.lower()


    # ------------------------------------------------------
    # Banana
    # ------------------------------------------------------

    if fruit_type == "banana":

        (
            ripeness,
            colour1,
            colour2
        ) = classify_banana(
            roi,
            defect_percentage
        )


    # ------------------------------------------------------
    # Apple
    # ------------------------------------------------------

    elif fruit_type == "apple":

        (
            ripeness,
            colour1,
            colour2
        ) = classify_apple(
            roi,
            defect_percentage
        )


    # ------------------------------------------------------
    # Orange
    # ------------------------------------------------------

    elif fruit_type == "orange":

        (
            ripeness,
            colour1,
            colour2
        ) = classify_orange(
            roi,
            defect_percentage
        )


    # ------------------------------------------------------
    # Unknown Fruit
    # ------------------------------------------------------

    else:

        return {
            "ripeness": "Unknown",
            "colour1": 0.0,
            "colour2": 0.0
        }


    return {
        "ripeness": ripeness,
        "colour1": colour1,
        "colour2": colour2
    }