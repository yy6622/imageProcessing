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
    if defect_percentage >= 20:
        ripeness = "Overripe"

    elif green >= 25:
        ripeness = "Unripe"

    elif defect_percentage >= 8:
        ripeness = "Overripe"

    elif orange >= 50 and green < 20:
        ripeness = "Ripe"

    else:
        ripeness = "Unripe"


    return (
        ripeness,
        green,
        orange
    )


# ==========================================================
# Strawberry Ripeness
# ==========================================================

def classify_strawberry(
    roi,
    defect_percentage
):
    """
    Classifies Strawberry as:
    - Unripe   : green fruit surface is dominant
    - Ripe     : red fruit surface is dominant
    - Overripe : visible defect / rot / mould is substantial

    IMPORTANT:
    This function is Strawberry-only.
    Banana, Apple and Orange rules are not changed.
    """

    hsv = cv.cvtColor(
        roi,
        cv.COLOR_BGR2HSV
    )

    fruit_mask = get_fruit_mask(
        roi
    )

    if cv.countNonZero(fruit_mask) == 0:
        return (
            "Unknown",
            0.0,
            0.0
        )

    # ------------------------------------------------------
    # Red Strawberry Surface
    # ------------------------------------------------------

    red1 = colour_percentage(
        hsv,
        [0, 55, 35],
        [12, 255, 255],
        fruit_mask
    )

    red2 = colour_percentage(
        hsv,
        [168, 55, 35],
        [180, 255, 255],
        fruit_mask
    )

    red = (
        red1
        + red2
    )

    # ------------------------------------------------------
    # Green / Unripe Strawberry Surface
    # ------------------------------------------------------

    green = colour_percentage(
        hsv,
        [25, 35, 35],
        [95, 255, 255],
        fruit_mask
    )

    # ------------------------------------------------------
    # Pale / Whitish Unripe Surface
    #
    # A strawberry can be unripe even when it is not strongly
    # green yet. Pale/white low-saturation skin is therefore
    # used only as a secondary unripe cue.
    # ------------------------------------------------------

    pale = colour_percentage(
        hsv,
        [0, 0, 85],
        [180, 70, 255],
        fruit_mask
    )

    # ======================================================
    # Strawberry Ripeness Decision
    # ======================================================

    # Clear deterioration has priority.
    #
    # Current defect detector already removes most seeds and
    # tiny isolated dots, so >= 8% is treated as meaningful
    # visible deterioration.
    # Strawberry-specific overripe rule:
    #
    # A healthy ripe strawberry can still produce around 8% apparent
    # defect because of seeds, highlights and natural surface texture.
    # Therefore 8% alone is too sensitive.
    #
    # Strong damage:
    #   defect >= 15% -> Overripe
    #
    # Moderate damage:
    #   defect >= 8% is treated as Overripe only when red coverage
    #   has already dropped below 70%, which is more consistent with
    #   deterioration / browning.
    if (
        defect_percentage >= 15.0
        or (
            defect_percentage >= 8.0
            and red < 70.0
        )
    ):

        ripeness = "Overripe"

    # Clearly green-dominant fruit.
    elif (
        green >= 30.0
        and green > red
    ):

        ripeness = "Unripe"

    # Pale / white fruit with little red is also still unripe.
    elif (
        pale >= 30.0
        and red < 35.0
    ):

        ripeness = "Unripe"

    # Mature strawberry is predominantly red.
    elif red >= 35.0:

        ripeness = "Ripe"

    # Weak red + more green than red means not mature yet.
    elif (
        green >= 18.0
        and green > red
    ):

        ripeness = "Unripe"

    # Conservative fallback:
    # if the fruit is not clearly red enough, treat it as unripe
    # rather than incorrectly calling a pale/green fruit ripe.
    else:

        ripeness = "Unripe"

    return (
        ripeness,
        green,
        red
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
    # Strawberry
    # ------------------------------------------------------

    elif fruit_type == "strawberry":

        (
            ripeness,
            colour1,
            colour2
        ) = classify_strawberry(
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