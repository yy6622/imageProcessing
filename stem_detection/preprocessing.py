"""Preprocessing: noise removal + image enhancement.

Usage:
    denoised = denoise(image, method="median")
    enhanced = enhance(denoised, method="clahe")
"""

from __future__ import annotations

import cv2
import numpy as np

DENOISE_METHODS = ("median", "gaussian", "bilateral", "none")
ENHANCE_METHODS = ("clahe", "histogram_equalize", "contrast_stretch", "none")


def denoise(image: np.ndarray, method: str = "median") -> np.ndarray:
    """Remove noise from a BGR image.

    - "median": good at speckle noise, keeps edges reasonably sharp.
    - "gaussian": smoother overall blur, softens edges more.
    - "bilateral": slower, preserves edges better, leaves more noise in
      textured areas.
    - "none": passthrough, for baseline comparison.
    """
    if method == "median":
        return cv2.medianBlur(image, 5)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    if method == "none":
        return image.copy()
    raise ValueError(f"Unknown denoise method: {method!r}. Choose from {DENOISE_METHODS}.")


def enhance(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Enhance contrast/detail in a BGR image (operates on the LAB lightness channel).

    - "clahe": local adaptive contrast, avoids blowing out bright regions.
    - "histogram_equalize": global equalization, simpler but can over-brighten.
    - "contrast_stretch": linear min-max stretch, cheapest option.
    - "none": passthrough, for baseline comparison.
    """
    if method == "none":
        return image.copy()
    if method not in ENHANCE_METHODS:
        raise ValueError(f"Unknown enhance method: {method!r}. Choose from {ENHANCE_METHODS}.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if method == "clahe":
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    elif method == "histogram_equalize":
        l = cv2.equalizeHist(l)
    elif method == "contrast_stretch":
        lo, hi = float(l.min()), float(l.max())
        if hi > lo:
            l = np.clip((l.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(image: np.ndarray, denoise_method: str = "median", enhance_method: str = "clahe") -> np.ndarray:
    """Convenience wrapper: denoise then enhance in one call."""
    return enhance(denoise(image, denoise_method), enhance_method)
