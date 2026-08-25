"""Preprocessing: noise removal + image enhancement.

Core Functional Requirement: "Implement noise removal and image enhancement
techniques (e.g., Gaussian/Median filtering, contrast stretching)."

Each step is a separate, independently selectable technique rather than one
fixed pipeline, so they can be compared against each other (which is the
whole point of Mode A - Comparative & Enhancement Study) instead of just
being hard-coded into StemDetector.preprocess().

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

    - "median": cv2.medianBlur. Good at removing salt-and-pepper style
      speckle noise (common in cheap camera sensors / compression
      artifacts) while keeping edges reasonably sharp. This is what
      StemDetector already used internally.
    - "gaussian": cv2.GaussianBlur. Smoother, more even blur; better for
      general sensor noise but softens edges more than median.
    - "bilateral": cv2.bilateralFilter. Slower, but smooths flat regions
      while preserving edges much better than either of the above - the
      trade-off is it can leave more noise in busy/textured areas.
    - "none": passthrough, useful as a baseline for comparison.
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
    """Enhance contrast/detail in a BGR image.

    - "clahe": Contrast Limited Adaptive Histogram Equalization on the LAB
      lightness channel. Boosts local contrast without blowing out
      already-bright regions the way global equalization can. This is
      what StemDetector already used internally.
    - "histogram_equalize": global histogram equalization on the LAB
      lightness channel. Simpler and faster than CLAHE, but can over-brighten
      large uniform areas (e.g. a plain background) since it has no local
      windowing.
    - "contrast_stretch": linear min-max normalization of the lightness
      channel to the full 0-255 range. Cheapest option; does nothing if the
      image already uses the full range, but helps a washed-out/low-contrast
      photo.
    - "none": passthrough, useful as a baseline for comparison.
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
