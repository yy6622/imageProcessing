"""
calibration.py
==============================================
Spatial calibration: converts pixel measurements into physical units
(cm / cm^2) so bounding boxes and areas reported by the dashboard mean
something in the real world, not just "pixels".

Satisfies the assignment requirement:
    "Image Calibration: Perform spatial scaling and rectification to
     maintain consistency between pixel dimensions and physical
     measurements."

Two ways to supply the pixel->cm scale (both MANUAL, no ArUco/marker
detection — that mode was removed since nothing used it):
    - manual_scale(cm_per_pixel): you already know the ratio.
    - manual_reference(reference_width_cm, reference_width_px): you
      know a reference object's real width and its width in the photo
      (in pixels) instead, and this derives the ratio from that.

Also provides `rectify_perspective` (+ its `order_points` helper): if
the camera isn't perfectly perpendicular to the inspection surface, a
4-point perspective warp straightens the image plane first so pixel
spacing is uniform across the frame before measurement. The 4 corners
can come from `detect_reference_quad()` (auto: largest 4-sided shape
found in the photo) or be entered manually as a fallback.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    cm_per_pixel: float
    method: str                       # "manual" | "none"
    marker_corners: Optional[np.ndarray] = None
    confidence: str = "ok"            # "ok" | "fallback" | "uncalibrated"

    def px_to_cm(self, pixels):
        return pixels * self.cm_per_pixel

    def px_area_to_cm2(self, pixels_area):
        return pixels_area * (self.cm_per_pixel ** 2)


def manual_scale(cm_per_pixel: float) -> CalibrationResult:
    """Directly supply a known cm-per-pixel ratio."""
    return CalibrationResult(cm_per_pixel=cm_per_pixel, method="manual", confidence="ok")


def manual_reference(reference_width_cm: float, reference_width_px: float) -> CalibrationResult:
    """
    Supply a known reference object width in both cm and pixels (e.g.
    you measured a card in the photo is 8.56cm wide and 240px wide).
    """
    if reference_width_px <= 0:
        raise ValueError("reference_width_px must be > 0")
    return CalibrationResult(
        cm_per_pixel=reference_width_cm / reference_width_px,
        method="manual",
        confidence="ok",
    )


def uncalibrated() -> CalibrationResult:
    """
    No calibration available — measurements will be reported in pixels
    only. cm_per_pixel=1.0 acts as an identity scale so downstream code
    doesn't need special-casing; `confidence` flags it as untrustworthy
    for physical units.
    """
    return CalibrationResult(cm_per_pixel=1.0, method="none", confidence="uncalibrated")


# ======================================================
# Perspective rectification
# ======================================================
def detect_reference_quad(image, min_area_frac: float = 0.05):
    """
    Attempts to automatically find the largest 4-sided (quadrilateral)
    contour in the image, to use as a rectify_perspective() reference
    (e.g. a card, tray, or table edge). Returns a (4, 2) float32 array
    of corner points, or None if nothing suitable was found.

    Less reliable than a coded marker (ArUco): there's no unique
    identity to lock onto, so this just picks the largest sufficiently
    rectangular shape it sees — it can pick the wrong object if the
    frame is cluttered, low-contrast, or the intended reference isn't
    the most prominent quadrilateral in view. Callers should treat a
    None return as "fall back to manual corner entry", not an error.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    min_area = min_area_frac * image_area

    best_quad = None
    best_area = 0.0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < min_area or area <= best_area:
            continue
        best_area = area
        best_quad = approx.reshape(4, 2).astype(np.float32)

    return best_quad


def rectify_perspective(image, src_points: np.ndarray, output_size: Tuple[int, int] = None):
    """
    Warps the quadrilateral defined by src_points (4 corner points, any
    order, in the *original* image) onto a straight-on rectangle, so
    that pixel spacing is uniform across the whole frame. Use this when
    the camera isn't mounted perfectly perpendicular to the inspection
    surface (a common source of measurement error: objects farther
    from the camera look smaller than they are).

    src_points: array of shape (4, 2) — e.g. the 4 corners of a
    reference card/marker, or a manually annotated region of interest.
    output_size: (width, height) of the rectified output. If None,
    it's estimated from the max side lengths of src_points.
    """
    pts = order_points(src_points)
    (tl, tr, br, bl) = pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    if output_size is not None:
        max_width, max_height = output_size

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return rect
