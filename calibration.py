
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
    return CalibrationResult(cm_per_pixel=cm_per_pixel, method="manual", confidence="ok")


def manual_reference(reference_width_cm: float, reference_width_px: float) -> CalibrationResult:
    if reference_width_px <= 0:
        raise ValueError("reference_width_px must be > 0")
    return CalibrationResult(
        cm_per_pixel=reference_width_cm / reference_width_px,
        method="manual",
        confidence="ok",
    )


def uncalibrated() -> CalibrationResult:
    return CalibrationResult(cm_per_pixel=1.0, method="none", confidence="uncalibrated")


# ======================================================
# Perspective rectification
# ======================================================
def detect_reference_quad(image, min_area_frac: float = 0.05):
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
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return rect
