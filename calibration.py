

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

DEFAULT_MARKER_SIZE_CM = 5.0

DEFAULT_ARUCO_DICT = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None


@dataclass
class CalibrationResult:
    cm_per_pixel: float
    method: str                   
    marker_corners: Optional[np.ndarray] = None
    confidence: str = "ok"     

    def px_to_cm(self, pixels):
        return pixels * self.cm_per_pixel

    def px_area_to_cm2(self, pixels_area):
        return pixels_area * (self.cm_per_pixel ** 2)


def detect_aruco_scale(image, marker_size_cm=DEFAULT_MARKER_SIZE_CM, aruco_dict=DEFAULT_ARUCO_DICT):

    if not hasattr(cv2, "aruco"):
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(aruco_dict)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(corners) == 0:
        return None
    marker_corners = corners[0].reshape((4, 2))
    side_lengths_px = [
        np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
        for i in range(4)
    ]
    avg_side_px = float(np.mean(side_lengths_px))

    if avg_side_px < 1e-6:
        return None

    cm_per_pixel = marker_size_cm / avg_side_px

    return CalibrationResult(
        cm_per_pixel=cm_per_pixel,
        method="aruco",
        marker_corners=marker_corners,
        confidence="ok",
    )


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

    return CalibrationResult(cm_per_pixel=1.0, method="none", confidence="uncalibrated")


def calibrate(image, marker_size_cm=DEFAULT_MARKER_SIZE_CM, manual_cm_per_pixel: Optional[float] = None):

    result = detect_aruco_scale(image, marker_size_cm=marker_size_cm)
    if result is not None:
        return result

    if manual_cm_per_pixel is not None:
        r = manual_scale(manual_cm_per_pixel)
        r.confidence = "fallback"
        return r

    return uncalibrated()


# ======================================================
# Perspective rectification
# ======================================================
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