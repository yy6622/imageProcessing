"""Image calibration: pixel-to-physical-size scaling.

Detects a printed ArUco marker of known side length to derive a
cm-per-pixel ratio, converting bounding box/contour measurements from
pixels to real-world units. Falls back to a manually supplied ratio, or
stays uncalibrated (pixel-only) if neither is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Tried in order since the marker's dictionary type isn't known in advance.
_ARUCO_DICTIONARIES = (
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
)


@dataclass
class CalibrationResult:
    cm_per_pixel: Optional[float]  # None if uncalibrated
    method: str                    # "aruco", "manual", or "uncalibrated"
    confidence: str                # human-readable note, e.g. "marker detected" / "no marker found"
    marker_corners: Optional[np.ndarray] = None  # for optionally drawing the detected marker

    @property
    def is_calibrated(self) -> bool:
        return self.cm_per_pixel is not None

    def px_to_cm(self, pixels: float) -> Optional[float]:
        return None if self.cm_per_pixel is None else pixels * self.cm_per_pixel

    def area_px_to_cm2(self, area_px: float) -> Optional[float]:
        return None if self.cm_per_pixel is None else area_px * (self.cm_per_pixel ** 2)


def uncalibrated() -> CalibrationResult:
    return CalibrationResult(cm_per_pixel=None, method="uncalibrated", confidence="no calibration requested")


def manual_scale(cm_per_pixel: float) -> CalibrationResult:
    return CalibrationResult(cm_per_pixel=cm_per_pixel, method="manual", confidence="user-supplied ratio")


def calibrate(
    image: np.ndarray, marker_size_cm: float = 5.0, manual_cm_per_pixel: Optional[float] = None
) -> CalibrationResult:
    """Try to auto-detect an ArUco marker in the photo; fall back to a
    manual ratio if one is supplied, otherwise return uncalibrated."""
    detected = _detect_aruco_scale(image, marker_size_cm)
    if detected is not None:
        return detected
    if manual_cm_per_pixel is not None:
        return manual_scale(manual_cm_per_pixel)
    return CalibrationResult(
        cm_per_pixel=None, method="uncalibrated",
        confidence="no ArUco marker found in photo, and no manual ratio given",
    )


def _detect_aruco_scale(image: np.ndarray, marker_size_cm: float) -> Optional[CalibrationResult]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for dict_id in _ARUCO_DICTIONARIES:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            continue

        # Average all four sides across every marker found, more robust
        # than a single edge if the marker is viewed at an angle.
        side_lengths_px = []
        for marker_corners in corners:
            pts = marker_corners.reshape(4, 2)
            for i in range(4):
                side_lengths_px.append(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
        avg_side_px = float(np.mean(side_lengths_px))
        if avg_side_px <= 0:
            continue

        cm_per_pixel = marker_size_cm / avg_side_px
        return CalibrationResult(
            cm_per_pixel=cm_per_pixel,
            method="aruco",
            confidence=f"marker detected ({len(corners)} marker(s), dict={dict_id})",
            marker_corners=corners[0].reshape(4, 2),
        )
    return None


def draw_marker_overlay(image: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Draw the detected ArUco marker outline, for visual confirmation in the dashboard."""
    if calibration.marker_corners is None:
        return image
    output = image.copy()
    pts = calibration.marker_corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
    return output
