"""Stem/crown/calyx detection: Traditional (classical CV), YOLO, Hybrid, and
Automatic (YOLO with a traditional fallback).

Mode A requirement this satisfies: "Technical Requirement: Implement and
benchmark multiple techniques" — the three underlying techniques
(detect_traditional / detect_yolo / detect_hybrid) are kept as separate,
independently callable methods so metrics.run_benchmark() can compare them
directly, while detect() (Automatic) is the single method app.py's normal
user workflow relies on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import cv2
import numpy as np

from . import preprocessing as _preprocessing

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

# Prefer locally trained V3 weights; fall back to the generic production
# path used by train_stem_yolo.py's "copy to models/best.pt" instructions.
_DEFAULT_MODEL_CANDIDATES = (
    "runs/detect/fruit_stem_detector_v3/weights/best.pt",
    "models/best.pt",
)

# Stems/crowns read as either a desaturated green (fresh) or a dark brown
# (dried/senescent) blob in HSV — both are covered so the traditional path
# isn't tied to only-fresh produce.
_STEM_HSV_RANGES = (
    ((25, 40, 20), (95, 255, 200)),   # green
    ((5, 30, 20), (25, 200, 150)),    # brown
)


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    method: str = ""
    contour: Optional[np.ndarray] = field(default=None, repr=False)


class StemDetector:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = next(
                (p for p in _DEFAULT_MODEL_CANDIDATES if os.path.isfile(p)),
                _DEFAULT_MODEL_CANDIDATES[-1],
            )
        self.model_path = model_path
        self.yolo_ready = False
        self._model = None
        if _ULTRALYTICS_AVAILABLE and os.path.isfile(model_path):
            try:
                self._model = YOLO(model_path)
                self.yolo_ready = True
            except Exception:
                self._model = None
                self.yolo_ready = False

    # ------------------------------------------------------------------
    # Traditional: classical HSV segmentation + morphology, no model file
    # needed. Stems sit near the top of the fruit in typical product
    # photos, so the search is restricted to that band to cut down on
    # false positives from background clutter.
    # ------------------------------------------------------------------
    def detect_traditional(
        self, image: np.ndarray, fruit_type: str, top_fraction: float = 0.45,
    ) -> tuple[list[Detection], float, str]:
        start = perf_counter()
        detections = self._detect_traditional_impl(image, top_fraction)
        elapsed = perf_counter() - start
        return detections, elapsed, "traditional"

    def traditional_mask(self, image: np.ndarray, top_fraction: float = 0.45) -> np.ndarray:
        """The binary stem-colour candidate mask the Traditional method
        filters into contours — exposed on its own so the UI can show it as
        a "candidate regions" step, independent of which method actually
        ran for a given detection."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in _STEM_HSV_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask[int(h * top_fraction):, :] = 0
        return mask

    def _detect_traditional_impl(self, image: np.ndarray, top_fraction: float) -> list[Detection]:
        h, w = image.shape[:2]
        mask = self.traditional_mask(image, top_fraction)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(20.0, 0.0005 * h * w)
        max_area = 0.05 * h * w

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = max(cw, ch) / max(1, min(cw, ch))
            if aspect > 6:
                continue
            fill_ratio = area / (cw * ch)
            score = min(1.0, (area / min_area) * 0.15 + fill_ratio * 0.3 + min(aspect / 4, 1.0) * 0.3)
            detections.append(Detection(
                bbox=(float(x), float(y), float(x + cw), float(y + ch)),
                confidence=float(score), method="traditional", contour=c,
            ))

        detections = _non_max_suppression(detections, iou_threshold=0.3)
        return detections[:3]

    # ------------------------------------------------------------------
    # YOLO: single-class ("Fruit-Stem-Detection") model trained by
    # train_stem_yolo.py. Silently returns no detections if no model
    # file was found/loaded — callers check yolo_ready to distinguish
    # "not available" from "ran and found nothing".
    # ------------------------------------------------------------------
    def detect_yolo(
        self, image: np.ndarray, fruit_type: str, confidence: float = 0.25,
    ) -> tuple[list[Detection], float]:
        start = perf_counter()
        detections: list[Detection] = []
        if self.yolo_ready:
            results = self._model.predict(image, conf=confidence, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    conf = float(box.conf[0].cpu().item())
                    detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, method="yolo"))
        detections = _non_max_suppression(detections, iou_threshold=0.4)
        elapsed = perf_counter() - start
        return detections, elapsed

    # ------------------------------------------------------------------
    # Hybrid: union of YOLO + Traditional, de-duplicated by IoU (YOLO
    # takes precedence on overlapping boxes since it's the trained model —
    # it sorts first because detect_yolo's boxes are listed before
    # detect_traditional's, and NMS keeps the highest-confidence box in
    # each overlapping cluster).
    # ------------------------------------------------------------------
    def detect_hybrid(
        self, image: np.ndarray, fruit_type: str, yolo_confidence: float = 0.25,
        iou_merge_threshold: float = 0.3,
    ) -> tuple[list[Detection], float]:
        start = perf_counter()
        yolo_detections, _ = self.detect_yolo(image, fruit_type, yolo_confidence)
        traditional_detections, _, _ = self.detect_traditional(image, fruit_type)

        merged = _non_max_suppression(yolo_detections + traditional_detections, iou_threshold=iou_merge_threshold)
        elapsed = perf_counter() - start
        return merged, elapsed

    # ------------------------------------------------------------------
    # Automatic: the single method the normal (non-benchmark) app.py
    # workflow uses. Runs YOLO at the caller-supplied (already permissive)
    # threshold, sanity-checks the boxes, and falls back to Traditional
    # when YOLO is unavailable or returns nothing plausible.
    # ------------------------------------------------------------------
    def detect(
        self, image: np.ndarray, fruit_type: str, method: str = "Automatic",
        yolo_confidence: float = 0.25, skip_preprocess: bool = False,
    ) -> tuple[list[Detection], float, str]:
        if not skip_preprocess:
            image = _preprocessing.preprocess(image)

        if method == "Traditional":
            return self.detect_traditional(image, fruit_type)
        if method == "YOLO":
            detections, elapsed = self.detect_yolo(image, fruit_type, yolo_confidence)
            return detections, elapsed, "yolo"
        if method == "Hybrid":
            detections, elapsed = self.detect_hybrid(image, fruit_type, yolo_confidence)
            return detections, elapsed, "hybrid"

        start = perf_counter()
        used = "yolo"
        detections: list[Detection] = []
        if self.yolo_ready:
            raw, _ = self.detect_yolo(image, fruit_type, yolo_confidence)
            detections = self._validate_yolo_boxes(raw, image.shape)
        if not detections:
            used = "traditional"
            detections, _, _ = self.detect_traditional(image, fruit_type)
        elapsed = perf_counter() - start
        return detections, elapsed, used

    def _validate_yolo_boxes(
        self, detections: list[Detection], image_shape: tuple, max_area_fraction: float = 0.6,
    ) -> list[Detection]:
        h, w = image_shape[:2]
        image_area = h * w
        valid = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 0 or area / image_area > max_area_fraction:
                continue
            valid.append(d)
        return valid

    # ------------------------------------------------------------------
    def annotate(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        annotated = image.copy()
        for d in detections:
            color = (0, 255, 0) if d.confidence >= 0.5 else (0, 165, 255)
            if d.contour is not None:
                cv2.drawContours(annotated, [d.contour], -1, color, 2)
            x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"stem {d.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated


def _non_max_suppression(detections: list[Detection], iou_threshold: float = 0.4) -> list[Detection]:
    """Greedy NMS: repeatedly keep the highest-confidence remaining box and
    drop any other box that overlaps it above iou_threshold (a duplicate/
    split detection of the same stem), so one stem doesn't show up as
    several stacked boxes."""
    remaining = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if _iou(best.bbox, d.bbox) < iou_threshold]
    return kept


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0
