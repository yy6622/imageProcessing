"""
colorDetection.py
==============================================
Inference / inspection pipeline for the fruit quality inspection
system: detects + counts every fruit in a photo, identifies each
one's TYPE (Apple/Banana/Orange), and classifies its QUALITY
(Fresh/Unripe/Rotten).

    Detection + fruit TYPE  — YOLOv8, pretrained on COCO. Apple/
                              banana/orange are already 3 of its 80
                              classes, so this needs zero extra
                              training/labeling. Handles densely
                              packed/overlapping fruit far better than
                              a classical distance-transform/Hough/
                              watershed splitter would.
    Quality                 — a CNN (train_cnn_quality.py), one
                              MobileNetV2 transfer-learning model per
                              fruit type, looking at the WHOLE fruit
                              at once rather than tiling it into
                              patches and voting.

Neither has a fallback to an older/classical approach — both were
deliberate choices made after head-to-head testing (see README.md's
"CNN quality classification" section for the full story, including a
data-leakage bug found and fixed along the way). If ultralytics or a
trained CNN model isn't available, this module reports that plainly
(a load error, or a ClassificationResult.error message) rather than
silently degrading to a weaker classical/SVM guess.

This file previously also contained a patch+color-histogram SVM
two-stage classifier (fruit type, then per-species quality) plus a
classical distance-transform + Hough-assisted watershed splitter for
separating touching fruit without an object detector — that code has
been removed (not just disconnected) per an explicit decision to drop
SVM entirely from this project. It's still recoverable from earlier
version history if a report needs to reference/compare against it;
colorDetection_Train.py's training functions for it are unaffected by
this file's changes.

Typical use (single image, full pipeline):

    from colorDetection import inspect_image_yolo

    result = inspect_image_yolo("sample.jpg", calibration=None)
    for obj in result["objects"]:
        print(obj["fruit_type"], obj["label"], obj["confidence"], obj["width_cm"])
"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from colorDetection_Train import segmentation_mask_and_contour

import preprocessing as prep
from calibration import CalibrationResult, uncalibrated

DEFAULT_IMAGE_SIZE = (512, 512)


# ======================================================
# Detection result + display/crop helpers
# ======================================================
@dataclass
class DetectionResult:
    found: bool
    bbox: Optional[tuple] = None          # (x, y, w, h) in pixels
    contour: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    area_px: float = 0.0
    perimeter_px: float = 0.0


def draw_detections(image, objects, box_color=(0, 255, 0), contour_color=(0, 165, 255)):
    """
    Returns a copy of `image` with every detected object's bounding box
    + contour + a small "#N Type Quality" label drawn on it. `objects`
    is the list of per-object dicts inspect_image_yolo() builds (each
    with a "detection" DetectionResult, "index", "fruit_type", "label").
    """
    annotated = image.copy()
    for obj in objects:
        detection = obj["detection"]
        if not detection.found:
            continue
        x, y, w, h = detection.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.drawContours(annotated, [detection.contour], -1, contour_color, 2)

        fruit_type = obj.get("fruit_type") or "?"
        quality = obj.get("label") or "?"
        text = f"#{obj['index'] + 1} {fruit_type} {quality}"
        text_y = max(y - 8, 15)
        # Black outline + white fill so the label stays legible over any
        # background color.
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


CROP_EXTRA_ERODE_PIXELS = 6  # applied only to isolate=True crops, on top of
                              # whatever erosion detection.mask already has —
                              # a few extra pixels trims any faint blended
                              # boundary/anti-aliasing ring that a classification
                              # mask can tolerate but looks untidy in a crop
                              # meant to be looked at directly by a person.


def crop_object(image, detection: DetectionResult, pad_frac=0.06, isolate=False):
    """
    Returns a standalone crop of just this one detected fruit — "cut
    out" of the full photo so it can be looked at (or saved/downloaded)
    on its own, one fruit at a time, instead of only ever seeing it
    embedded in the group photo with a bounding box drawn around it.

    pad_frac: small margin added around the tight bounding box so the
    crop doesn't slice right along the fruit's edge.
    isolate: if True, also blacks out every pixel outside this fruit's
    own contour (using detection.mask, eroded a few pixels further —
    see CROP_EXTRA_ERODE_PIXELS) — useful when several fruits were
    touching, so the crop shows ONLY this fruit, not slivers of its
    neighbors that happened to fall inside the same bounding box, and
    not a faint fringe of blended boundary pixels either.
    """
    if not detection.found:
        return image.copy()

    x, y, w, h = detection.bbox
    pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)

    src = image
    if isolate and detection.mask is not None:
        mask = detection.mask
        if CROP_EXTRA_ERODE_PIXELS > 0:
            kernel = np.ones((CROP_EXTRA_ERODE_PIXELS, CROP_EXTRA_ERODE_PIXELS), np.uint8)
            mask = cv2.erode(mask, kernel)
        src = cv2.bitwise_and(image, image, mask=mask)

    return src[y0:y1, x0:x1].copy()


# ======================================================
# Classification result
# ======================================================
@dataclass
class ClassificationResult:
    fruit_type: Optional[str] = None        # e.g. "Apple" / "Banana" / "Orange" — from YOLO's own label
    fruit_type_confidence: float = 0.0      # YOLO's box confidence
    label: Optional[str] = None             # quality: Fresh / Unripe / Rotten — from the CNN
    confidence: float = 0.0                 # CNN softmax score for the predicted quality label
    quality_backend: str = "cnn"            # always "cnn" — kept as a field (rather than removed)
                                              # in case a second quality backend is ever added again.
    error: Optional[str] = None             # e.g. "No CNN quality model found for fruit type 'Apple'"


# ======================================================
# CNN quality classification (see train_cnn_quality.py) — the ONLY
# quality (Fresh/Unripe/Rotten) path this pipeline uses.
# ======================================================
# train_cnn_quality.py trains a CNN (MobileNetV2 transfer learning)
# that looks at the WHOLE fruit at once, one model per fruit type, and
# reached ~98% held-out accuracy — a real, measured improvement over an
# earlier patch+majority-vote SVM approach's ~73% (that approach had a
# confirmed structural blind spot: a small localized rotten spot gets
# outvoted by the majority of an otherwise-normal-looking surface — a
# synthetic dark spot covering up to 25% of a real Fresh apple never
# flipped the vote to Rotten). If no cnn_quality_models/<FruitType>.pt
# exists for a detected fruit type (or PyTorch isn't installed), that
# object's quality is reported as unavailable with a clear
# ClassificationResult.error message — there is no fallback.
CNN_MODELS_DIR = "cnn_quality_models"
_cnn_model_cache = {}


def _load_cnn_quality_model(fruit_type, models_dir=CNN_MODELS_DIR):
    """Lazily loads+caches the per-fruit-type CNN checkpoint. Returns
    (model, classes) or (None, None) if the weights file doesn't exist
    or PyTorch/train_cnn_quality can't be imported — callers report
    quality as unavailable for that fruit type in that case."""
    key = (os.path.abspath(models_dir) if os.path.isdir(models_dir) else models_dir, fruit_type)
    if key in _cnn_model_cache:
        return _cnn_model_cache[key]

    path = os.path.join(models_dir, f"{fruit_type}.pt")
    if not os.path.isfile(path):
        _cnn_model_cache[key] = (None, None)
        return None, None

    try:
        import train_cnn_quality as _cnn_module
        model, classes = _cnn_module.load_cnn_model(path)
    except Exception as e:  # missing torch, corrupt checkpoint, etc.
        print(f"[colorDetection] CNN quality model at {path} unavailable ({e}); quality will be reported unavailable for '{fruit_type}'.")
        _cnn_model_cache[key] = (None, None)
        return None, None

    _cnn_model_cache[key] = (model, classes)
    return model, classes


def classify_quality_cnn(crop_bgr, fruit_type, models_dir=CNN_MODELS_DIR):
    """
    Classifies Fresh/Unripe/Rotten for one fruit crop with the CNN
    (see train_cnn_quality.py). `crop_bgr` should be a plain
    bounding-box crop of one fruit on its natural background (e.g.
    crop_object(original, det, isolate=False)) — matches what the
    CNN's training photos look like. Returns (label, confidence), or
    (None, 0.0) if no CNN model is available for this fruit type.
    """
    model, classes = _load_cnn_quality_model(fruit_type, models_dir)
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    import train_cnn_quality as _cnn_module
    return _cnn_module.predict_quality(model, classes, crop_bgr)


# ======================================================
# YOLO detection + fruit type
# ======================================================
# Which COCO class names count as "fruit" here — YOLO checkpoints
# pretrained on COCO already include these three as distinct classes,
# with zero labeling/training needed.
YOLO_FRUIT_CLASS_NAMES = {"apple", "banana", "orange"}
YOLO_CONFIDENCE_THRESHOLD = 0.25

_yolo_model_cache = {}


def _load_yolo_model(weights):
    """Caches the loaded YOLO model per weights-path within a process,
    so a Streamlit dashboard calling inspect_image_yolo() once per
    uploaded photo doesn't reload (and re-download, on first use) the
    model every single time."""
    if weights not in _yolo_model_cache:
        from ultralytics import YOLO
        _yolo_model_cache[weights] = YOLO(weights)
    return _yolo_model_cache[weights]


# ======================================================
# Full pipeline: preprocess -> YOLO detect+type -> local re-segment ->
#                calibrate -> CNN classify quality
# ======================================================
def inspect_image_yolo(
    image_or_path,
    calibration: Optional[CalibrationResult] = None,
    image_size=DEFAULT_IMAGE_SIZE,
    denoise_method="median",
    enhance_method="clahe",
    erode_pixels=10,
    yolo_weights="yolov8n.pt",
    yolo_confidence=YOLO_CONFIDENCE_THRESHOLD,
):
    """
    Runs the full inspection pipeline on one photo, which may contain
    MULTIPLE fruits, and returns a plain dict (dashboard/report-friendly):

        {
          "original": np.ndarray,
          "preprocessed": np.ndarray,
          "annotated": np.ndarray,      # every object's bbox+contour+"#N Type Quality" label drawn
          "objects": [
              {
                "index": int,
                "detection": DetectionResult,
                "bbox": (x, y, w, h),
                "area_px"/"width_px"/"height_px": float,
                "area_cm2"/"width_cm"/"height_cm": float | None,
                "classification": ClassificationResult,
                "fruit_type": str | None, "fruit_type_confidence": float,
                "label": str | None,        # quality: Fresh/Unripe/Rotten
                "confidence": float,
                "crop": np.ndarray,          # this fruit cropped out of the original photo
                "crop_isolated": np.ndarray, # same crop, background + neighboring fruits blacked out
              }, ...
          ],
          "summary": {"Apple": {"Fresh": 2, "Rotten": 1}, "Orange": {"Fresh": 1}, ...},
          "count": int,                 # total fruits detected in this photo
          "calibration_method": str,
          "calibration_confidence": str,
        }

    Detection + fruit TYPE come from a pretrained YOLO model (COCO
    already includes apple/banana/orange, so no extra training/
    labeling is needed) — this handles densely packed/heavily
    overlapping same-species fruit meaningfully better than a
    classical distance-transform/Hough/watershed splitter would, since
    it recognizes "this looks like an apple" from learned visual
    texture/shading rather than needing a clean geometric boundary
    between touching objects.

    Requires `pip install ultralytics` (imported lazily inside
    _load_yolo_model(), not at module load time, so importing
    colorDetection.py elsewhere doesn't require it unless this
    function actually gets called).

    Each YOLO box is re-segmented locally (segmentation_mask_and_
    contour — LAB chroma-distance segmentation, imported from
    colorDetection_Train.py) within just that box's crop, to get a
    real contour/mask for measurements and for "crop_isolated" — the
    box alone is just a rectangle, it doesn't say which pixels inside
    it are this fruit vs. a sliver of its neighbor.

    Quality (Fresh/Unripe/Rotten) comes entirely from the CNN via
    classify_quality_cnn() (see train_cnn_quality.py) — YOLO has no
    concept of ripeness, it only knows what a generic apple looks
    like. No fallback: if no CNN model is trained for a detected
    fruit's type, that object's quality comes back as unavailable with
    an error message.
    """
    if isinstance(image_or_path, str):
        original = cv2.imread(image_or_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_or_path}")
    else:
        original = image_or_path

    original = cv2.resize(original, image_size)
    preprocessed = prep.preprocess_image(original, denoise_method=denoise_method, enhance_method=enhance_method)

    if calibration is None:
        calibration = uncalibrated()

    yolo_model = _load_yolo_model(yolo_weights)  # lazy-imports ultralytics; see that function
    yolo_results = yolo_model.predict(original, conf=yolo_confidence, verbose=False)

    from collections import defaultdict

    summary = defaultdict(lambda: defaultdict(int))
    objects = []
    i = 0
    for r in yolo_results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            yolo_label = names.get(cls_id, str(cls_id))
            if yolo_label not in YOLO_FRUIT_CLASS_NAMES:
                continue
            yolo_conf = float(box.conf[0])
            x1f, y1f, x2f, y2f = box.xyxy[0].tolist()

            h_img, w_img = original.shape[:2]
            pad = 0.08
            bw, bh = x2f - x1f, y2f - y1f
            x0 = max(0, int(x1f - pad * bw))
            y0 = max(0, int(y1f - pad * bh))
            x1 = min(w_img, int(x2f + pad * bw))
            y1 = min(h_img, int(y2f + pad * bh))
            if x1 <= x0 or y1 <= y0:
                continue

            # Re-segment WITHIN just this box's crop to get a real
            # contour/mask (the YOLO box is only a rectangle) — reliable
            # once there's roughly one dominant fruit in frame, true
            # almost by construction inside a tight YOLO box, even
            # though it wasn't true for the whole photo.
            crop_for_seg = prep.preprocess_image(original[y0:y1, x0:x1], denoise_method=denoise_method, enhance_method="none")
            local_mask, local_contour = segmentation_mask_and_contour(crop_for_seg)
            if local_contour is None:
                # Fall back to the full box as the contour if local
                # segmentation finds nothing usable (e.g. a very small
                # or oddly-lit crop) — still better than dropping the
                # detection entirely.
                local_contour = np.array([[[0, 0]], [[x1 - x0 - 1, 0]], [[x1 - x0 - 1, y1 - y0 - 1]], [[0, y1 - y0 - 1]]])
                local_mask = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)

            contour = local_contour + [x0, y0]  # shift into full-image coordinates
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            if erode_pixels > 0:
                kernel = np.ones((erode_pixels, erode_pixels), np.uint8)
                mask = cv2.erode(mask, kernel)

            bx, by, bw2, bh2 = cv2.boundingRect(contour)
            det = DetectionResult(
                found=True, bbox=(bx, by, bw2, bh2), contour=contour, mask=mask,
                area_px=float(cv2.contourArea(contour)),
                perimeter_px=float(cv2.arcLength(contour, closed=True)),
            )

            fruit_type = yolo_label.capitalize()  # "apple" -> "Apple", matches train_cnn_quality.py's convention

            classification = ClassificationResult(fruit_type=fruit_type, fruit_type_confidence=yolo_conf)
            cnn_label, cnn_conf = classify_quality_cnn(crop_object(original, det, isolate=False), fruit_type)
            if cnn_label is not None:
                classification.label = cnn_label
                classification.confidence = cnn_conf
            else:
                classification.error = (
                    f"No CNN quality model found for fruit type '{fruit_type}' "
                    f"(train one with train_cnn_quality.py, saved as {CNN_MODELS_DIR}/{fruit_type}.pt)"
                )

            width_px, height_px = float(bw2), float(bh2)
            area_px = det.area_px
            width_cm = height_cm = area_cm2 = None
            if calibration.confidence != "uncalibrated":
                width_cm = calibration.px_to_cm(width_px)
                height_cm = calibration.px_to_cm(height_px)
                area_cm2 = calibration.px_area_to_cm2(area_px)

            objects.append({
                "index": i,
                "detection": det,
                "bbox": det.bbox,
                "area_px": area_px,
                "width_px": width_px,
                "height_px": height_px,
                "area_cm2": area_cm2,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "classification": classification,
                "fruit_type": classification.fruit_type,
                "fruit_type_confidence": classification.fruit_type_confidence,
                "label": classification.label,
                "confidence": classification.confidence,
                "crop": crop_object(original, det, isolate=False),
                "crop_isolated": crop_object(original, det, isolate=True),
            })

            fruit_key = classification.fruit_type or "Unknown"
            quality_key = classification.label or "Unclassified"
            summary[fruit_key][quality_key] += 1
            i += 1

    annotated = draw_detections(preprocessed, objects)

    return {
        "original": original,
        "preprocessed": preprocessed,
        "annotated": annotated,
        "objects": objects,
        "summary": {k: dict(v) for k, v in summary.items()},
        "count": len(objects),
        "calibration_method": calibration.method,
        "calibration_confidence": calibration.confidence,
    }
