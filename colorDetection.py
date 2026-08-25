

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from segmentation import (
    segmentation_mask_and_contour,
    segment_all_objects,
    detect_defect_fraction,
    contour_shape_metrics,
    compute_texture_roughness,
    DEFAULT_DEFECT_AREA_FRACTION,
    DEFAULT_DEFECT_LOW_FRACTION,
)

import preprocessing as prep
from calibration import CalibrationResult, uncalibrated

DEFAULT_IMAGE_SIZE = (512, 512)

FRUIT_DEBUG = os.environ.get("FRUIT_DEBUG", "0") not in ("0", "", "false", "False")


def _dbg(*args):
    if FRUIT_DEBUG:
        print("[FRUIT_DEBUG]", *args)


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


DEDUPE_IOU_THRESHOLD = 0.5  # mask-overlap fraction above which two detections are treated as the same physical fruit


def _dedupe_overlapping_detections(objects, iou_threshold=DEDUPE_IOU_THRESHOLD):
    ranked = sorted(objects, key=lambda o: o["fruit_type_confidence"], reverse=True)
    kept = []
    for obj in ranked:
        mask = obj["detection"].mask
        is_dup = False
        for kept_obj in kept:
            kept_mask = kept_obj["detection"].mask
            if mask is None or kept_mask is None:
                continue
            inter = int(np.logical_and(mask > 0, kept_mask > 0).sum())
            union = int(np.logical_or(mask > 0, kept_mask > 0).sum())
            iou = (inter / union) if union > 0 else 0.0
            if iou > iou_threshold:
                is_dup = True
                _dbg(f"dedupe: DROP {obj.get('fruit_type')} (conf={obj['fruit_type_confidence']:.3f}) "
                     f"-- overlaps {kept_obj.get('fruit_type')} (conf={kept_obj['fruit_type_confidence']:.3f}) "
                     f"at IoU={iou:.2f}")
                break
        if not is_dup:
            kept.append(obj)
    kept.sort(key=lambda o: o["index"])
    return kept


def draw_detections(image, objects, box_color=(0, 255, 0), contour_color=(0, 165, 255)):
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
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


CROP_EXTRA_ERODE_PIXELS = 6  # applied only to isolate=True crops, on top of


def crop_object(image, detection: DetectionResult, pad_frac=0.06, isolate=False):

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
    error: Optional[str] = None             # e.g. "No CNN quality model found for fruit type 'Apple'"
    defect_fraction: float = 0.0            # fraction of the fruit's OWN surface flagged as a dark
    defect_override: bool = False           # True if defect_fraction crossed the threshold and


CNN_MODELS_DIR = "cnn_quality_models"
_cnn_model_cache = {}


def _load_cnn_quality_model(fruit_type, models_dir=CNN_MODELS_DIR):
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
    model, classes = _load_cnn_quality_model(fruit_type, models_dir)
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, {}
    import train_cnn_quality as _cnn_module
    prepared = _cnn_module.prepare_cnn_input_bgr(crop_bgr)
    return _cnn_module.predict_quality_with_probs(model, classes, prepared)


CNN_TYPE_MODELS_DIR = "cnn_type_models"
_cnn_type_model_cache = {}


def _load_cnn_type_model(models_dir=CNN_TYPE_MODELS_DIR):
    key = os.path.abspath(models_dir) if os.path.isdir(models_dir) else models_dir
    if key in _cnn_type_model_cache:
        return _cnn_type_model_cache[key]

    path = os.path.join(models_dir, "fruit_type.pt")
    if not os.path.isfile(path):
        _cnn_type_model_cache[key] = (None, None)
        return None, None

    try:
        import train_fruit_type as _type_module
        model, classes = _type_module.load_type_model(path)
    except Exception as e:  # missing torch, corrupt checkpoint, etc.
        print(f"[colorDetection] CNN fruit-type model at {path} unavailable ({e}); "
              f"fallback (non-YOLO) fruit detection will be skipped.")
        _cnn_type_model_cache[key] = (None, None)
        return None, None

    _cnn_type_model_cache[key] = (model, classes)
    return model, classes


def classify_fruit_type_cnn(crop_bgr, models_dir=CNN_TYPE_MODELS_DIR):
    model, classes = _load_cnn_type_model(models_dir)
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, {}
    import train_cnn_quality as _cnn_module
    import train_fruit_type as _type_module
    prepared = _cnn_module.prepare_cnn_input_bgr(crop_bgr)
    return _type_module.predict_fruit_type_with_probs(model, classes, prepared)


# ======================================================
# YOLO detection + fruit type
# ======================================================
FALLBACK_ONLY_SPECIES = {"Mango", "Strawberry"}

STRAWBERRY_OVERRIDE_MIN_CONF = 0.9   # every real strawberry test photo measured 1.000; no observed false positive at any confidence
MANGO_OVERRIDE_MIN_CONF = 0.6        # lower than strawberry's bar on purpose -- this alone is NOT trusted; see MANGO_OVERRIDE_MIN_ASPECT, both must pass
MANGO_OVERRIDE_MIN_ASPECT = 1.5      # local contour's long/short side ratio. Raised from 1.3 after a live
MANGO_OVERRIDE_MAX_ROUGHNESS = 19.5  # compute_texture_roughness() (segmentation.py) -- mean Sobel gradient
MANGO_OVERRIDE_MIN_CONF_DOUBLE = 0.5  # lower floor used ONLY when BOTH aspect_ok and roughness_ok pass --

YOLO_FRUIT_CLASS_NAMES = {"apple", "banana", "orange", "mango", "strawberry"}
YOLO_CONFIDENCE_THRESHOLD = 0.5

DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"
FINE_TUNED_YOLO_WEIGHTS = os.path.join("yolo_fruit_models", "best.pt")


def _resolve_yolo_weights(requested):
    if requested == DEFAULT_YOLO_WEIGHTS and os.path.isfile(FINE_TUNED_YOLO_WEIGHTS):
        return FINE_TUNED_YOLO_WEIGHTS
    return requested


FINE_TUNED_YOLO_IMGSZ = 416


_yolo_model_cache = {}


def _load_yolo_model(weights):
    if weights not in _yolo_model_cache:
        from ultralytics import YOLO
        _yolo_model_cache[weights] = YOLO(weights)
    return _yolo_model_cache[weights]


def inspect_image_yolo(
    image_or_path,
    calibration: Optional[CalibrationResult] = None,
    image_size=DEFAULT_IMAGE_SIZE,
    denoise_method="median",
    enhance_method="clahe",
    erode_pixels=10,
    yolo_weights=DEFAULT_YOLO_WEIGHTS,
    yolo_confidence=YOLO_CONFIDENCE_THRESHOLD,
):
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

    resolved_weights = _resolve_yolo_weights(yolo_weights)
    yolo_model = _load_yolo_model(resolved_weights)  # lazy-imports ultralytics; see that function
    predict_imgsz = FINE_TUNED_YOLO_IMGSZ if resolved_weights == FINE_TUNED_YOLO_WEIGHTS else 640
    yolo_results = yolo_model.predict(original, conf=yolo_confidence, imgsz=predict_imgsz, verbose=False)

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
            _dbg(f"main loop: YOLO box label={yolo_label} conf={yolo_conf:.3f} "
                 f"xyxy=({x1f:.0f},{y1f:.0f},{x2f:.0f},{y2f:.0f})")

            h_img, w_img = original.shape[:2]
            pad = 0.08
            bw, bh = x2f - x1f, y2f - y1f
            x0 = max(0, int(x1f - pad * bw))
            y0 = max(0, int(y1f - pad * bh))
            x1 = min(w_img, int(x2f + pad * bw))
            y1 = min(h_img, int(y2f + pad * bh))
            if x1 <= x0 or y1 <= y0:
                continue

            crop_for_seg = prep.preprocess_image(original[y0:y1, x0:x1], denoise_method=denoise_method, enhance_method="none")
            local_mask, local_contour = segmentation_mask_and_contour(crop_for_seg)
            used_fallback_rect = local_contour is None
            if used_fallback_rect:
                local_contour = np.array([[[0, 0]], [[x1 - x0 - 1, 0]], [[x1 - x0 - 1, y1 - y0 - 1]], [[0, y1 - y0 - 1]]])
                local_mask = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
                local_aspect = None  # a synthetic rectangle's aspect isn't informative about the real object
                local_circularity = None
            else:
                _local_solidity, local_aspect, local_circularity = contour_shape_metrics(local_contour)

            local_roughness = compute_texture_roughness(crop_for_seg, local_mask)
            _dbg(f"main loop: shape metrics aspect="
                 f"{'n/a' if local_aspect is None else f'{local_aspect:.3f}'} "
                 f"circularity={'n/a' if local_circularity is None else f'{local_circularity:.3f}'} "
                 f"roughness={local_roughness:.1f}")

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

            raw_isolated_crop = crop_object(original, det, isolate=True)

            yolo_species = yolo_label.capitalize()  # "apple" -> "Apple", matches train_cnn_quality.py's convention
            type_label, type_conf, type_probs = classify_fruit_type_cnn(raw_isolated_crop)
            if FRUIT_DEBUG and type_probs:
                probs_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(type_probs.items(), key=lambda kv: -kv[1]))
                _dbg(f"main loop: type-CNN full breakdown: {probs_str}")
            fruit_type, fruit_type_confidence = yolo_species, yolo_conf

            if type_label == "Strawberry" and type_conf >= STRAWBERRY_OVERRIDE_MIN_CONF:
                fruit_type, fruit_type_confidence = type_label, type_conf
                _dbg(f"main loop: OVERRIDE {yolo_species}(conf={yolo_conf:.3f}) -> "
                     f"Strawberry(conf={type_conf:.3f})")
            elif type_label == "Mango" and type_conf >= MANGO_OVERRIDE_MIN_CONF_DOUBLE:
                aspect_ok = local_aspect is not None and local_aspect >= MANGO_OVERRIDE_MIN_ASPECT
                roughness_ok = local_roughness <= MANGO_OVERRIDE_MAX_ROUGHNESS
                both_ok = aspect_ok and roughness_ok
                required_conf = MANGO_OVERRIDE_MIN_CONF_DOUBLE if both_ok else MANGO_OVERRIDE_MIN_CONF
                if type_conf >= required_conf and (aspect_ok or roughness_ok):
                    fruit_type, fruit_type_confidence = type_label, type_conf
                    passed_via = "+".join(g for g, ok in (("aspect", aspect_ok), ("roughness", roughness_ok)) if ok)
                    _dbg(f"main loop: OVERRIDE {yolo_species}(conf={yolo_conf:.3f}) -> "
                         f"Mango(conf={type_conf:.3f}, aspect="
                         f"{'n/a' if local_aspect is None else f'{local_aspect:.3f}'}, "
                         f"roughness={local_roughness:.1f}, passed_via={passed_via}, "
                         f"required_conf={required_conf})")
                else:
                    _dbg(f"main loop: Mango candidate REJECTED (shape+texture+conf gate) {yolo_species}(conf={yolo_conf:.3f}) "
                         f"type-CNN=Mango(conf={type_conf:.3f}, needs >= {required_conf}) aspect="
                         f"{'n/a' if local_aspect is None else f'{local_aspect:.3f}'} "
                         f"(needs >= {MANGO_OVERRIDE_MIN_ASPECT}) roughness={local_roughness:.1f} "
                         f"(needs <= {MANGO_OVERRIDE_MAX_ROUGHNESS})")

            _already_logged_disagreement = type_label == "Mango" and type_conf >= MANGO_OVERRIDE_MIN_CONF_DOUBLE
            if (FRUIT_DEBUG and type_label is not None and type_label != fruit_type
                    and fruit_type == yolo_species and not _already_logged_disagreement):
                _dbg(f"main loop: FINAL label={fruit_type} (YOLO conf={yolo_conf:.3f}) | "
                     f"type-CNN says {type_label} (conf={type_conf:.3f}) -- DISAGREES, not overridden")

            classification = ClassificationResult(fruit_type=fruit_type, fruit_type_confidence=fruit_type_confidence)
            cnn_label, cnn_conf, cnn_probs = classify_quality_cnn(raw_isolated_crop, fruit_type)
            if cnn_label is not None:
                classification.label = cnn_label
                classification.confidence = cnn_conf
            else:
                classification.error = (
                    f"No CNN quality model found for fruit type '{fruit_type}' "
                    f"(train one with train_cnn_quality.py, saved as {CNN_MODELS_DIR}/{fruit_type}.pt)"
                )

            denoised_only = prep.denoise(original, method=denoise_method)
            defect_fraction = detect_defect_fraction(denoised_only, det.mask)
            classification.defect_fraction = defect_fraction
            if classification.label is not None:
                if defect_fraction >= DEFAULT_DEFECT_AREA_FRACTION:
                    classification.label = "Rotten"
                    classification.defect_override = True
                elif classification.label == "Rotten" and defect_fraction < DEFAULT_DEFECT_LOW_FRACTION:
                    alt_candidates = {k: v for k, v in cnn_probs.items() if k != "Rotten"}
                    if alt_candidates:
                        alt_label = max(alt_candidates, key=alt_candidates.get)
                        classification.label = alt_label
                        classification.confidence = alt_candidates[alt_label]
                        classification.defect_override = True

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
                "defect_fraction": classification.defect_fraction,
                "defect_override": classification.defect_override,
                "crop": crop_object(original, det, isolate=False),
                "crop_isolated": crop_object(original, det, isolate=True),
            })

            fruit_key = classification.fruit_type or "Unknown"
            quality_key = classification.label or "Unclassified"
            summary[fruit_key][quality_key] += 1
            i += 1

    fallback_type_model, _fallback_type_classes = _load_cnn_type_model()
    if fallback_type_model is not None:
        for blob in segment_all_objects(original):
            bx, by, bw3, bh3 = blob["bbox"]
            cx, cy = bx + bw3 // 2, by + bh3 // 2

            already_claimed = any(
                obj["detection"].mask is not None
                and 0 <= cy < obj["detection"].mask.shape[0]
                and 0 <= cx < obj["detection"].mask.shape[1]
                and obj["detection"].mask[cy, cx] > 0
                for obj in objects
            )
            if already_claimed:
                _dbg(f"fallback: SKIP (already_claimed by YOLO) bbox={blob['bbox']}")
                continue

            det = DetectionResult(
                found=True, bbox=blob["bbox"], contour=blob["contour"], mask=blob["mask"],
                area_px=blob["area_px"],
                perimeter_px=float(cv2.arcLength(blob["contour"], closed=True)),
            )

            raw_isolated_crop = crop_object(original, det, isolate=True)
            type_label, type_conf, type_probs = classify_fruit_type_cnn(raw_isolated_crop)
            if FRUIT_DEBUG and type_probs:
                probs_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(type_probs.items(), key=lambda kv: -kv[1]))
                _dbg(f"fallback: type-CNN full breakdown: {probs_str}")

            _blob_solidity, blob_aspect, blob_circularity = contour_shape_metrics(blob["contour"])
            blob_roughness = compute_texture_roughness(prep.denoise(original, method=denoise_method), blob["mask"])
            _dbg(f"fallback: bbox={blob['bbox']} type-CNN says {type_label} (conf={type_conf:.3f}) "
                 f"aspect={blob_aspect:.3f} circularity={blob_circularity:.3f} roughness={blob_roughness:.1f}")
            if type_label is None:
                continue
            if type_label not in FALLBACK_ONLY_SPECIES:
                continue

            required_conf = STRAWBERRY_OVERRIDE_MIN_CONF if type_label == "Strawberry" else MANGO_OVERRIDE_MIN_CONF
            if type_conf < required_conf:
                _dbg(f"fallback: REJECT (confidence floor) bbox={blob['bbox']} "
                     f"type-CNN={type_label}(conf={type_conf:.3f}) needs >= {required_conf}")
                continue

            classification = ClassificationResult(fruit_type=type_label, fruit_type_confidence=type_conf)
            cnn_label, cnn_conf, cnn_probs = classify_quality_cnn(raw_isolated_crop, type_label)
            if cnn_label is not None:
                classification.label = cnn_label
                classification.confidence = cnn_conf
            else:
                classification.error = (
                    f"No CNN quality model found for fruit type '{type_label}' "
                    f"(train one with train_cnn_quality.py, saved as {CNN_MODELS_DIR}/{type_label}.pt)"
                )

            denoised_only = prep.denoise(original, method=denoise_method)
            defect_fraction = detect_defect_fraction(denoised_only, det.mask)
            classification.defect_fraction = defect_fraction
            if classification.label is not None:
                if defect_fraction >= DEFAULT_DEFECT_AREA_FRACTION:
                    classification.label = "Rotten"
                    classification.defect_override = True
                elif classification.label == "Rotten" and defect_fraction < DEFAULT_DEFECT_LOW_FRACTION:
                    alt_candidates = {k: v for k, v in cnn_probs.items() if k != "Rotten"}
                    if alt_candidates:
                        alt_label = max(alt_candidates, key=alt_candidates.get)
                        classification.label = alt_label
                        classification.confidence = alt_candidates[alt_label]
                        classification.defect_override = True

            width_px, height_px = float(bw3), float(bh3)
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
                "defect_fraction": classification.defect_fraction,
                "defect_override": classification.defect_override,
                "crop": crop_object(original, det, isolate=False),
                "crop_isolated": raw_isolated_crop,
            })

            fruit_key = classification.fruit_type or "Unknown"
            quality_key = classification.label or "Unclassified"
            summary[fruit_key][quality_key] += 1
            i += 1

    objects = _dedupe_overlapping_detections(objects)
    summary = defaultdict(lambda: defaultdict(int))
    for new_i, obj in enumerate(objects):
        obj["index"] = new_i
        fruit_key = obj["fruit_type"] or "Unknown"
        quality_key = obj["label"] or "Unclassified"
        summary[fruit_key][quality_key] += 1

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
