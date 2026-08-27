"""
Bridge between the teammate's standalone V12 morphological/texture pipeline
(fruit_v12_hybrid_final_agreement_guard.py -- a script, not an importable
API) and overall.py, which needs a plain function it can call per photo.

This mirrors ASS's own app.py `_load_morph_texture_v12_models()` /
`inspect_image_morph_texture_v12()` wrapper almost line for line (that is
the ONLY place this integration pattern exists -- the v12 module itself is
a standalone `if __name__ == "__main__"` script), but only keeps the fields
overall.py actually consumes: species vote (fruit_type/fruit_type_confidence),
box, geometry (geo_*), texture (tex_*), and size_class. Streamlit-only
extras (annotated image, crop_isolated, etc.) are left out on purpose --
this file is not meant to render anything, only to feed the fusion logic in
overall.py.

V12 replaces the old v10/v11 morph_texture_module.py entirely, matching
ASS's current app.py (which no longer imports morph_texture_module at all):
- pure geometry + texture features (no colour) feeding a Random Forest
- agreement-aware fusion between V12's own YOLO and that feature model,
  with an explicit "unknown" class instead of forcing a guess when both
  sides disagree and neither is confident
- Mango gets a stricter rescue/override bar since it's the weakest class
  for the feature classifier
"""

import fruit_v12_hybrid_final_agreement_guard as morph_v12

_MODEL_CACHE = {
    "yolo": None,
    "feature_model": None,
    "feature_names": None,
    "size_thresholds": None,
}


def _load_models():
    if _MODEL_CACHE["yolo"] is None:
        _MODEL_CACHE["yolo"] = morph_v12.YOLO(str(morph_v12.YOLO_MODEL_PATH))
    if _MODEL_CACHE["feature_model"] is None:
        _MODEL_CACHE["feature_model"] = morph_v12.joblib.load(morph_v12.FEATURE_MODEL_PATH)
        _MODEL_CACHE["feature_names"] = list(morph_v12.joblib.load(morph_v12.FEATURE_NAMES_PATH))
    if _MODEL_CACHE["size_thresholds"] is None:
        _MODEL_CACHE["size_thresholds"] = morph_v12.load_size_thresholds()
    return (
        _MODEL_CACHE["yolo"],
        _MODEL_CACHE["feature_model"],
        _MODEL_CACHE["feature_names"],
        _MODEL_CACHE["size_thresholds"],
    )


def get_active_model_info():
    _load_models()
    return {
        "yolo": str(morph_v12.YOLO_MODEL_PATH),
        "feature_model": str(morph_v12.FEATURE_MODEL_PATH),
        "feature_names": str(morph_v12.FEATURE_NAMES_PATH),
    }


def inspect_image_morph_texture(image):
    """
    Run V12 (its own YOLO localisation + geometry/texture feature
    classifier + agreement-aware fusion) on one whole photo.

    Returns {"objects": [...]}, each object with:
      box (x1,y1,x2,y2), fruit_type (Title-case or None if V12 itself
      couldn't agree on a class), fruit_type_confidence, raw_yolo_type,
      yolo_confidence, feature_type, feature_confidence,
      classification_method, size_class, geo_*, tex_*.
    """
    yolo_model, feature_model, feature_names, size_thresholds = _load_models()

    image_h, image_w = image.shape[:2]

    result = yolo_model.predict(
        source=image,
        conf=morph_v12.YOLO_CONF,
        iou=morph_v12.YOLO_IOU,
        imgsz=morph_v12.YOLO_IMGSZ,
        max_det=morph_v12.MAX_DET,
        verbose=False,
    )[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w, x2), min(image_h, y2)
        current_box = (x1, y1, x2, y2)
        if not morph_v12.valid_box(current_box, image_w, image_h):
            continue
        detections.append({
            "box": current_box,
            "yolo_class": str(yolo_model.names[class_id]).strip().lower(),
            "yolo_confidence": confidence,
        })

    detections = morph_v12.remove_duplicate_detections(detections)

    # Built in the module's own shape first (final_class/size_value) so
    # apply_group_relative_size -- which expects exactly that shape -- can
    # refine S/M/L/XL across multiple same-species fruit in this photo,
    # same as the standalone script and ASS's app.py both do.
    hybrid_objects = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        features, mask, contour = morph_v12.extract_all_features(crop, feature_names)
        if features is None or contour is None:
            continue

        feature_class, feature_confidence, feature_probabilities = morph_v12.predict_feature_class(
            feature_model, feature_names, features
        )
        final_class, final_confidence, decision_method, rejected = morph_v12.fuse_predictions(
            detection["yolo_class"], detection["yolo_confidence"],
            feature_class, feature_confidence, feature_probabilities,
        )
        if rejected:
            continue

        geometry = morph_v12.extract_size_geometry(contour, image.shape)
        if final_class == morph_v12.UNKNOWN_CLASS_NAME:
            threshold_size, size_feature, size_value = "Unknown", None, None
        else:
            threshold_size, size_feature, size_value = morph_v12.classify_dataset_size(
                final_class, geometry, size_thresholds
            )

        hybrid_objects.append({
            "box": detection["box"],
            "yolo_class": detection["yolo_class"],
            "yolo_confidence": detection["yolo_confidence"],
            "feature_class": feature_class,
            "feature_confidence": feature_confidence,
            "final_class": final_class,
            "final_confidence": final_confidence,
            "decision_method": decision_method,
            "features": features,
            "threshold_size": threshold_size,
            "size": threshold_size,
            "size_feature": size_feature,
            "size_value": size_value,
            "relative_ratio": None,
        })

    morph_v12.apply_group_relative_size(hybrid_objects)

    objects = []
    for obj in hybrid_objects:
        is_unknown = obj["final_class"] == morph_v12.UNKNOWN_CLASS_NAME
        features = obj["features"]
        objects.append({
            "box": obj["box"],
            "fruit_type": None if is_unknown else obj["final_class"].title(),
            "fruit_type_confidence": float(obj["final_confidence"]),
            "raw_yolo_type": obj["yolo_class"].title(),
            "yolo_confidence": float(obj["yolo_confidence"]),
            "feature_type": obj["feature_class"].title(),
            "feature_confidence": float(obj["feature_confidence"]),
            "classification_method": obj["decision_method"],
            "size_class": "Unknown" if is_unknown else obj["size"],
            "geo_aspect_ratio": features.get("geo_aspect_ratio"),
            "geo_circularity": features.get("geo_circularity"),
            "geo_extent": features.get("geo_extent"),
            "tex_contrast": features.get("tex_contrast"),
            "tex_energy": features.get("tex_energy"),
            "tex_homogeneity": features.get("tex_homogeneity"),
            "tex_entropy": features.get("tex_entropy"),
            "tex_mean_intensity": features.get("tex_mean_intensity"),
            "tex_std_intensity": features.get("tex_std_intensity"),
        })

    return {"objects": objects}
