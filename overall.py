
"""
Overall analysis orchestrator -- ties the independently-built modules
(colour, fruit type, defect, stem; morphological is not implemented yet)
together into one combined per-fruit result.

This file is NEW integration code written for the "overall" system. It does
not modify any of the copied module files in common/, colour/, fruit_type/,
defect/, or stem/ -- it only imports and calls their existing public
functions.
"""

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

FRUIT_DEBUG = os.environ.get("FRUIT_DEBUG", "0") not in ("0", "", "false", "False")


def _dbg(*args):
    if FRUIT_DEBUG:
        print("[FRUIT_DEBUG]", *args)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("common", "colour", "fruit_type", "defect", "morphological"):
    _p = os.path.join(BASE_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocessing as prep                                   # common/
from segmentation import (                                     # common/
    segmentation_mask_and_contour,
    segment_all_objects,
    contour_shape_metrics,
    compute_texture_roughness,
)
import train_color_knn as color_knn_module                     # colour/
import train_fruit_type as fruit_type_module                   # fruit_type/
import defect_detection                                        # defect/
import ripeness_detection                                      # defect/ (defect's own ripeness rule)
from stem.detector import StemDetector                         # stem/ (package)

try:
    # V12 (fruit_v12_hybrid_final_agreement_guard.py, wrapped by
    # morph_v12_bridge.py) replaces the old v10/v11 morph_texture_module.py
    # entirely, matching ASS's current app.py (which no longer imports
    # morph_texture_module at all). Aliased to the old name so every
    # downstream reference below keeps working unchanged.
    import morph_v12_bridge as morph_texture_module              # morphological/
    _MORPH_IMPORT_ERROR = None
except Exception as _morph_import_exc:                          # pragma: no cover
    morph_texture_module = None
    _MORPH_IMPORT_ERROR = str(_morph_import_exc)

# ---------------------------------------------------------------
# Species detection: ported from ASS's CURRENT app.py (run_exact_latest_
# defect_pipeline / choose_cnn_override), which has evolved well past
# colorDetection.py. Key change: the primary detector is now a YOLO model
# fine-tuned on the project's own photos (fruit_yolo_v4, apple/banana/orange
# only) instead of generic COCO yolov8n.pt -- it localises touching/occluded
# fruit far better, which is why ASS separates 3 touching strawberries into
# 3 boxes while yolov8n.pt + classical segmentation could not.
# ---------------------------------------------------------------
YOLO_FRUIT_CLASS_NAMES = {"apple", "banana", "orange", "mango", "strawberry"}
FRUIT_YOLO_V4_WEIGHTS = os.path.join(BASE_DIR, "defect", "weights", "fruit_yolo_v4_best.pt")
YOLO_CONF = 0.25
YOLO_IOU = 0.45
# colorDetection.py's analyze_image() resizes every input to this fixed size
# BEFORE running YOLO or segmentation -- every pixel-based constant in this
# file (10px/6px erosion, roughness/aspect thresholds, segment_all_objects'
# own area/peak-window fractions) was tuned against images at this size.
DEFAULT_IMAGE_SIZE = (512, 512)

# Applied AFTER choose_cnn_override, only if species is still Apple/Banana
# (i.e. the CNN did not override it to Mango/Strawberry). No extra floor for
# Orange -- a badly rotten orange may legitimately score lower confidence.
APPLE_MIN_CONF = 0.85
BANANA_MIN_CONF = 0.55

# choose_cnn_override: four independent Strawberry acceptance rules (normal /
# strong-red / green-unripe / damaged) plus one Mango rule, each using CNN
# confidence + red-pixel-fraction + shape (aspect, circularity) as a safety
# gate -- because this YOLO has no Mango/Strawberry class at all, so it can
# only ever guess Apple/Banana/Orange on those, sometimes confidently wrong.
STRAWBERRY_MIN_CONF = 0.95
STRAWBERRY_MAX_CIRCULARITY = 0.82
STRAWBERRY_MIN_RED_RATIO = 0.70

STRAWBERRY_STRONG_RED_RATIO = 0.80
STRAWBERRY_STRONG_RED_MIN_CONF = 0.90

STRAWBERRY_GREEN_MIN_CONF = 0.98
STRAWBERRY_GREEN_MAX_RED_RATIO = 0.15
STRAWBERRY_GREEN_MAX_CIRCULARITY = 0.65
STRAWBERRY_GREEN_MIN_ASPECT = 1.08

STRAWBERRY_DAMAGED_MIN_CONF = 0.97
STRAWBERRY_DAMAGED_MIN_RED_RATIO = 0.15
STRAWBERRY_DAMAGED_MAX_RED_RATIO = 0.70
STRAWBERRY_DAMAGED_MAX_CIRCULARITY = 0.72
STRAWBERRY_DAMAGED_MIN_ASPECT = 1.02

MANGO_MIN_CONF = 0.75  # lowered from 0.85 -- a real unripe/green mango scored 0.797 (aspect/circularity both passed) and was wrongly rejected
MANGO_MIN_ASPECT = 1.18
MANGO_MAX_CIRCULARITY = 0.90

# NOT part of the ASS port -- added per explicit instruction. choose_cnn_override
# only ever corrects YOLO into Mango/Strawberry (classes it was never trained
# on); it has no rule for YOLO confusing two classes it DOES know (e.g. a real
# green Apple boxed as Orange). This lets the CNN win in THAT situation too,
# but only when it is both very confident on its own AND clearly more
# confident than YOLO -- a real photo scored CNN=Apple(0.99) vs YOLO=Orange
# (0.74), gap=0.25; these numbers are a starting point from that one photo.
KNOWN_CLASS_OVERRIDE_MIN_CNN_CONF = 0.90
KNOWN_CLASS_OVERRIDE_MIN_GAP = 0.15

# Duplicate/fragment YOLO box removal: drop a new box if its centre falls
# inside an already-accepted box AND its area is under this fraction of it.
DUPLICATE_FRAGMENT_MAX_AREA_RATIO = 0.35
# Drop boxes smaller than this fraction of the whole image as noise.
MIN_BOX_AREA_RATIO = 0.002

FALLBACK_ONLY_SPECIES = {"Mango", "Strawberry"}

DEFECT_SUPPORTED_SPECIES = {"Apple", "Banana", "Orange", "Mango", "Strawberry"}  # defect_detection.py's dispatcher now covers all 5

# ---------------------------------------------------------------
# Final quality fusion: two independent ripeness detectors are run on
# every fruit --
#   1. The user's own colour-KNN model (colour/train_color_knn.py)
#   2. The defect module's own rule-based ripeness classifier
#      (defect/ripeness_detection.py), which already folds each species'
#      own defect-percentage thresholds into its verdict.
# Whichever one reports the higher confidence for THIS specific photo wins.
# defect/ripeness_detection.py has no native probability, so its confidence
# is derived from numbers it already computes: for Ripe/Unripe, the
# dominant colour-percentage it based its own decision on; for Overripe,
# the defect percentage itself (normalised so 20% defect == full confidence,
# a starting point to tune against real photos, not a measured constant).
# ---------------------------------------------------------------
_RIPENESS_TO_QUALITY = {"Ripe": "Fresh", "Unripe": "Unripe", "Overripe": "Rotten"}
RIPENESS_RULE_SPECIES = {"Apple", "Banana", "Orange", "Strawberry", "Mango"}  # ripeness_detection.py now covers Mango too (classify_mango)
OVERRIPE_FULL_CONFIDENCE_DEFECT_PCT = 20.0

# Agreed rule: colour alone is NOT allowed to call something Rotten -- the
# defect side must confirm it (either its ripeness rule says Overripe, or
# raw defect percentage clears this floor, which covers Mango since it has
# no ripeness rule). This is a starting point to tune against real photos.
DEFECT_CONFIRMS_DEFECT_PCT = 3.0

# When colour KNN and the defect ripeness rule are compared head-to-head and
# BOTH end up under this confidence, neither number is meaningful (e.g. "1%
# vs 0%") -- flag the result as "Uncertain" instead of picking whichever is
# barely higher.
LOW_CONFIDENCE_THRESHOLD = 0.10

STEM_EXPECTED_SPECIES = {"Mango", "Apple", "Strawberry"}  # stem is only meaningful evidence for these

_yolo_model_cache = {}
_type_model_cache = {}
_color_model_cache = {}
_stem_detector_cache = {}


def _load_yolo(weights=FRUIT_YOLO_V4_WEIGHTS):
    if weights not in _yolo_model_cache:
        from ultralytics import YOLO
        _yolo_model_cache[weights] = YOLO(weights)
    return _yolo_model_cache[weights]


def calculate_red_ratio(roi, mask=None):
    """
    Estimate how much of the fruit surface is strawberry-red. Used only as a
    safety gate so a green/orange citrus fruit can't become Strawberry just
    because the CNN is confident. Ported from ASS's app.py, unchanged.
    """
    if roi is None or roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 90, 45]), np.array([8, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 90, 45]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_low, red_high)
    if mask is not None and mask.shape[:2] == roi.shape[:2] and cv2.countNonZero(mask) > 0:
        red_mask = cv2.bitwise_and(red_mask, mask)
        total = cv2.countNonZero(mask)
    else:
        total = roi.shape[0] * roi.shape[1]
    if total <= 0:
        return 0.0
    return cv2.countNonZero(red_mask) / total


def get_shape_values(contour, roi_width, roi_height):
    """Orientation-independent aspect ratio and circularity. Ported from ASS's app.py, unchanged."""
    aspect = max(roi_width / max(1, roi_height), roi_height / max(1, roi_width))
    circularity = None
    if contour is not None:
        try:
            _sol, contour_aspect, contour_circularity = contour_shape_metrics(contour)
            if contour_aspect is not None:
                contour_aspect = float(contour_aspect)
                aspect = max(contour_aspect, 1.0 / max(contour_aspect, 1e-6))
            circularity = contour_circularity
        except Exception:
            pass
    return aspect, circularity


def choose_cnn_override(yolo_confidence, cnn_type, cnn_conf, aspect, circularity, red_ratio):
    """
    Safely allow Mango/Strawberry to replace a weak YOLO guess.

    fruit_yolo_v4 was trained only on Apple/Banana/Orange, so a real Mango
    or Strawberry can score high confidence as one of those three -- YOLO
    confidence alone must not be trusted to block the CNN here. Instead each
    species has its own confidence + colour + shape safety gate. Ported
    verbatim from ASS's app.py (choose_cnn_override), unchanged.
    """
    strawberry_normal_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_MIN_CONF
        and red_ratio >= STRAWBERRY_MIN_RED_RATIO
        and (circularity is None or circularity < 0.78)
    )
    strawberry_strong_red_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_STRONG_RED_MIN_CONF
        and red_ratio >= STRAWBERRY_STRONG_RED_RATIO
        and (circularity is None or circularity < 0.78)
    )
    strawberry_green_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_GREEN_MIN_CONF
        and red_ratio <= STRAWBERRY_GREEN_MAX_RED_RATIO
        and aspect >= STRAWBERRY_GREEN_MIN_ASPECT
        and circularity is not None and circularity < STRAWBERRY_GREEN_MAX_CIRCULARITY
    )
    strawberry_damaged_ok = (
        cnn_type == "Strawberry" and cnn_conf >= STRAWBERRY_DAMAGED_MIN_CONF
        and STRAWBERRY_DAMAGED_MIN_RED_RATIO <= red_ratio < STRAWBERRY_DAMAGED_MAX_RED_RATIO
        and aspect >= STRAWBERRY_DAMAGED_MIN_ASPECT
        and circularity is not None and circularity < STRAWBERRY_DAMAGED_MAX_CIRCULARITY
    )
    if strawberry_normal_ok or strawberry_strong_red_ok or strawberry_green_ok or strawberry_damaged_ok:
        return "strawberry"

    if (
        cnn_type == "Mango" and cnn_conf >= MANGO_MIN_CONF
        and aspect >= MANGO_MIN_ASPECT
        and (circularity is None or circularity < MANGO_MAX_CIRCULARITY)
    ):
        return "mango"

    return None


def _load_type_model():
    path = os.path.join(BASE_DIR, "fruit_type", "cnn_type_models", "fruit_type.pt")
    if path not in _type_model_cache:
        _type_model_cache[path] = fruit_type_module.load_type_model(path)
    return _type_model_cache[path]


def _load_color_model(fruit_type):
    path = os.path.join(BASE_DIR, "colour", "color_knn_models", f"{fruit_type}.joblib")
    if path not in _color_model_cache:
        if os.path.isfile(path):
            _color_model_cache[path] = color_knn_module.load_color_knn_model(path)
        else:
            _color_model_cache[path] = (None, None, None)
    return _color_model_cache[path]


def _load_stem_detector():
    if "default" not in _stem_detector_cache:
        weights = os.path.join(BASE_DIR, "stem", "weights", "best.pt")
        _stem_detector_cache["default"] = StemDetector(model_path=weights)
    return _stem_detector_cache["default"]


@dataclass
class FruitResult:
    bbox: tuple
    crop: np.ndarray
    species: Optional[str] = None
    species_confidence: float = 0.0
    species_source: Optional[str] = None
    yolo_species: Optional[str] = None
    yolo_confidence: float = 0.0
    cnn_species: Optional[str] = None
    cnn_confidence: float = 0.0
    own_species: Optional[str] = None
    own_confidence: float = 0.0
    morph_fruit_type: Optional[str] = None
    morph_fruit_type_confidence: float = 0.0
    colour_quality: Optional[str] = None
    colour_confidence: float = 0.0
    defect_percentage: Optional[float] = None
    defect_note: str = ""
    defect_ripeness: Optional[str] = None
    defect_ripeness_confidence: float = 0.0
    stem_detected: bool = False
    stem_confidence: float = 0.0
    stem_crop: Optional[np.ndarray] = None  # the crop stem detection ran on, with its own box/contour drawn on it
    morphological_note: str = "Not matched yet"
    morph_aspect_ratio: Optional[float] = None
    morph_circularity: Optional[float] = None
    morph_extent: Optional[float] = None
    morph_size_class: Optional[str] = None
    tex_contrast: Optional[float] = None
    tex_energy: Optional[float] = None
    tex_homogeneity: Optional[float] = None
    tex_entropy: Optional[float] = None
    tex_mean_intensity: Optional[float] = None
    tex_std_intensity: Optional[float] = None
    final_quality: Optional[str] = None
    final_quality_confidence: float = 0.0
    quality_note: str = ""


def _classify_species(crop_bgr, yolo_label, yolo_conf, local_contour, morph_species=None, morph_confidence=0.0):
    """
    Species is decided by THREE independent algorithms, each reported with
    its own percentage, and decided by MAJORITY VOTE across them (not just
    "take the highest confidence" -- explicit instruction):

      1. "own"  -- fruit_yolo_v4 + the CNN (train_fruit_type), fused via the
         tuned choose_cnn_override/KNOWN_CLASS_OVERRIDE gates ported from
         ASS's app.py. This pair stays fused as ONE candidate, not split
         apart into two raw numbers, because it is already-integrated
         tuning of our own system, not a comparison against someone else's
         separate algorithm.
      2. "yolo_raw" -- fruit_yolo_v4's own raw guess/confidence on its own,
         kept as a separate candidate too (per explicit instruction) in case
         the fused/overridden "own" result ends up lower-confidence than
         plain YOLO would have been by itself.
      3. "morph" -- the teammate's independent morphological/texture module
         (V12: its own YOLO + pure geometry/texture feature classifier,
         fused internally with an agreement-aware Unknown guard). A
         genuinely separate algorithm, compared like "yolo_raw", not
         folded into "own".

    Whichever species label at least 2 of the (up to 3) candidates agree on
    wins, using the highest confidence among the agreeing candidates. Only
    when all candidates disagree (no majority possible) does it fall back
    to the single highest-confidence candidate, since that is the only
    signal left to break a 3-way (or 1-vs-1) tie.
    """
    model, classes = _load_type_model()
    type_label, type_conf, _probs = fruit_type_module.predict_fruit_type_with_probs(model, classes, crop_bgr)

    yolo_species, yolo_confidence = yolo_label.capitalize(), yolo_conf
    own_species, own_confidence = yolo_species, yolo_confidence

    roi_h, roi_w = crop_bgr.shape[:2]
    aspect, circularity = get_shape_values(local_contour, roi_w, roi_h)
    red_ratio = calculate_red_ratio(crop_bgr)

    override = choose_cnn_override(yolo_confidence, type_label, type_conf, aspect, circularity, red_ratio)
    if override == "strawberry":
        own_species, own_confidence = "Strawberry", type_conf
    elif override == "mango":
        own_species, own_confidence = "Mango", type_conf
    elif (
        type_label in ("Apple", "Banana", "Orange")
        and type_label != yolo_species
        and type_conf >= KNOWN_CLASS_OVERRIDE_MIN_CNN_CONF
        and (type_conf - yolo_confidence) >= KNOWN_CLASS_OVERRIDE_MIN_GAP
    ):
        # NOT part of the ASS port -- YOLO knows these 3 classes natively, but
        # can still misidentify one as another (e.g. a green Apple boxed as
        # Orange). Only override when the CNN is both very confident AND
        # clearly beats YOLO's own confidence, not on a marginal disagreement.
        own_species, own_confidence = type_label, type_conf
        override = f"known_class:{type_label.lower()}"

    _dbg(f"_classify_species: yolo={yolo_species}({yolo_confidence:.3f}) cnn={type_label}({type_conf:.3f}) "
         f"aspect={aspect:.3f} circularity={'n/a' if circularity is None else f'{circularity:.3f}'} "
         f"red_ratio={red_ratio:.3f} -> override={override} own={own_species}({own_confidence:.3f})")
    if type_label == "Mango" and override != "mango":
        _dbg(f"_classify_species: Mango REJECTED -- needs cnn_conf>={MANGO_MIN_CONF} (got {type_conf:.3f}), "
             f"aspect>={MANGO_MIN_ASPECT} (got {aspect:.3f}), "
             f"circularity<{MANGO_MAX_CIRCULARITY} (got {'n/a' if circularity is None else f'{circularity:.3f}'})")

    # Strawberry exception (explicit instruction): fruit_yolo_v4 never learned
    # Strawberry natively, and morph_texture_module's own YOLO can be shaky on
    # it too -- once the tuned choose_cnn_override strawberry gate (four
    # sub-rules: normal/strong-red/green-unripe/damaged) has already decided
    # "own" IS a Strawberry, trust that directly and skip the majority vote,
    # instead of letting a less-reliable morph guess pull it away again.
    if own_species == "Strawberry":
        species, confidence, source = own_species, own_confidence, "own_strawberry_priority"
        _dbg(f"_classify_species: own={own_species}({own_confidence:.3f}) is Strawberry -- "
             f"skipping majority vote, using own directly (source={source})")
        return {
            "species": species,
            "species_confidence": confidence,
            "species_source": source,
            "yolo_species": yolo_species,
            "yolo_confidence": yolo_confidence,
            "cnn_species": type_label,
            "cnn_confidence": type_conf,
            "own_species": own_species,
            "own_confidence": own_confidence,
            "morph_species": morph_species,
            "morph_confidence": morph_confidence,
        }

    candidates = [
        ("own", own_species, own_confidence),
        ("yolo_raw", yolo_species, yolo_confidence),
    ]
    if morph_species is not None:
        candidates.append(("morph", morph_species, morph_confidence))

    # Majority vote across the (up to 3) candidates, not just "take the
    # highest confidence" -- per explicit instruction. Whichever species
    # label at least 2 of them agree on wins; its confidence is the
    # highest confidence among the agreeing candidates. If all candidates
    # disagree (a 3-way split, or a 1-vs-1 tie when morph has no match),
    # there is no majority, so the highest single confidence is the only
    # signal left to break the tie.
    vote_counts = Counter(sp for _, sp, _ in candidates)
    top_species, top_votes = vote_counts.most_common(1)[0]

    if top_votes >= 2:
        agreeing = [(s, sp, c) for s, sp, c in candidates if sp == top_species]
        species = top_species
        confidence = max(c for _, _, c in agreeing)
        source = "majority(" + "+".join(s for s, _, _ in agreeing) + ")"
    else:
        source, species, confidence = max(candidates, key=lambda c: c[2])
        source = f"no_majority_highest_conf:{source}"

    _dbg(f"_classify_species: candidates=[{', '.join(f'{s}={sp}({c:.3f})' for s, sp, c in candidates)}] "
         f"-> winner={source} final={species}({confidence:.3f})")

    return {
        "species": species,
        "species_confidence": confidence,
        "species_source": source,
        "yolo_species": yolo_species,
        "yolo_confidence": yolo_confidence,
        "cnn_species": type_label,
        "cnn_confidence": type_conf,
        "own_species": own_species,
        "own_confidence": own_confidence,
        "morph_species": morph_species,
        "morph_confidence": morph_confidence,
    }


def _defect_ripeness_confidence(ripeness, colour1, colour2, defect_pct):
    """
    Derive a 0-1 confidence for defect/ripeness_detection.py's rule-based
    verdict, using numbers it already computed (not invented):
    - Overripe verdicts are driven by defect percentage crossing a
      per-species threshold inside classify_ripeness itself, so defect
      percentage is the evidence -- normalised against a 20% reference.
    - Ripe/Unripe verdicts are driven by whichever colour percentage
      (colour1 or colour2) dominated the decision.
    """
    if ripeness == "Overripe":
        return min(1.0, (defect_pct or 0.0) / OVERRIPE_FULL_CONFIDENCE_DEFECT_PCT)
    if ripeness not in _RIPENESS_TO_QUALITY:
        return 0.0
    return min(1.0, max(colour1, colour2) / 100.0)


def _best_non_rotten(colour_probs, colour_label, colour_conf):
    """
    Fallback when colour alone says Rotten but the defect side found no
    corroborating defect: pick colour's own best NON-Rotten class instead of
    trusting the Rotten call. Mirrors the original apple-rotten-guard idea
    in colorDetection.py (_best_non_rotten_quality) using the probabilities
    the colour KNN already computed -- nothing invented.
    """
    candidates = {k: v for k, v in (colour_probs or {}).items() if k != "Rotten"}
    if not candidates:
        return colour_label, colour_conf
    label = max(candidates, key=candidates.get)
    total = sum(candidates.values())
    conf = candidates[label] / total if total > 1e-9 else 0.0
    return label, float(conf)


def _fuse_quality(species, colour_label, colour_conf, colour_probs, defect_label, defect_conf, defect_pct, stem_detected):
    """
    Run TWO independent ripeness detectors -- the user's own colour-KNN and
    the defect module's own rule-based ripeness classifier -- and trust
    whichever produced the higher confidence for this specific photo.

    Agreed rule: colour is NOT allowed to call something Rotten on its own.
    The defect side must confirm it (its ripeness rule says Overripe, or the
    raw defect percentage clears DEFECT_CONFIRMS_DEFECT_PCT -- the latter
    covers Mango, which has no ripeness rule). If colour says Rotten but
    defect finds nothing, colour's own best non-Rotten class is used instead.

    Stem presence/absence is only meaningful for Mango, Apple and
    Strawberry; for other species it is not used as quality evidence.
    Having a stem is a small positive signal, not a hard override.
    """
    notes = []

    # When the defect module HAS a ripeness verdict for this species (Apple/
    # Banana/Orange/Strawberry), that verdict is authoritative -- it already
    # applied its own per-species defect% threshold (e.g. strawberry needs
    # 8-15%, not a flat number) and concluded, so a generic floor here must
    # not override an explicit "Fresh"/"Unripe" call. The flat percentage
    # floor is only a fallback for species with no ripeness rule (Mango).
    if defect_label is not None:
        defect_confirms_defect = defect_label == "Rotten"
    else:
        defect_confirms_defect = defect_pct is not None and defect_pct >= DEFECT_CONFIRMS_DEFECT_PCT

    if colour_label == "Rotten" and not defect_confirms_defect:
        fallback_label, fallback_conf = _best_non_rotten(colour_probs, colour_label, colour_conf)
        pct_str = f"{defect_pct:.1f}%" if defect_pct is not None else "n/a"
        notes.append(f"colour KNN said Rotten but defect found nothing (defect%={pct_str}) -- downgraded to {fallback_label}")
        colour_label, colour_conf = fallback_label, fallback_conf

    if colour_label is None and defect_label is None:
        final_label, final_conf = None, 0.0
    elif defect_label is None:
        final_label, final_conf = colour_label, colour_conf
    elif colour_label is None:
        final_label, final_conf = defect_label, defect_conf
        notes.append(f"defect ripeness rule only ({defect_conf * 100:.0f}%)")
    else:
        if defect_conf > colour_conf:
            winner_label, winner_conf, winner_name = defect_label, defect_conf, "defect ripeness rule"
            loser_conf, loser_name = colour_conf, "colour KNN"
        else:
            winner_label, winner_conf, winner_name = colour_label, colour_conf, "colour KNN"
            loser_conf, loser_name = defect_conf, "defect ripeness rule"

        # Both sides can end up with near-zero confidence at once -- e.g.
        # colour KNN's Rotten call got downgraded (leaving almost no
        # probability for any other class) while the defect ripeness
        # rule's own colour-percentage confidence also happened to be
        # tiny. Picking whichever is barely higher (1% vs 0%) is not a
        # real decision -- it's noise. Flag it instead of pretending
        # either number means something.
        if max(colour_conf, defect_conf) < LOW_CONFIDENCE_THRESHOLD:
            final_label, final_conf = "Uncertain", max(colour_conf, defect_conf)
            notes.append(
                f"LOW CONFIDENCE on both sides ({winner_name} {winner_conf * 100:.0f}% vs {loser_name} "
                f"{loser_conf * 100:.0f}%, neither clears {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%) -- "
                f"not enough evidence to trust either signal, flagged for manual review instead of "
                f"picking the barely-higher one"
            )
        else:
            final_label, final_conf = winner_label, winner_conf
            notes.append(f"{winner_name} won ({winner_conf * 100:.0f}% vs {loser_name} {loser_conf * 100:.0f}%)")

    if colour_label is not None and defect_label is not None and colour_label != defect_label:
        notes.append(f"disagreement: colour KNN={colour_label}, defect rule={defect_label}")

    if species in STEM_EXPECTED_SPECIES:
        notes.append("Stem present" if stem_detected else "Stem not detected")

    return final_label, final_conf, "; ".join(notes)


def analyse_fruit(crop_bgr, yolo_label, yolo_conf, mask=None, contour=None, roughness=None, raw_crop_bgr=None,
                   morph_species=None, morph_confidence=0.0, morph_obj=None, morph_iou=0.0, morph_status=None):
    """Run every available module on one already-cropped fruit and return a
    combined FruitResult.

    crop_bgr is the ISOLATED crop (background blackened, cropped tight to the
    contour) -- used for species CNN and colour KNN, matching ASS's
    colorDetection.py (classify_fruit_type_cnn / classify_quality_color_knn
    both take raw_isolated_crop).

    raw_crop_bgr is the RAW, un-isolated rectangular crop (falls back to
    crop_bgr if not given) -- used for defect_detection/ripeness_detection,
    matching ASS's CURRENT app.py, which explicitly warns that a calibrated/
    black-background crop "can make healthy peel look defective" for those
    two modules specifically.

    morph_species/morph_confidence: this fruit's matched result (if any)
    from the teammate's independent morphological/texture module, already
    IoU-matched by the caller against that module's own whole-image
    detection pass -- fed into _classify_species as the third candidate.
    morph_obj/morph_iou are used purely to attach the geometry/texture
    numbers (aspect ratio, circularity, GLCM texture) for display; they do
    NOT affect final_quality.
    """
    if raw_crop_bgr is None:
        raw_crop_bgr = crop_bgr
    result = FruitResult(bbox=(0, 0, crop_bgr.shape[1], crop_bgr.shape[0]), crop=crop_bgr)

    species_out = _classify_species(crop_bgr, yolo_label, yolo_conf, contour,
                                     morph_species=morph_species, morph_confidence=morph_confidence)
    species = species_out["species"]
    result.species = species
    result.species_confidence = species_out["species_confidence"]
    result.species_source = species_out["species_source"]
    result.yolo_species = species_out["yolo_species"]
    result.yolo_confidence = species_out["yolo_confidence"]
    result.cnn_species = species_out["cnn_species"]
    result.cnn_confidence = species_out["cnn_confidence"]
    result.own_species = species_out["own_species"]
    result.own_confidence = species_out["own_confidence"]
    result.morph_fruit_type = species_out["morph_species"]
    result.morph_fruit_type_confidence = species_out["morph_confidence"]

    if morph_status is not None:
        result.morphological_note = morph_status
    elif morph_obj is not None:
        result.morph_aspect_ratio = morph_obj.get("geo_aspect_ratio")
        result.morph_circularity = morph_obj.get("geo_circularity")
        result.morph_extent = morph_obj.get("geo_extent")
        result.morph_size_class = morph_obj.get("size_class")
        result.tex_contrast = morph_obj.get("tex_contrast")
        result.tex_energy = morph_obj.get("tex_energy")
        result.tex_homogeneity = morph_obj.get("tex_homogeneity")
        result.tex_entropy = morph_obj.get("tex_entropy")
        result.tex_mean_intensity = morph_obj.get("tex_mean_intensity")
        result.tex_std_intensity = morph_obj.get("tex_std_intensity")
        result.morphological_note = f"Matched morphological/texture detection (IoU={morph_iou:.2f})"
    else:
        result.morphological_note = "No matching morphological/texture detection for this box"

    scaler, knn_model, classes = _load_color_model(species)
    colour_probs = {}
    if scaler is not None:
        label, conf, colour_probs = color_knn_module.predict_ripeness_from_color(crop_bgr, scaler, knn_model, classes)
        result.colour_quality = label
        result.colour_confidence = conf
    else:
        result.colour_quality = None
        result.colour_confidence = 0.0

    if species in DEFECT_SUPPORTED_SPECIES:
        try:
            defect_out = defect_detection.detect_defect(raw_crop_bgr, species)
            # defect_detection.py's helpers mostly return (annotated, mask, percentage);
            # be defensive since this module was built independently and not touched here.
            if isinstance(defect_out, tuple) and len(defect_out) >= 1:
                numeric = [v for v in defect_out if isinstance(v, (int, float))]
                result.defect_percentage = float(numeric[-1]) if numeric else None
            elif isinstance(defect_out, (int, float)):
                result.defect_percentage = float(defect_out)
            else:
                result.defect_note = "Unrecognised return shape from defect_detection"
        except Exception as e:
            result.defect_note = f"defect_detection error: {e}"
        _dbg(f"analyse_fruit: species={species} defect_percentage={result.defect_percentage} "
             f"raw_crop_shape={raw_crop_bgr.shape[:2]} isolated_crop_shape={crop_bgr.shape[:2]}")
    else:
        result.defect_note = f"Not supported for {species} by the current defect module"

    defect_label, defect_conf = None, 0.0
    if species in RIPENESS_RULE_SPECIES:
        try:
            ripeness_out = ripeness_detection.classify_ripeness(
                raw_crop_bgr, species, result.defect_percentage or 0.0
            )
            raw_ripeness = ripeness_out.get("ripeness")
            colour1 = ripeness_out.get("colour1", 0.0)
            colour2 = ripeness_out.get("colour2", 0.0)
            defect_label = _RIPENESS_TO_QUALITY.get(raw_ripeness)
            defect_conf = _defect_ripeness_confidence(raw_ripeness, colour1, colour2, result.defect_percentage)
            result.defect_ripeness = defect_label
            result.defect_ripeness_confidence = defect_conf
            _dbg(f"analyse_fruit: species={species} raw_ripeness={raw_ripeness} "
                 f"colour1(green)={colour1:.2f} colour2(red)={colour2:.2f} "
                 f"defect_percentage={result.defect_percentage} -> defect_label={defect_label}")
        except Exception as e:
            _dbg(f"analyse_fruit: ripeness_detection error: {e}")

    try:
        # Stem detection has two real candidate algorithms (YOLO model and
        # classical HSV/contour heuristics). Run both and keep whichever one
        # produced the higher-confidence detection for this specific fruit,
        # instead of always preferring YOLO first (the old "Automatic" mode).
        #
        # Runs on raw_crop_bgr (NOT the isolated/background-blackened
        # crop_bgr) -- confirmed via ASS's own app.py: its standalone stem
        # tool runs on the plain un-segmented photo, never on an
        # isolated/masked crop. A stem/calyx sits right at the fruit's own
        # edge, so blackening everything outside the contour is exactly the
        # kind of edge artefact that would confuse stem detection, same
        # reasoning already applied to defect_detection/ripeness_detection.
        stem_detector = _load_stem_detector()
        yolo_detections, _elapsed_a, _m1 = stem_detector.detect(raw_crop_bgr, species, method="YOLO")
        traditional_detections, _elapsed_b, _m2 = stem_detector.detect(raw_crop_bgr, species, method="Traditional")

        yolo_best = max((d.confidence for d in yolo_detections), default=0.0)
        traditional_best = max((d.confidence for d in traditional_detections), default=0.0)

        detections = yolo_detections if yolo_best >= traditional_best else traditional_detections
        result.stem_detected = len(detections) > 0
        result.stem_confidence = max((d.confidence for d in detections), default=0.0)
        # Draw the winning method's box/contour onto the same crop it ran
        # on, so the UI can show it right next to the fruit photo -- same
        # idea as StemDetector.annotate() already used by the standalone
        # stem app, just applied per-fruit here. Only when a stem was
        # actually found -- otherwise leave stem_crop blank instead of
        # showing an unannotated duplicate of the fruit photo.
        result.stem_crop = stem_detector.annotate(raw_crop_bgr, detections) if result.stem_detected else None
    except Exception:
        result.stem_detected = False
        result.stem_confidence = 0.0
        result.stem_crop = None

    final_label, final_conf, note = _fuse_quality(
        species, result.colour_quality, result.colour_confidence, colour_probs,
        defect_label, defect_conf, result.defect_percentage, result.stem_detected,
    )
    result.final_quality = final_label
    result.final_quality_confidence = final_conf
    result.quality_note = note

    return result


def _isolate_and_crop(original_bgr, local_contour, shift_x, shift_y, mask_erode=10, extra_erode=6, pad_frac=0.06):
    """
    Reproduces colorDetection.py's crop_object(..., isolate=True): blank out
    every pixel outside this fruit's own contour (so a touching, neighbouring
    fruit doesn't bleed into the crop), then crop tightly to the contour's
    own bounding box with a small pad -- not the original padded YOLO box.
    """
    global_contour = local_contour + [shift_x, shift_y]
    h_img, w_img = original_bgr.shape[:2]
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(full_mask, [global_contour], -1, 255, thickness=cv2.FILLED)
    if mask_erode > 0:
        full_mask = cv2.erode(full_mask, np.ones((mask_erode, mask_erode), np.uint8))
    if extra_erode > 0:
        full_mask = cv2.erode(full_mask, np.ones((extra_erode, extra_erode), np.uint8))

    bx, by, bw, bh = cv2.boundingRect(global_contour)
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    ix0, iy0 = max(0, bx - pad_x), max(0, by - pad_y)
    ix1, iy1 = min(w_img, bx + bw + pad_x), min(h_img, by + bh + pad_y)

    isolated = cv2.bitwise_and(original_bgr, original_bgr, mask=full_mask)
    return isolated[iy0:iy1, ix0:ix1].copy()


DEFECT_NEUTRAL_FILL = (190, 190, 190)


def _neutral_isolate_and_crop(original_bgr, local_contour, shift_x, shift_y, pad_frac=0.06,
                               fill_value=DEFECT_NEUTRAL_FILL):
    """
    Same idea as _isolate_and_crop, but for defect_detection.py/
    ripeness_detection.py instead of the species CNN / colour KNN.

    Those two modules build their own "what counts as fruit" mask via HSV
    thresholds (get_fruit_mask: saturation>25 & value>20, OR a dark-pixel
    catch-all), then fill the LARGEST external contour of that mask solid --
    so if a neighbouring, touching fruit (or the shadow gap between two
    touching fruits) falls inside the same raw rectangular box, it can get
    silently welded into "this fruit"'s mask, and the dark gap/neighbour
    pixels then get flagged as rot/mould, inflating defect% (confirmed via
    FRUIT_DEBUG: a visibly clean strawberry in a 3-strawberry cluster showed
    defect_percentage=88%).

    Blacking out the background (like _isolate_and_crop does for the CNN)
    is not an option here -- that is the exact "isolated crop can make
    healthy peel look defective" problem this project already hit, since
    near-black is itself one of the rot signals those modules look for.

    Instead, fill outside-this-fruit pixels with a bright, LOW-SATURATION
    grey. That grey fails both halves of get_fruit_mask()'s test (needs
    saturation>25 for normal skin, needs value<~110-170 for the dark/rot
    catch-alls), so it is invisible to the fruit-mask step and cannot be
    mistaken for either "more fruit" or "rot" -- it just disappears,
    correctly excluding the neighbour/gap without darkening real edges.
    """
    global_contour = local_contour + [shift_x, shift_y]
    h_img, w_img = original_bgr.shape[:2]
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(full_mask, [global_contour], -1, 255, thickness=cv2.FILLED)

    bx, by, bw, bh = cv2.boundingRect(global_contour)
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    ix0, iy0 = max(0, bx - pad_x), max(0, by - pad_y)
    ix1, iy1 = min(w_img, bx + bw + pad_x), min(h_img, by + bh + pad_y)

    neutral = np.full_like(original_bgr, fill_value, dtype=np.uint8)
    composed = np.where(full_mask[:, :, None] > 0, original_bgr, neutral)
    return composed[iy0:iy1, ix0:ix1].copy()


def run_overall_pipeline(original_bgr, yolo_confidence=YOLO_CONF, yolo_iou=YOLO_IOU):
    """Detect every fruit in the photo (fruit_yolo_v4 + classical-segmentation
    fallback) and analyse each one.

    Returns (results, resized_image) -- resized_image is what every bbox is
    relative to (see DEFAULT_IMAGE_SIZE), so callers must display/draw on
    THIS image, not the original upload, or boxes will be misaligned.
    """
    original_bgr = cv2.resize(original_bgr, DEFAULT_IMAGE_SIZE)

    # Run the teammate's independent morphological/texture module ONCE per
    # photo (it does its own whole-image YOLO pass), so its per-object
    # species guess is available as the third species candidate below AND
    # its geometry/texture numbers can be attached in the same pass --
    # no need to run it a second time at the end.
    morph_objects = None  # None = module unavailable/errored; [] = ran fine, found nothing
    morph_status = None
    if morph_texture_module is None:
        morph_status = f"Morphological module unavailable: {_MORPH_IMPORT_ERROR}"
    else:
        try:
            morph_out = morph_texture_module.inspect_image_morph_texture(original_bgr)
            morph_objects = morph_out.get("objects", [])
            if not morph_objects:
                morph_status = "Morphological module found no matching object in this photo"
        except Exception as e:
            _dbg(f"run_overall_pipeline: morphological module error: {e}")
            morph_status = f"Morphological module error: {e}"

    def _match_morph(bbox_xywh):
        """Best-IoU match for this box against morph_objects, or (None, 0.0)."""
        if not morph_objects:
            return None, 0.0
        best_obj, best_iou = None, 0.0
        for obj in morph_objects:
            iou = _box_iou_xywh_xyxy(bbox_xywh, obj["box"])
            if iou > best_iou:
                best_iou, best_obj = iou, obj
        if best_obj is None or best_iou < MORPH_MATCH_MIN_IOU:
            return None, 0.0
        return best_obj, best_iou

    yolo_model = _load_yolo()
    # agnostic_nms + low conf on purpose: fruit_yolo_v4 only knows Apple/
    # Banana/Orange, so a real Mango/Strawberry can only surface as a WEAK
    # guess in one of those classes. Rejecting low-confidence boxes here
    # would throw away the only localisation evidence choose_cnn_override
    # needs to correct the label later.
    yolo_results = yolo_model.predict(original_bgr, conf=yolo_confidence, iou=yolo_iou, agnostic_nms=True, verbose=False)

    results = []
    claimed_mask = np.zeros(original_bgr.shape[:2], dtype=np.uint8)
    image_area = original_bgr.shape[0] * original_bgr.shape[1]
    accepted_boxes = []  # (x0, y0, x1p, y1p, area) of every box that survived dedup, for the next box's dedup check

    for r in yolo_results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            yolo_label = names.get(cls_id, str(cls_id))
            if yolo_label not in YOLO_FRUIT_CLASS_NAMES:
                continue
            yolo_conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            h_img, w_img = original_bgr.shape[:2]
            pad = 0.08
            bw, bh = x2 - x1, y2 - y1
            x0 = max(0, int(x1 - pad * bw)); y0 = max(0, int(y1 - pad * bh))
            x1p = min(w_img, int(x2 + pad * bw)); y1p = min(h_img, int(y2 + pad * bh))
            if x1p <= x0 or y1p <= y0:
                continue

            # Duplicate/fragment removal: drop a box whose centre falls inside
            # an already-accepted box AND whose area is a small fraction of
            # it -- a smaller echo of the same fruit, not a second fruit.
            box_area = (x1p - x0) * (y1p - y0)
            cx, cy = (x0 + x1p) // 2, (y0 + y1p) // 2
            is_fragment = any(
                ox0 <= cx <= ox1 and oy0 <= cy <= oy1 and box_area < oarea * DUPLICATE_FRAGMENT_MAX_AREA_RATIO
                for ox0, oy0, ox1, oy1, oarea in accepted_boxes
            )
            if is_fragment:
                continue
            if box_area / image_area < MIN_BOX_AREA_RATIO:
                continue
            accepted_boxes.append((x0, y0, x1p, y1p, box_area))

            crop_for_seg = prep.preprocess_image(original_bgr[y0:y1p, x0:x1p], denoise_method="median", enhance_method="none")
            local_mask, local_contour = segmentation_mask_and_contour(crop_for_seg)
            used_fallback_rect = local_contour is None
            if used_fallback_rect:
                ch, cw = crop_for_seg.shape[:2]
                local_contour = np.array([[[0, 0]], [[cw - 1, 0]], [[cw - 1, ch - 1]], [[0, ch - 1]]])
                local_mask = np.full((ch, cw), 255, dtype=np.uint8)
            # A synthetic full-box rectangle's shape says nothing about the
            # real object, so it must NOT be passed to the species shape gate.
            shape_contour = None if used_fallback_rect else local_contour

            # Isolate this fruit's own pixels before classification, exactly like
            # the original colorDetection.py's crop_object(isolate=True): without
            # this, two touching fruits in the same padded box bleed into each
            # other's crop and can flip the CNN's species guess.
            crop = _isolate_and_crop(original_bgr, local_contour, x0, y0)
            raw_crop = original_bgr[y0:y1p, x0:x1p]
            if crop.size == 0:
                crop = raw_crop

            # Defect/ripeness get a neutral-filled crop (real colours inside
            # this fruit's own contour, neutral grey outside) instead of the
            # literal box rectangle, UNLESS the contour is just the synthetic
            # full-box fallback rectangle -- in that case there is no real
            # boundary to isolate against, so the plain box is all we have.
            if used_fallback_rect:
                defect_raw_crop = raw_crop
            else:
                defect_raw_crop = _neutral_isolate_and_crop(original_bgr, local_contour, x0, y0)
                if defect_raw_crop.size == 0:
                    defect_raw_crop = raw_crop

            morph_obj, morph_iou = _match_morph((x0, y0, x1p - x0, y1p - y0))
            if morph_obj is None:
                # Constraint: the morphological/texture module must confirm
                # this box (IoU-matched against its own independent YOLO
                # detection) -- if it found no match here, drop the box
                # entirely instead of showing it as "not matched".
                _dbg(f"run_overall_pipeline: main-loop box ({x0},{y0})-({x1p},{y1p}) yolo_label={yolo_label}"
                     f"({yolo_conf:.3f}) DROPPED -- no morphological/texture match ({morph_status or 'no match'})")
                continue
            # V12 can spatially confirm a box (IoU match) yet still be
            # unsure WHAT it is (fruit_type=None, its own "Unknown" class)
            # -- that still satisfies the "morph must confirm this box
            # exists" gate above, it just contributes no species vote.
            morph_species = morph_obj["fruit_type"].capitalize() if morph_obj.get("fruit_type") else None
            morph_confidence = float(morph_obj["fruit_type_confidence"]) if morph_obj.get("fruit_type") else 0.0

            result = analyse_fruit(crop, yolo_label, yolo_conf, mask=local_mask, contour=shape_contour, raw_crop_bgr=defect_raw_crop,
                                    morph_species=morph_species, morph_confidence=morph_confidence,
                                    morph_obj=morph_obj, morph_iou=morph_iou, morph_status=morph_status)

            # Confidence floors applied AFTER species override, and only if
            # the CNN did NOT move it away from Apple/Banana -- a real Mango/
            # Strawberry showing up as a low-confidence Apple/Banana box must
            # not be thrown away before the CNN gets a chance to correct it.
            # No floor for Orange: a badly rotten orange may score lower.
            # Rejected here (not a confirmed CNN override candidate), so it
            # does NOT claim this region -- the fallback pass may still find
            # a real fruit here.
            #
            # BUG FIX (confirmed via FRUIT_DEBUG on a real 3-mango photo):
            # this must only fire when species is Apple/Banana BECAUSE that
            # is literally YOLO's own uncorrected guess. If a different
            # mechanism (choose_cnn_override, KNOWN_CLASS_OVERRIDE, or the
            # morph candidate winning the 3-way comparison) already moved
            # the species TO Apple/Banana -- e.g. YOLO said Banana(0.57) but
            # the CNN override corrected it to Apple(0.92) -- that route
            # already has its own, separate confidence gate, so re-applying
            # this floor against YOLO's original (pre-override) confidence
            # was silently discarding a correctly-identified fruit. This is
            # exactly how a real 3rd mango vanished: YOLO's own guess for it
            # was Banana(0.57), the CNN override correctly flagged it as
            # Apple(0.92) via known_class override, but the floor below then
            # rejected the whole box because 0.57 < APPLE_MIN_CONF.
            species_is_raw_yolo_guess = result.species == result.yolo_species
            if species_is_raw_yolo_guess and result.species == "Apple" and yolo_conf < APPLE_MIN_CONF:
                continue
            if species_is_raw_yolo_guess and result.species == "Banana" and yolo_conf < BANANA_MIN_CONF:
                continue

            cv2.rectangle(claimed_mask, (x0, y0), (x1p, y1p), 255, thickness=-1)
            result.bbox = (x0, y0, x1p - x0, y1p - y0)
            results.append(result)
            _dbg(f"run_overall_pipeline: main-loop box claimed region=({x0},{y0})-({x1p},{y1p}) "
                 f"yolo_label={yolo_label}({yolo_conf:.3f}) final_species={result.species}")

    # Fallback pass: classical segmentation for fruit YOLO's base weights can't
    # name at all (Mango, Strawberry), mirroring the original system's approach.
    denoised_full = prep.denoise(original_bgr, method="median")
    for blob in segment_all_objects(original_bgr):
        bx, by, bw, bh = blob["bbox"]
        cx, cy = bx + bw // 2, by + bh // 2
        if 0 <= cy < claimed_mask.shape[0] and 0 <= cx < claimed_mask.shape[1] and claimed_mask[cy, cx] > 0:
            # NOTE: this only checks whether this blob's CENTRE POINT falls
            # inside a main-loop box's full RECTANGLE (not that box's real
            # fruit silhouette) -- if one YOLO box happens to span across two
            # real, touching fruits (common when they're the same colour),
            # its rectangle can swallow a second fruit's centre here and
            # silently drop it, even though it was correctly segmented.
            _dbg(f"run_overall_pipeline: fallback blob bbox=({bx},{by},{bw},{bh}) centre=({cx},{cy}) "
                 f"SKIPPED -- centre falls inside an already-claimed main-loop box rectangle")
            continue
        _dbg(f"run_overall_pipeline: fallback blob bbox=({bx},{by},{bw},{bh}) centre=({cx},{cy}) accepted, analysing...")
        # blob["contour"]/["mask"] are already in full-image coordinates.
        # NOTE: unlike the main loop, colorDetection.py's fallback pass does
        # NOT apply the extra 10px mask erosion (that only happens when a
        # fresh mask is drawn from a YOLO-box contour) -- blob["mask"] goes
        # straight into crop_object's isolate step, which only adds its own
        # fixed 6px erosion. Using the main loop's 10+6=16px here over-erodes
        # smaller/concave blobs (e.g. strawberries with a calyx notch),
        # collapsing the mask to near-nothing and making the isolated crop
        # mostly black -- which defect_detection then misreads as rot.
        crop = _isolate_and_crop(original_bgr, blob["contour"], 0, 0, mask_erode=0, extra_erode=6)
        raw_crop = original_bgr[by:by + bh, bx:bx + bw]
        if crop.size == 0:
            crop = raw_crop
        if crop.size == 0:
            continue
        defect_raw_crop = _neutral_isolate_and_crop(original_bgr, blob["contour"], 0, 0)
        if defect_raw_crop.size == 0:
            defect_raw_crop = raw_crop
        blob_roughness = compute_texture_roughness(denoised_full, blob["mask"])
        morph_obj, morph_iou = _match_morph((bx, by, bw, bh))
        if morph_obj is None:
            # Same constraint as the main loop: no morphological/texture
            # match means this box is dropped, not shown as "not matched".
            _dbg(f"run_overall_pipeline: fallback blob bbox=({bx},{by},{bw},{bh}) "
                 f"DROPPED -- no morphological/texture match ({morph_status or 'no match'})")
            continue
        morph_species = morph_obj["fruit_type"].capitalize() if morph_obj.get("fruit_type") else None
        morph_confidence = float(morph_obj["fruit_type_confidence"]) if morph_obj.get("fruit_type") else 0.0
        result = analyse_fruit(crop, "unknown", 0.0, mask=blob["mask"], contour=blob["contour"], roughness=blob_roughness, raw_crop_bgr=defect_raw_crop,
                                morph_species=morph_species, morph_confidence=morph_confidence,
                                morph_obj=morph_obj, morph_iou=morph_iou, morph_status=morph_status)
        if result.species not in FALLBACK_ONLY_SPECIES:
            continue
        result.bbox = (bx, by, bw, bh)
        results.append(result)

    return results, original_bgr


def _box_iou_xywh_xyxy(box_xywh, box_xyxy):
    """IoU between our own (x, y, w, h) bbox and morph_texture_module's
    (x1, y1, x2, y2) bbox -- both are already in the same 512x512 image
    coordinate frame since both run on the same resized `original_bgr`."""
    ax0, ay0, aw, ah = box_xywh
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bx1, by1 = box_xyxy
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


MORPH_MATCH_MIN_IOU = 0.3


# ---------------------------------------------------------------
# Shared display helpers -- used by both app.py (Streamlit UI) and
# report.py (PDF export), so the two never drift apart on box colours.
# ---------------------------------------------------------------
BOX_COLOURS = {"Fresh": (0, 200, 0), "Unripe": (0, 165, 255), "Rotten": (0, 0, 255), "Uncertain": (255, 255, 0)}


def draw_annotations(img_bgr, fruits):
    """Draw each fruit's bbox + 'species: quality' label, colour-coded by
    final_quality, onto a copy of img_bgr (BGR, same frame fr.bbox is in)."""
    annotated = img_bgr.copy()
    for fr in fruits:
        x, y, w, h = [int(round(v)) for v in fr.bbox]
        colour = BOX_COLOURS.get(fr.final_quality, (255, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        label = f"{fr.species or '?'}: {fr.final_quality or '?'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(0, y - th - 8)
        cv2.rectangle(annotated, (x, label_y), (x + tw + 6, label_y + th + 8), colour, -1)
        cv2.putText(annotated, label, (x + 3, label_y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
