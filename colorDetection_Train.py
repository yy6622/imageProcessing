"""
colorDetection_Train.py
==============================================
LAB chroma-distance background segmentation — the one piece of this
project's original classical-CV pipeline that's still an active,
live dependency. `colorDetection.py`'s `inspect_image_yolo()` imports
`segmentation_mask_and_contour()` directly, to re-segment the crop
inside each YOLO detection box (getting a real contour/mask, since a
YOLO box is just a rectangle).

Everything else that used to live in this file has been deleted
entirely, not just disconnected, per an explicit decision to keep only
code that's actually called by something:
  - The SVM/RandomForest/KNN classifier training pipeline
    (build_dataset(), build_model()/train_model()/
    train_hierarchical_model()/compare_models(), VotingEnsemble,
    evaluate_majority_vote(), every save_*/load_* model function, the
    CLI) — superseded by YOLOv8 (detection + fruit type) and a CNN
    (train_cnn_quality.py, quality); see colorDetection.py's and
    app.py's module docstrings for that story.
  - The classical multi-fruit detection/splitting algorithms
    (find_object_instances(), the distance-transform + Hough-assisted
    watershed splitter, debris/leaf filtering) — superseded by YOLO's
    own detector, which separates individual fruit without needing any
    of this.
  - The shape/color-histogram/color-moment/LBP-texture feature
    extractors (extract_shape_features(), extract_color_histogram(),
    extract_color_moments(), extract_texture_histogram(),
    extract_patch_feature()) — those fed the SVM classifiers above;
    the CNN learns directly from raw image crops instead and never
    needed them.
None of that is recoverable from this file anymore — it's genuinely
gone, not hidden/disabled. If a report ever needs to reference the
original classical-CV approach (watershed splitting, hand-engineered
features, etc.), that needs pulling from an earlier saved version of
this project rather than from the current file.
==============================================
"""

import cv2
import numpy as np

# segmentation_mask_and_contour()'s parameters — see its docstring for
# why chroma distance (not a fixed HSV threshold) is used.
SEGMENTATION_CHROMA_THRESHOLD = 15.0   # min LAB a/b distance from background to count as foreground
SEGMENTATION_BORDER_FRAC = 0.04        # fraction of each edge sampled to estimate background color


# ======================================================
# segment_fruit — background removal
# ======================================================
def estimate_background_chroma(image, border_frac=SEGMENTATION_BORDER_FRAC):
    """
    Samples a strip along all 4 edges of the image and estimates the
    background's (a, b) chroma in LAB space from it.

    Uses the MODE of the border pixels' (a, b) values (the densest
    cluster in a coarse 2D histogram), not the median — confirmed
    necessary directly on a real photo where the fruit was packed
    edge-to-edge, touching or nearly touching all 4 sides of the frame
    (a tray of apples filling the whole shot, only a sliver of the true
    blue tray visible in a couple of corners). When most of the border
    strip is actually fruit rather than background, the MEDIAN gets
    dragged toward the fruit's own color — measured directly: median
    landed at LAB b=178 on a synthetic reproduction of that photo,
    nowhere near the true background's b=89 — corrupting the "what
    does background look like" estimate the whole segmentation pipeline
    depends on, so it started keeping fruit pixels OUT of the mask
    (mistaking them for background) almost everywhere except the few
    spots that still looked different enough from that wrong estimate.
    The MODE is far more robust to this: even when true background
    pixels are a minority of the border, they form one small, consistent
    color cluster, while the contaminating fruit pixels (varied red/
    green apple hues, highlights, shadow) are comparatively spread out
    and don't win any single histogram bin — confirmed on the same
    synthetic reproduction: mode-based estimate landed at (117, 87),
    within a few units of the true (121, 89), vs. the median's (121, 178).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = image.shape[:2]
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    border_pixels = np.concatenate([
        lab[:bh, :, :].reshape(-1, 3),
        lab[-bh:, :, :].reshape(-1, 3),
        lab[:, :bw, :].reshape(-1, 3),
        lab[:, -bw:, :].reshape(-1, 3),
    ], axis=0)

    a_vals = border_pixels[:, 1]
    b_vals = border_pixels[:, 2]
    hist, a_edges, b_edges = np.histogram2d(a_vals, b_vals, bins=25, range=[[0, 255], [0, 255]])
    idx = np.unravel_index(np.argmax(hist), hist.shape)

    # The mode bin only narrows down roughly WHERE the background
    # cluster is (each bin spans ~10 units of a/b) — precise enough to
    # ignore a large contaminating minority (fruit pixels along the
    # border), but too coarse to use as the final answer on its own,
    # since it would round every estimate to one of 25 fixed positions
    # regardless of the border's actual pixel values. Confirmed this
    # mattered directly: using the raw bin center regressed several
    # already-verified plain-background test photos that used to split
    # correctly. Refining by taking the median of just the border
    # pixels that FELL INTO the winning bin recovers the same precision
    # the old median-of-everything approach had, while keeping the
    # contamination-resistance the histogram mode step provides.
    in_bin = (
        (a_vals >= a_edges[idx[0]]) & (a_vals < a_edges[idx[0] + 1] + 1e-6) &
        (b_vals >= b_edges[idx[1]]) & (b_vals < b_edges[idx[1] + 1] + 1e-6)
    )
    if np.any(in_bin):
        bg_a = float(np.median(a_vals[in_bin]))
        bg_b = float(np.median(b_vals[in_bin]))
    else:
        bg_a = float((a_edges[idx[0]] + a_edges[idx[0] + 1]) / 2)
        bg_b = float((b_edges[idx[1]] + b_edges[idx[1] + 1]) / 2)
    return bg_a, bg_b


def _foreground_contours(
    image,
    chroma_threshold=SEGMENTATION_CHROMA_THRESHOLD,
    border_frac=SEGMENTATION_BORDER_FRAC,
):
    """
    Core mask-building step used by segmentation_mask_and_contour():
    builds the LAB chroma-distance foreground mask (see that function's
    docstring below for why chroma distance, not a fixed HSV threshold)
    and returns it together with EVERY external contour found in it —
    unsorted, unfiltered. segmentation_mask_and_contour() picks just the
    largest one from this.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_a, bg_b = estimate_background_chroma(image, border_frac=border_frac)

    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma_dist = np.sqrt((a - bg_a) ** 2 + (b - bg_b) ** 2)
    mask = (chroma_dist > chroma_threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours


def segmentation_mask_and_contour(
    image,
    chroma_threshold=SEGMENTATION_CHROMA_THRESHOLD,
    border_frac=SEGMENTATION_BORDER_FRAC,
):
    """
    Shared core of the segmentation logic: builds the LAB chroma-distance
    foreground mask and finds its largest contour. This is the function
    colorDetection.py's inspect_image_yolo() imports and calls directly,
    to re-segment the crop inside each YOLO detection box.

    Returns (mask, largest_contour). largest_contour is None if nothing
    was detected.

    Why chroma distance instead of a fixed HSV saturation/value
    threshold: a soft drop-shadow under the fruit is the SAME background
    surface, just darker — it keeps the background's hue/chroma (LAB a,
    b channels), it just has lower lightness (L). A threshold on
    saturation/value can't tell "background, but darker" apart from "a
    genuinely different-colored object" nearly as well, so shadows were
    leaking into the mask. Distance in (a, b) only — deliberately
    ignoring L — treats a darkened patch of background as still
    background, while true fruit colors (red/orange/yellow/green,
    clearly shifted in a/b) still stand out.
    """
    mask, contours = _foreground_contours(image, chroma_threshold, border_frac)
    if not contours:
        return mask, None

    largest = max(contours, key=cv2.contourArea)
    return mask, largest

