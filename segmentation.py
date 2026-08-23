

import cv2
import numpy as np

# segmentation_mask_and_contour()'s parameters — see its docstring for
# why chroma distance (not a fixed HSV threshold) is used.
SEGMENTATION_CHROMA_THRESHOLD = 15.0   # min LAB a/b distance from background to count as foreground
SEGMENTATION_BORDER_FRAC = 0.04        # fraction of each edge sampled to estimate background color

# CLOSE runs with a bigger kernel than OPEN — see _foreground_contours()
# for why: closing needs to bridge internal gaps (a bright specular
# highlight on the fruit's own surface can read as close enough to the
# background's chroma to punch a false hole through the mask) BEFORE
# opening removes small background noise, so opening's small kernel
# doesn't also eat into fine true boundary detail (e.g. a stem).
SEGMENTATION_CLOSE_KSIZE = 9
SEGMENTATION_OPEN_KSIZE = 5


# ======================================================
# segment_fruit — background removal
# ======================================================
def estimate_background_chroma(image, border_frac=SEGMENTATION_BORDER_FRAC):

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

    close_kernel = np.ones((SEGMENTATION_CLOSE_KSIZE, SEGMENTATION_CLOSE_KSIZE), np.uint8)
    open_kernel = np.ones((SEGMENTATION_OPEN_KSIZE, SEGMENTATION_OPEN_KSIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

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

    # Rebuild the mask as a SOLID fill of just the chosen contour,
    # rather than returning the raw thresholded mask. The raw mask can
    # have small internal holes (a bright specular highlight on the
    # fruit's own surface can read as close enough to the background's
    # chroma to slip under the threshold) and can include other,
    # smaller disconnected foreground blobs (background noise, a second
    # fruit's edge clipped into frame) that findContours also picked up
    # but which aren't part of the chosen object. A solid fill of just
    # `largest` guarantees one clean silhouette with no background
    # peeking through the middle, regardless of what the raw threshold
    # happened to miss internally.
    solid_mask = np.zeros_like(mask)
    cv2.drawContours(solid_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return solid_mask, largest


# ======================================================
# Defect/wound-spot detection — a SECOND, targeted color-feature check
# on top of the whole-fruit CNN classifier (see
# colorDetection.classify_quality_cnn). The CNN looks at the whole
# fruit and its own average color, which — confirmed directly with a
# real test photo — can misjudge a fruit that's just naturally a deep/
# dark color throughout (no visible damage) as Rotten, because the
# training set's Fresh examples skew toward brighter red/pink apples.
# This function instead looks for a LOCALIZED dark blemish relative to
# the fruit's OWN median color, which is a much closer match to how a
# person actually judges rot: not "how dark is this fruit overall" but
# "is there a patch here that's darker than the rest of THIS fruit."
# ======================================================
DEFAULT_DEFECT_DARKNESS_THRESHOLD = 25.0   # LAB L (0-100) drop below the LOCAL baseline to count a pixel as "defective"
DEFAULT_DEFECT_AREA_FRACTION = 0.10        # fraction of the fruit's own mask that must be flagged before it counts as a real wound (not just noise/shadow/stem) -- forces label to Rotten.
                                             # Calibrated against real photos, not guessed: several genuinely
                                             # healthy apples' own stem/calyx cavity (a real, normal dark
                                             # anatomical feature every apple has) measured 0.056-0.060 with
                                             # this detector, while a real synthetic rot-sized dark patch
                                             # measured ~0.20 -- 0.10 sits with real margin above the former
                                             # and well below the latter.
DEFAULT_DEFECT_LOW_FRACTION = 0.02         # below this, treat the fruit as having essentially NO visible wound -- used to distrust a CNN "Rotten" call that has no localized evidence behind it
DEFAULT_DEFECT_BLUR_FRAC = 0.35            # local-baseline median-blur kernel size, as a fraction of the fruit's own bounding-box size
DEFAULT_DEFECT_EDGE_MARGIN_FRAC = 0.08     # fraction of the fruit's own bounding-box size to exclude right at the silhouette edge when COUNTING defect pixels


def detect_defect_fraction(
    bgr,
    mask,
    darkness_threshold=DEFAULT_DEFECT_DARKNESS_THRESHOLD,
    blur_frac=DEFAULT_DEFECT_BLUR_FRAC,
    edge_margin_frac=DEFAULT_DEFECT_EDGE_MARGIN_FRAC,
):
    """
    Estimates what fraction of ONE fruit's own surface looks like a
    localized dark blemish (bruise, rot spot, mold) — bruises and rot
    consistently show up as patches distinctly DARKER than their own
    immediate surroundings, regardless of the fruit's base color.

    Compares each pixel to a LOCAL lightness baseline (a large median
    blur of the L channel — the fruit's own general shading, smoothed
    out) rather than either a fixed brightness cutoff or the fruit's
    single overall median. This matters for real glossy round fruit: a
    healthy apple naturally shades from bright (facing the light) to
    noticeably darker toward its silhouette edge ("limb darkening") —
    comparing against one global median flagged that entire natural
    rim as one giant false "wound" on real test photos, since the rim
    can easily be >20% darker than the bright center even with zero
    damage. A local median blur follows that same gradual curve, so the
    gradient itself is never "darker than its own local baseline" — only
    an actual small, abrupt dark patch stands out against it.

    edge_margin_frac additionally excludes a thin ring right at the
    fruit's own silhouette edge from COUNTING toward the result (the
    single darkest part of the limb-darkening gradient, and also where
    the local blur itself gets least reliable from mixing in whatever
    is just outside the mask) — a real wound anywhere else, including
    fairly close to the edge, still gets caught.

    bgr: a denoised (NOT CLAHE-enhanced — local contrast enhancement
    can manufacture or exaggerate false dark patches, which is exactly
    what this function is trying to avoid) crop or full image
    containing the fruit.
    mask: uint8 array the same shape as bgr's first two dims; 255 =
    this fruit's own pixels, 0 = everything else (background, other
    fruit). Typically detection.mask from colorDetection.DetectionResult.

    Returns a float in [0, 1]: fraction of the fruit's own pixels
    flagged as a defect. 0.0 if the mask is empty.
    """
    fg = mask > 0
    fg_pixel_count = int(np.sum(fg))
    if fg_pixel_count == 0:
        return 0.0

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    ys, xs = np.where(fg)
    bbox_size = max(int(ys.max()) - int(ys.min()), int(xs.max()) - int(xs.min()), 1)

    ksize = int(bbox_size * blur_frac)
    ksize = max(9, ksize | 1)  # odd, at least 9
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
    local_baseline = cv2.medianBlur(l_uint8, ksize).astype(np.float32)

    margin_px = max(1, int(bbox_size * edge_margin_frac))
    erode_kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8)
    eligible = cv2.erode(mask, erode_kernel) > 0

    is_defect = eligible & ((local_baseline - l_channel) > darkness_threshold)

    # Use the LARGEST single connected blob's area, not the total
    # flagged-pixel count. Two reasons this matters in practice: (1) it
    # naturally ignores scattered small dark specks (mottling, minor
    # noise, JPEG artifacts) that don't form one real wound, only a real
    # confluent patch counts; (2) because the local-baseline blur above
    # partially "absorbs" a large, solid dark patch into its own
    # baseline (a wide bruise can end up only flagged around its own
    # rim, not its interior — confirmed directly on a synthetic test
    # spot), a contour's ENCLOSED area recovers the patch's true full
    # extent even when only its edge cleared the per-pixel threshold, a
    # plain pixel count would not.
    is_defect_u8 = (is_defect * 255).astype(np.uint8)
    is_defect_u8 = cv2.morphologyEx(is_defect_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(is_defect_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_defect_area = max(cv2.contourArea(c) for c in contours)
    return float(largest_defect_area) / float(fg_pixel_count)


# ======================================================
# Standalone multi-object segmentation — separates EVERY fruit in a
# photo, including ones touching/overlapping in frame.
# ======================================================
DEFAULT_MULTI_OBJECT_MIN_AREA_FRAC = 0.005     # objects smaller than this fraction of the image are dropped as noise
DEFAULT_MULTI_OBJECT_PEAK_WINDOW_FRAC = 0.05   # local-maxima search window, as a fraction of the image's shorter side -- roughly one fruit's expected radius


def segment_all_objects(
    image,
    chroma_threshold=SEGMENTATION_CHROMA_THRESHOLD,
    border_frac=SEGMENTATION_BORDER_FRAC,
    min_area_frac=DEFAULT_MULTI_OBJECT_MIN_AREA_FRAC,
    peak_window_frac=DEFAULT_MULTI_OBJECT_PEAK_WINDOW_FRAC,
):
    """
    Standalone multi-object segmentation: finds and separates EVERY
    distinct fruit in a photo, including fruits that are touching or
    overlapping — segmentation_mask_and_contour() above only ever
    returns the SINGLE largest contour, which merges any touching
    objects into one blob (confirmed directly on a real test photo: 4
    touching apples came back as one silhouette). colorDetection.py's
    actual per-fruit classification pipeline doesn't need this, since
    YOLO already gives it one box per fruit to locally re-segment; this
    is for a standalone "separate everything in this photo" use (e.g.
    a report demo/figure for the Colour Feature Extraction section).

    Approach: build the same LAB chroma-distance foreground mask as
    _foreground_contours(), then split it with a distance-transform +
    watershed — the standard classical-CV technique for separating
    touching round objects (coins, cells, fruit). A point deep inside
    one fruit is far from any background pixel, so the distance
    transform's local peaks mark individual fruit centers even when
    their silhouettes touch; watershed then floods outward from each
    peak and draws a dividing line wherever two floods would otherwise
    meet — right along the touching boundary.

    Returns a list of dicts, one per separated object, LARGEST first:
        {"mask": uint8 HxW mask (255 = this object, matches `image`'s
         size), "contour": ndarray, "area_px": float,
         "bbox": (x, y, w, h)}
    Objects smaller than min_area_frac of the image area are dropped
    as noise.
    """
    mask, _contours = _foreground_contours(image, chroma_threshold, border_frac)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    # LOCAL maxima of the distance transform, not one global threshold
    # relative to the single deepest point in the whole photo — fruit
    # sizes vary from one photo to the next AND within one photo (a
    # smaller apple's own deepest point can be well below a fixed
    # fraction of the largest apple's), so a single global cutoff
    # either misses smaller fruits' peaks entirely or, if loosened
    # enough to catch them, merges everything into one blob (confirmed
    # directly: a 0.15-of-global-max cutoff on an 18-fruit photo found
    # only 1 surviving peak). Finding each pixel that's the maximum
    # within its own local window (a dilate-and-compare trick) gives
    # one peak per fruit regardless of how their individual sizes
    # compare to each other.
    win = max(3, int(min(image.shape[:2]) * peak_window_frac) | 1)
    dilated = cv2.dilate(dist, np.ones((win, win), np.uint8))
    # Small ABSOLUTE floor (pixels), not relative to this photo's own
    # largest object — a relative floor re-creates the exact "small
    # fruit's real peak doesn't clear a threshold set by the biggest
    # fruit" problem this local-maxima approach was meant to fix. This
    # only needs to reject genuine noise (thin slivers, single-pixel
    # artifacts at the mask's own ragged edge), not compete with other
    # fruits' sizes.
    min_abs_thresh = 3.0
    local_max = ((dist >= dilated) & (dist > min_abs_thresh)).astype(np.uint8) * 255

    _num_labels, markers = cv2.connectedComponents(local_max)
    # connectedComponents' background label is 0; shift every marker up
    # by 1 so watershed's reserved "unknown" label (0) stays free, per
    # OpenCV's documented watershed marker convention.
    markers = markers + 1

    sure_bg = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, local_max)
    markers[unknown == 255] = 0

    color_img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    image_area = image.shape[0] * image.shape[1]
    min_area = min_area_frac * image_area

    objects = []
    # Marker labels start at 2 (1 = background, -1 = the watershed
    # boundary line drawn between touching objects) — every label >= 2
    # is one separated object.
    for label in range(2, int(markers.max()) + 1):
        obj_mask = np.zeros(mask.shape, dtype=np.uint8)
        obj_mask[markers == label] = 255
        if int(np.sum(obj_mask > 0)) < min_area:
            continue
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        objects.append({"mask": obj_mask, "contour": contour, "area_px": float(area), "bbox": (x, y, w, h)})

    objects.sort(key=lambda o: o["area_px"], reverse=True)
    return objects

