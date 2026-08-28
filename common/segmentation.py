

import os

import cv2
import numpy as np

FRUIT_DEBUG = os.environ.get("FRUIT_DEBUG", "0") not in ("0", "", "false", "False")


def _dbg(*args):
    if FRUIT_DEBUG:
        print("[FRUIT_DEBUG]", *args)


SEGMENTATION_CHROMA_THRESHOLD = 15.0   # min LAB a/b distance from background to count as foreground
SEGMENTATION_BORDER_FRAC = 0.04        # fraction of each edge sampled to estimate background color

# These two were tuned by eye at the pipeline's old fixed working
# resolution (whole photo resized to 512x512 before any cropping/
# segmentation ran). They are fixed ABSOLUTE pixel sizes, so once the
# pipeline stopped resizing photos down first and started segmenting at
# the uploaded photo's native resolution (which can be several times
# larger, e.g. a 3000px+ phone photo), a 9px/5px kernel became too small
# relative to the fruit to properly close small chroma gaps (reflections,
# water droplets, seed texture) -- the foreground mask fragmented into
# scattered blobs and cv2.findContours' "largest contour" pick could end
# up being a small fragment instead of the whole fruit (observed in
# practice: an isolated crop collapsing to a tiny sliver on an otherwise
# black background, and colour KNN misreading fresh fruit as Rotten
# because its features were computed on that broken fragment).
# Keeping these as "the tuned size at 512px" and scaling them by the
# actual image's size below reproduces the exact old behaviour for any
# caller still working at ~512px, and generalises correctly for callers
# that now pass full-resolution images.
SEGMENTATION_CLOSE_KSIZE = 9
SEGMENTATION_OPEN_KSIZE = 5
SEGMENTATION_KSIZE_REFERENCE_SIDE = 512.0  # the side length these two were tuned against


def _scaled_kernel_size(image_shape, ksize_at_reference, reference_side=SEGMENTATION_KSIZE_REFERENCE_SIDE, min_ksize=3):
    """Scale a kernel size tuned at `reference_side` px to this image's actual size, staying odd."""
    h, w = image_shape[:2]
    scale = min(h, w) / reference_side
    size = int(round(ksize_at_reference * scale))
    size = max(min_ksize, size)
    if size % 2 == 0:
        size += 1
    return size


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


LEAF_HUE_RANGE = (35, 85)      # OpenCV hue (0-179 scale)
LEAF_MIN_SATURATION = 70
LEAF_MIN_VALUE = 40


def _leaf_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lo, hi = LEAF_HUE_RANGE
    return (h >= lo) & (h <= hi) & (s >= LEAF_MIN_SATURATION) & (v >= LEAF_MIN_VALUE)


def _foreground_contours(
    image,
    chroma_threshold=SEGMENTATION_CHROMA_THRESHOLD,
    border_frac=SEGMENTATION_BORDER_FRAC,
    exclude_leaf_green=False,
):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_a, bg_b = estimate_background_chroma(image, border_frac=border_frac)

    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma_dist = np.sqrt((a - bg_a) ** 2 + (b - bg_b) ** 2)
    mask = (chroma_dist > chroma_threshold).astype(np.uint8) * 255

    if exclude_leaf_green:
        mask[_leaf_green_mask(image)] = 0

    close_ksize = _scaled_kernel_size(image.shape, SEGMENTATION_CLOSE_KSIZE)
    open_ksize = _scaled_kernel_size(image.shape, SEGMENTATION_OPEN_KSIZE)
    close_kernel = np.ones((close_ksize, close_ksize), np.uint8)
    open_kernel = np.ones((open_ksize, open_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours


def segmentation_mask_and_contour(
    image,
    chroma_threshold=SEGMENTATION_CHROMA_THRESHOLD,
    border_frac=SEGMENTATION_BORDER_FRAC,
    exclude_leaf_green=False,
):
    mask, contours = _foreground_contours(image, chroma_threshold, border_frac, exclude_leaf_green=exclude_leaf_green)
    if not contours:
        return mask, None

    largest = max(contours, key=cv2.contourArea)

    solid_mask = np.zeros_like(mask)
    cv2.drawContours(solid_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return solid_mask, largest


DEFAULT_DEFECT_DARKNESS_THRESHOLD = 25.0   # LAB L (0-100) drop below the LOCAL baseline to count a pixel as "defective"
DEFAULT_DEFECT_AREA_FRACTION = 0.10        # fraction of the fruit's own mask that must be flagged before it counts as a real wound (not just noise/shadow/stem) -- forces label to Rotten.
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

    is_defect_u8 = (is_defect * 255).astype(np.uint8)
    is_defect_u8 = cv2.morphologyEx(is_defect_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(is_defect_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_defect_area = max(cv2.contourArea(c) for c in contours)
    return float(largest_defect_area) / float(fg_pixel_count)


DEFAULT_MULTI_OBJECT_MIN_AREA_FRAC = 0.008     # objects smaller than this fraction of the image are dropped as noise.
DEFAULT_MULTI_OBJECT_PEAK_WINDOW_FRAC = 0.05   # local-maxima search window, as a fraction of the image's shorter side -- roughly one fruit's expected radius
DEFAULT_MULTI_OBJECT_MIN_VALUE = 110           # HSV V channel (0-255), averaged over the candidate blob's own
DEFAULT_MULTI_OBJECT_MAX_ASPECT = 2.5          # longer/shorter side of the contour's own rotated bounding rect.
DEFAULT_MULTI_OBJECT_MIN_SOLIDITY = 0.75       # contour_area / convex_hull_area -- real fruit are round/oval and


def contour_shape_metrics(contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contour)
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    aspect = long_side / short_side
    perimeter = cv2.arcLength(contour, closed=True)
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    return solidity, aspect, circularity


def compute_texture_roughness(bgr, mask, edge_margin_frac=0.08):
    fg_all = mask > 0
    if not np.any(fg_all):
        return 0.0

    ys, xs = np.where(fg_all)
    bbox_size = max(int(ys.max()) - int(ys.min()), int(xs.max()) - int(xs.min()), 1)
    margin_px = max(1, int(bbox_size * edge_margin_frac))
    erode_kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8)
    eligible = cv2.erode(mask, erode_kernel) > 0
    if not np.any(eligible):
        eligible = fg_all

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    return float(grad_mag[eligible].mean())


def segment_all_objects(
    image,
    chroma_threshold=SEGMENTATION_CHROMA_THRESHOLD,
    border_frac=SEGMENTATION_BORDER_FRAC,
    min_area_frac=DEFAULT_MULTI_OBJECT_MIN_AREA_FRAC,
    peak_window_frac=DEFAULT_MULTI_OBJECT_PEAK_WINDOW_FRAC,
    min_solidity=DEFAULT_MULTI_OBJECT_MIN_SOLIDITY,
    max_aspect=DEFAULT_MULTI_OBJECT_MAX_ASPECT,
    min_value=DEFAULT_MULTI_OBJECT_MIN_VALUE,
):
    mask, _contours = _foreground_contours(image, chroma_threshold, border_frac)
    value_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]  # for the brightness-plausibility check below

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    win = max(3, int(min(image.shape[:2]) * peak_window_frac) | 1)
    dilated = cv2.dilate(dist, np.ones((win, win), np.uint8))
    min_abs_thresh = 3.0
    local_max = ((dist >= dilated) & (dist > min_abs_thresh)).astype(np.uint8) * 255

    merge_ksize = max(3, int(win * 0.5) | 1)
    local_max = cv2.dilate(local_max, np.ones((merge_ksize, merge_ksize), np.uint8))

    _num_labels, markers = cv2.connectedComponents(local_max)
    markers = markers + 1

    sure_bg = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, local_max)
    markers[unknown == 255] = 0

    color_img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    image_area = image.shape[0] * image.shape[1]
    min_area = min_area_frac * image_area

    objects = []
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

        solidity, aspect, _circularity = contour_shape_metrics(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if solidity < min_solidity or aspect > max_aspect:
            _dbg(f"segment_all_objects: REJECT bbox={(x,y,w,h)} area={area:.0f} "
                 f"solidity={solidity:.3f} (min {min_solidity}) aspect={aspect:.3f} (max {max_aspect})")
            continue

        mean_value = float(value_channel[obj_mask > 0].mean())
        if mean_value < min_value:
            _dbg(f"segment_all_objects: REJECT bbox={(x,y,w,h)} area={area:.0f} "
                 f"mean_value={mean_value:.1f} (min {min_value}) [solidity={solidity:.3f} aspect={aspect:.3f} OK]")
            continue

        _dbg(f"segment_all_objects: ACCEPT bbox={(x,y,w,h)} area={area:.0f} "
             f"solidity={solidity:.3f} aspect={aspect:.3f} mean_value={mean_value:.1f}")
        objects.append({"mask": obj_mask, "contour": contour, "area_px": float(area), "bbox": (x, y, w, h)})

    objects.sort(key=lambda o: o["area_px"], reverse=True)
    return objects

